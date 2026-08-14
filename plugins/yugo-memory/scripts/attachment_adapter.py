#!/usr/bin/env python3
"""Dependency-free attachment metadata extraction for Codex JSONL events.

Binary payloads stay exclusively in the raw Codex archive.  This module emits
small, searchable descriptors that connect an attachment to its surrounding
conversation without copying or interpreting the media bytes.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit


MAX_ATTACHMENTS_PER_EVENT = 24
MAX_REFERENCE_CHARS = 2_048
DATA_URL_PREFIX_RE = re.compile(
    r"^data:(?P<mime>[a-z0-9.+-]+/[a-z0-9.+-]+)(?P<params>(?:;[^,]*)*),(?P<payload>.*)$",
    re.IGNORECASE | re.DOTALL,
)
DATA_URL_MIME_RE = re.compile(
    r"data:(?P<mime>[a-z0-9.+-]+/[a-z0-9.+-]+)(?:;[^,]*)*;base64,",
    re.IGNORECASE,
)
MEDIA_REFERENCE_RE = re.compile(
    r"(?P<reference>(?:(?:https?|file)://|/)[^\s<>'\"\]\[()]{1,1900}?"
    r"\.(?:pdf|png|jpe?g|gif|webp|bmp|tiff?|svg|heic|avif|mp3|wav|m4a|flac|ogg|mp4|mov|mkv|webm|avi|"
    r"docx?|pptx?|xlsx?)(?:[?#][^\s<>'\"\]\[()]*)?)",
    re.IGNORECASE,
)
MEDIA_KEYS = {
    "image_url", "audio_url", "video_url", "file_url", "file_path", "local_path",
    "path", "uri", "url", "name", "filename", "file_name",
}
OPAQUE_KEYS = {"data", "blob", "bytes", "base64", "image", "audio", "video"}
EXTENSION_KIND = {
    ".pdf": ("document", "PDF文档", "application/pdf"),
    ".png": ("image", "图片", "image/png"),
    ".jpg": ("image", "图片", "image/jpeg"),
    ".jpeg": ("image", "图片", "image/jpeg"),
    ".gif": ("image", "图片", "image/gif"),
    ".webp": ("image", "图片", "image/webp"),
    ".bmp": ("image", "图片", "image/bmp"),
    ".tif": ("image", "图片", "image/tiff"),
    ".tiff": ("image", "图片", "image/tiff"),
    ".svg": ("image", "图片", "image/svg+xml"),
    ".heic": ("image", "图片", "image/heic"),
    ".avif": ("image", "图片", "image/avif"),
    ".mp3": ("audio", "音频", "audio/mpeg"),
    ".wav": ("audio", "音频", "audio/wav"),
    ".m4a": ("audio", "音频", "audio/mp4"),
    ".flac": ("audio", "音频", "audio/flac"),
    ".ogg": ("audio", "音频", "audio/ogg"),
    ".mp4": ("video", "视频", "video/mp4"),
    ".mov": ("video", "视频", "video/quicktime"),
    ".mkv": ("video", "视频", "video/x-matroska"),
    ".webm": ("video", "视频", "video/webm"),
    ".avi": ("video", "视频", "video/x-msvideo"),
    ".doc": ("document", "文档", "application/msword"),
    ".docx": ("document", "文档", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ".ppt": ("document", "演示文稿", "application/vnd.ms-powerpoint"),
    ".pptx": ("document", "演示文稿", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    ".xls": ("document", "表格", "application/vnd.ms-excel"),
    ".xlsx": ("document", "表格", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
}


def _kind_for_mime(mime: str) -> tuple[str, str]:
    value = mime.lower()
    if value == "application/pdf" or value.startswith("application/"):
        return "document", "文档"
    if value.startswith("image/"):
        return "image", "图片"
    if value.startswith("audio/"):
        return "audio", "音频"
    if value.startswith("video/"):
        return "video", "视频"
    return "file", "文件"


def _basename(reference: str) -> str:
    value = reference[:MAX_REFERENCE_CHARS]
    parsed = urlsplit(value)
    path = parsed.path if parsed.scheme else value.split("?", 1)[0].split("#", 1)[0]
    return unquote(PurePosixPath(path.replace("\\", "/")).name)[:240]


def _descriptor_for_data_url(value: str, declared_mime: str = "") -> str | None:
    match = DATA_URL_PREFIX_RE.match(value)
    if not match:
        return None
    mime = (declared_mime or match.group("mime")).lower()
    params = match.group("params").lower()
    payload = match.group("payload")
    kind, alias = _kind_for_mime(mime)
    if ";base64" in params:
        compact_length = sum(not char.isspace() for char in payload)
        padding = 2 if payload.rstrip().endswith("==") else 1 if payload.rstrip().endswith("=") else 0
        approximate_bytes = max(0, compact_length * 3 // 4 - padding)
        encoded_digest = hashlib.sha256(payload.encode("ascii", errors="ignore")).hexdigest()[:20]
        return (
            f"[attachment 附件 kind={kind} {alias} mime={mime} storage=inline-base64 "
            f"approx_bytes={approximate_bytes} encoded_sha256={encoded_digest}]"
        )
    digest = hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:20]
    return (
        f"[attachment 附件 kind={kind} {alias} mime={mime} storage=inline-data "
        f"chars={len(payload)} payload_sha256={digest}]"
    )


def _descriptor_for_reference(reference: str, declared_mime: str = "") -> str | None:
    if reference.startswith("data:"):
        return _descriptor_for_data_url(reference, declared_mime)
    name = _basename(reference)
    suffix = PurePosixPath(name.lower()).suffix
    detected = EXTENSION_KIND.get(suffix)
    if not detected and not declared_mime:
        return None
    if declared_mime:
        kind, alias = _kind_for_mime(declared_mime)
        mime = declared_mime.lower()
    else:
        kind, alias, mime = detected  # type: ignore[misc]
    source = "url" if reference.startswith(("http://", "https://")) else "local-reference"
    return (
        f"[attachment 附件 kind={kind} {alias} mime={mime} source={source} "
        f"name={name or 'unnamed'}]"
    )


def attachment_descriptors(value: Any) -> list[str]:
    """Return bounded, deduplicated attachment descriptors for one event."""

    output: list[str] = []
    seen: set[str] = set()

    def add(descriptor: str | None) -> None:
        if descriptor and descriptor not in seen and len(output) < MAX_ATTACHMENTS_PER_EVENT:
            seen.add(descriptor)
            output.append(descriptor)

    def visit(item: Any, depth: int = 0, key: str = "", declared_mime: str = "") -> None:
        if len(output) >= MAX_ATTACHMENTS_PER_EVENT or depth > 10:
            return
        if isinstance(item, str):
            if item.startswith("data:"):
                add(_descriptor_for_data_url(item, declared_mime))
                return
            if key in MEDIA_KEYS:
                add(_descriptor_for_reference(item, declared_mime))
            for match in MEDIA_REFERENCE_RE.finditer(item[:MAX_REFERENCE_CHARS * 4]):
                add(_descriptor_for_reference(match.group("reference")))
            return
        if isinstance(item, list):
            for child in item:
                visit(child, depth + 1, key, declared_mime)
            return
        if not isinstance(item, dict):
            return
        mime = str(item.get("mime_type") or item.get("mimeType") or item.get("media_type") or declared_mime)
        item_type = str(item.get("type") or "").lower()
        attachment_typed = any(token in item_type for token in ("image", "audio", "video", "file", "document", "resource"))
        for child_key, child in item.items():
            child_mime = mime if attachment_typed or child_key in MEDIA_KEYS | OPAQUE_KEYS else ""
            visit(child, depth + 1, child_key, child_mime)

    visit(value)
    return output


def oversized_attachment_descriptors(
    prefix: bytes,
    suffix: bytes,
    raw_event_bytes: int,
    raw_event_sha256: str,
) -> list[str]:
    """Describe media visible in the bounded edges of a very large JSON event."""

    preview = (prefix + b"\n" + suffix).decode("utf-8", errors="replace")
    output: list[str] = []
    seen: set[str] = set()
    for match in DATA_URL_MIME_RE.finditer(preview):
        mime = match.group("mime").lower()
        kind, alias = _kind_for_mime(mime)
        descriptor = (
            f"[attachment 附件 kind={kind} {alias} mime={mime} storage=oversized-inline "
            f"raw_event_bytes={raw_event_bytes} raw_event_sha256={raw_event_sha256[:20]}]"
        )
        if descriptor not in seen:
            seen.add(descriptor)
            output.append(descriptor)
    for match in MEDIA_REFERENCE_RE.finditer(preview):
        descriptor = _descriptor_for_reference(match.group("reference"))
        if descriptor and descriptor not in seen:
            seen.add(descriptor)
            output.append(descriptor)
    return output[:MAX_ATTACHMENTS_PER_EVENT]


def is_opaque_attachment_key(key: str) -> bool:
    """Whether a structured field should be represented only by its descriptor."""

    return key in OPAQUE_KEYS or key in {"image_url", "audio_url", "video_url"}
