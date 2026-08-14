#!/usr/bin/env python3
"""Bounded, incremental parsing of Codex JSONL archives.

The parser never reads an archive or an unbounded JSONL line into one string.
Raw JSONL remains the source of truth; the extracted exchanges are navigation
records used by the local recall index.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from attachment_adapter import (
    attachment_descriptors,
    is_opaque_attachment_key,
    oversized_attachment_descriptors,
)


READ_CHUNK_BYTES = 64 * 1024
MAX_JSON_LINE_BYTES = 8 * 1024 * 1024
MAX_EVENT_TEXT_CHARS = 96_000
MAX_PENDING_TEXT_CHARS = 384_000
OVERSIZED_PREVIEW_BYTES = 64 * 1024
APPROVAL_REVIEW_PREFIX = "The following is the Codex agent history whose request action you are assessing."
SESSION_ID_RE = re.compile(r"([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})", re.IGNORECASE)
BROWSER_CONTEXT_OPEN = "<in-app-browser-" + "context"
DELEGATION_OPEN = "<codex_" + "delegation>"
BROWSER_CONTEXT_RE = re.compile(
    re.escape(BROWSER_CONTEXT_OPEN) + r"\b[^>]*>.*?</in-app-browser-" + r"context>\s*",
    re.IGNORECASE | re.DOTALL,
)
REQUEST_HEADING_RE = re.compile(r"^\s*##\s*My request for Codex:\s*", re.IGNORECASE)
DATA_URL_RE = re.compile(
    r"data:[a-z0-9.+-]+/[a-z0-9.+-]+;base64,[a-z0-9+/=\s]{16,}",
    re.IGNORECASE,
)
OPAQUE_TOKEN_RE = re.compile(r"(?<![a-z0-9+/])[a-z0-9+/]{2048,}={0,2}(?![a-z0-9+/])", re.IGNORECASE)
OPAQUE_HEX_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{4096,}(?![0-9a-f])", re.IGNORECASE)


@dataclass
class RawLine:
    number: int
    byte_offset: int
    next_byte_offset: int
    raw: bytes | None
    prefix: bytes
    suffix: bytes
    prefix_sha256: str
    prefix_length: int
    raw_sha256: str
    raw_bytes: int
    oversized: bool
    lexical_sketch: str = ""


@dataclass
class Exchange:
    exchange_id: str
    archive_path: str
    session_id: str
    timestamp: str
    line_start: int
    line_end: int
    user_message: str
    assistant_message: str
    project: str
    cwd: str
    git_branch: str
    turn_kind: str


@dataclass
class ToolEvidence:
    """Searchable locator for visible tool/code/command evidence in one raw event."""

    archive_path: str
    session_id: str
    timestamp: str
    line_number: int
    ordinal: int
    tool_kind: str
    tool_name: str
    text: str
    raw_sha256: str
    routing_anchors: tuple[str, ...] = ()


@dataclass
class ParserState:
    next_byte_offset: int = 0
    next_line_number: int = 1
    session_id: str = ""
    forked_from_id: str = ""
    project: str = ""
    cwd: str = ""
    git_branch: str = ""
    pending_line_start: int = 0
    pending_timestamp: str = ""
    pending_user: str = ""
    pending_assistant: str = ""
    pending_turn_kind: str = "user"

    @classmethod
    def from_json(cls, value: str | None) -> "ParserState":
        if not value:
            return cls()
        try:
            raw = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return cls()
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: raw[key] for key in allowed if key in raw})

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


def _navigation_safe_text(value: str) -> str:
    """Remove opaque binary encodings while preserving an auditable marker."""

    def marker(match: re.Match[str]) -> str:
        return f"[opaque payload omitted from index: {len(match.group(0))} chars]"

    value = DATA_URL_RE.sub(marker, value)
    value = OPAQUE_TOKEN_RE.sub(marker, value)
    return OPAQUE_HEX_RE.sub(marker, value)


LEXICAL_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9+/=])(?:/[A-Za-z0-9_.@%:+~-]+(?:/[A-Za-z0-9_.@%:+~-]+)*|"
    r"[A-Za-z_][A-Za-z0-9_.:@/+~-]{3,239}|[\u3400-\u9fff]{2,80})(?![A-Za-z0-9+/=])"
)


def _lexical_sketch_update(raw: bytes, terms: dict[str, None], carry: str = "") -> str:
    """Collect bounded exact anchors without retaining an oversized JSON event."""

    decoded = carry + raw.decode("utf-8", errors="ignore")
    for match in LEXICAL_TOKEN_RE.finditer(decoded):
        token = match.group(0)
        if len(token) <= 240 and token not in terms and len(terms) < 8_000:
            terms[token] = None
    return decoded[-512:]


def _bounded_append(existing: str, addition: str, limit: int = MAX_PENDING_TEXT_CHARS) -> str:
    addition = _navigation_safe_text(addition or "").strip()
    if not addition or len(existing) >= limit:
        return existing
    separator = "\n" if existing else ""
    return (existing + separator + addition)[:limit]


def _text_fragments(value: Any, budget: int = MAX_EVENT_TEXT_CHARS) -> str:
    """Extract human-visible strings from one already-bounded decoded event."""

    descriptors = attachment_descriptors(value)
    parts: list[str] = list(descriptors)
    used = sum(len(item) for item in descriptors)

    def visit(item: Any, depth: int = 0) -> None:
        nonlocal used
        if used >= budget or depth > 8:
            return
        if isinstance(item, str):
            remaining = budget - used
            if remaining <= 0:
                return
            piece = _navigation_safe_text(item[:remaining])
            if piece.strip():
                parts.append(piece)
                used += len(piece)
            return
        if isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
                if used >= budget:
                    break
            return
        if isinstance(item, dict):
            preferred = (
                "text", "input_text", "output_text", "message", "content", "output",
                "input", "arguments", "name", "command", "path", "result",
            )
            seen: set[str] = set()
            for key in preferred:
                if key in item:
                    seen.add(key)
                    visit(item[key], depth + 1)
            for key, child in item.items():
                if key not in seen and key not in {
                    "encrypted_content", "internal_chat_message_metadata_passthrough",
                    "images", "local_images", "audio", "local_audio",
                    "type", "role", "id", "status", "phase", "call_id",
                    "annotations", "metadata", "mime_type",
                } and not is_opaque_attachment_key(key):
                    visit(child, depth + 1)
                if used >= budget:
                    break

    visit(value)
    return "\n".join(part for part in parts if part.strip())[:budget]


def _jsonish_preview(line: RawLine) -> str:
    if not line.oversized:
        return ""
    prefix = line.prefix.decode("utf-8", errors="replace")
    suffix = line.suffix.decode("utf-8", errors="replace")
    return (
        f"[oversized JSONL event at raw line {line.number}; bytes={line.raw_bytes}; "
        f"raw_sha256={line.raw_sha256}; bounded navigation preview]\n"
        f"{prefix}\n[...middle retained only in raw archive...]\n{suffix}"
    )[:MAX_EVENT_TEXT_CHARS]


JSON_TEXT_FIELD_RE = re.compile(
    r'"(?:text|input_text|output_text|message|path|file_path|name|filename)"\s*:\s*'
    r'(?P<value>"(?:\\.|[^"\\]){0,48000}")',
    re.DOTALL,
)
JSON_ROLE_RE = re.compile(r'"role"\s*:\s*"(?P<role>user|assistant)"')


def oversized_evidence_text(line: RawLine) -> tuple[str, str]:
    """Recover bounded text and media routing from a too-large JSONL event."""

    prefix = line.prefix.decode("utf-8", errors="replace")
    suffix = line.suffix.decode("utf-8", errors="replace")
    combined = prefix + "\n" + suffix
    role_match = JSON_ROLE_RE.search(prefix)
    role = role_match.group("role") if role_match else "tool"
    parts = oversized_attachment_descriptors(
        line.prefix, line.suffix, line.raw_bytes, line.raw_sha256,
    )
    used = sum(len(item) for item in parts)
    for match in JSON_TEXT_FIELD_RE.finditer(combined):
        try:
            value = json.loads(match.group("value"))
        except json.JSONDecodeError:
            continue
        if isinstance(value, str) and value.strip() and not value.startswith("data:"):
            piece = _navigation_safe_text(value)
            parts.append(piece)
            used += len(piece)
        if used >= MAX_EVENT_TEXT_CHARS:
            break
    if not parts:
        parts.append(_jsonish_preview(line))
    return role, "\n".join(parts)[:MAX_EVENT_TEXT_CHARS]


def iter_bounded_lines(
    handle: BinaryIO,
    start_offset: int = 0,
    start_line: int = 1,
) -> Iterator[RawLine]:
    """Yield JSONL lines while capping per-line memory.

    Oversized lines retain only bounded prefix/suffix previews. Their complete
    bytes remain untouched in the archive and can be paged by read_evidence.
    """

    handle.seek(start_offset)
    number = start_line
    while True:
        offset = handle.tell()
        first = handle.readline(MAX_JSON_LINE_BYTES + 1)
        if not first:
            return
        digest_prefix = first[: min(len(first), 4096)]
        if first.endswith(b"\n") or len(first) <= MAX_JSON_LINE_BYTES:
            yield RawLine(
                number=number,
                byte_offset=offset,
                next_byte_offset=handle.tell(),
                raw=first,
                prefix=first[:OVERSIZED_PREVIEW_BYTES],
                suffix=first[-OVERSIZED_PREVIEW_BYTES:],
                prefix_sha256=hashlib.sha256(digest_prefix).hexdigest(),
                prefix_length=len(digest_prefix),
                raw_sha256=hashlib.sha256(first).hexdigest(),
                raw_bytes=len(first),
                oversized=False,
                lexical_sketch="",
            )
            number += 1
            continue

        prefix = first[:OVERSIZED_PREVIEW_BYTES]
        suffix = bytearray(first[-OVERSIZED_PREVIEW_BYTES:])
        raw_digest = hashlib.sha256(first)
        raw_bytes = len(first)
        lexical_terms: dict[str, None] = {}
        lexical_carry = _lexical_sketch_update(first, lexical_terms)
        while not first.endswith(b"\n"):
            first = handle.readline(READ_CHUNK_BYTES)
            if not first:
                break
            raw_digest.update(first)
            raw_bytes += len(first)
            lexical_carry = _lexical_sketch_update(first, lexical_terms, lexical_carry)
            suffix.extend(first)
            if len(suffix) > OVERSIZED_PREVIEW_BYTES:
                del suffix[:-OVERSIZED_PREVIEW_BYTES]
        yield RawLine(
            number=number,
            byte_offset=offset,
            next_byte_offset=handle.tell(),
            raw=None,
            prefix=prefix,
            suffix=bytes(suffix),
            prefix_sha256=hashlib.sha256(digest_prefix).hexdigest(),
            prefix_length=len(digest_prefix),
            raw_sha256=raw_digest.hexdigest(),
            raw_bytes=raw_bytes,
            oversized=True,
            lexical_sketch=" ".join(lexical_terms)[:MAX_EVENT_TEXT_CHARS],
        )
        number += 1


def _message_role_and_text(row: dict[str, Any]) -> tuple[str | None, str]:
    row_type = row.get("type")
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    if row_type == "response_item" and payload.get("type") == "message":
        role = payload.get("role")
        if role in {"user", "assistant"}:
            return role, _text_fragments(payload.get("content"))
    if row_type == "event_msg" and payload.get("type") == "user_message":
        return "user-fallback", _text_fragments(payload.get("message"))
    if row_type == "event_msg" and payload.get("type") == "agent_message":
        return "assistant-fallback", _text_fragments(payload.get("message"))
    return None, ""


def _clean_user_text(text: str) -> tuple[str, str]:
    value = (text or "").strip()
    if not value:
        return "", "meta"
    if value.startswith("<recommended_plugins>"):
        return "", "meta"
    if value.startswith("<environment_context>") and value.rstrip().endswith("</environment_context>"):
        return "", "meta"
    if value.startswith(DELEGATION_OPEN):
        return value, "delegation"
    value = BROWSER_CONTEXT_RE.sub("", value).strip()
    value = REQUEST_HEADING_RE.sub("", value).strip()
    return value, "user" if value else "meta"


def _tool_text(row: dict[str, Any]) -> str:
    payload, _tool_kind, _tool_name = _visible_tool_payload(row)
    if payload is None:
        return ""
    return _text_fragments(payload)


TOOL_PAYLOAD_TYPES = {
    "custom_tool_call", "custom_tool_call_output", "function_call", "function_call_output",
    "local_shell_call", "local_shell_call_output", "computer_call", "computer_call_output",
    "web_search_call", "web_search_call_output", "mcp_call", "mcp_call_output",
    "apply_patch_call", "apply_patch_call_output",
}
TOOL_SKIP_KEYS = {
    "encrypted_content", "internal_chat_message_metadata_passthrough", "metadata", "images",
    "local_images", "audio", "local_audio", "video", "local_video", "blob", "bytes", "base64",
}
TOOL_CHUNK_CHARS = 12_000
TOOL_NAVIGATION_CHARS = 32_000
TOOL_LONG_FIELD_EDGE_CHARS = 4_000


def _visible_tool_payload(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str, str]:
    if row.get("type") != "response_item":
        return None, "", ""
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    payload_type = str(payload.get("type") or "")
    visible = payload_type in TOOL_PAYLOAD_TYPES or (
        any(token in payload_type for token in ("tool", "function", "shell", "computer", "patch"))
        and any(token in payload_type for token in ("call", "output", "result"))
    )
    if not visible:
        return None, "", ""
    tool_name = str(payload.get("name") or payload.get("tool_name") or payload.get("command_name") or "")
    return payload, payload_type or "tool", tool_name


def _tool_string_chunks(value: Any) -> Iterator[str]:
    """Yield every visible textual tool field; binary and hidden fields stay raw-only."""

    descriptors = attachment_descriptors(value)
    for descriptor in descriptors:
        yield descriptor

    def visit(item: Any, key: str = "", depth: int = 0) -> Iterator[str]:
        if depth > 16:
            return
        if isinstance(item, str):
            if item.startswith("data:") or is_opaque_attachment_key(key):
                return
            safe = _navigation_safe_text(item)
            label = f"{key}: " if key and key not in {"text", "content", "output_text", "input_text"} else ""
            for start in range(0, len(safe), TOOL_CHUNK_CHARS):
                piece = safe[start:start + TOOL_CHUNK_CHARS]
                if piece.strip():
                    yield label + piece
            return
        if isinstance(item, list):
            for child in item:
                yield from visit(child, key, depth + 1)
            return
        if isinstance(item, dict):
            for child_key, child in item.items():
                if child_key in TOOL_SKIP_KEYS or child_key == "encrypted_content" or is_opaque_attachment_key(child_key):
                    continue
                yield from visit(child, child_key, depth + 1)

    yield from visit(value)


def _tool_navigation_text(payload: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Build one bounded locator that still scans every visible textual field."""

    parts = list(attachment_descriptors(payload))
    used = sum(len(part) for part in parts)
    routing_anchors: dict[str, None] = {}

    def add(piece: str) -> None:
        nonlocal used
        if not piece.strip() or used >= TOOL_NAVIGATION_CHARS:
            return
        available = TOOL_NAVIGATION_CHARS - used
        parts.append(piece[:available])
        used += min(len(piece), available)

    def visit(item: Any, key: str = "", depth: int = 0) -> None:
        # Continue scanning after the text preview is full so exact structured
        # anchors from later fields still receive direct-index postings.
        if depth > 16:
            return
        if isinstance(item, str):
            if item.startswith("data:") or is_opaque_attachment_key(key):
                return
            safe = _navigation_safe_text(item)
            for match in LEXICAL_TOKEN_RE.finditer(safe):
                token = match.group(0)
                if (
                    len(routing_anchors) < 20_000
                    and len(token) >= 6
                    and any(char.isdigit() or char in "/_.:@+-" for char in token)
                ):
                    routing_anchors[token] = None
            label = f"{key}: " if key and key not in {"text", "content", "output_text", "input_text"} else ""
            if len(safe) <= TOOL_CHUNK_CHARS:
                add(label + safe)
                return
            digest = hashlib.sha256(safe.encode("utf-8")).hexdigest()
            anchors: dict[str, None] = {}
            for match in LEXICAL_TOKEN_RE.finditer(safe):
                token = match.group(0)
                if token not in anchors and len(anchors) < 1_200:
                    anchors[token] = None
            add(
                f"{label}[long visible tool field chars={len(safe)} sha256={digest}]\n"
                f"{safe[:TOOL_LONG_FIELD_EDGE_CHARS]}\n[...raw middle indexed as lexical anchors...]\n"
                f"{safe[-TOOL_LONG_FIELD_EDGE_CHARS:]}\nlexical anchors: {' '.join(anchors)}"
            )
            return
        if isinstance(item, list):
            for child in item:
                visit(child, key, depth + 1)
            return
        if isinstance(item, dict):
            for child_key, child in item.items():
                if child_key in TOOL_SKIP_KEYS or is_opaque_attachment_key(child_key):
                    continue
                visit(child, child_key, depth + 1)

    visit(payload)
    return (
        "\n".join(part for part in parts if part.strip())[:TOOL_NAVIGATION_CHARS],
        tuple(routing_anchors),
    )


def iter_tool_evidence(path: Path, session_id: str) -> Iterator[ToolEvidence]:
    """Index visible tool calls/results independently from bounded exchange summaries."""

    with path.open("rb") as handle:
        for line in iter_bounded_lines(handle):
            if line.raw is None:
                preview = (line.prefix + line.suffix).decode("utf-8", errors="ignore")
                if not any(token in preview for token in ("tool", "function_call", "shell", "custom_tool", "apply_patch")):
                    continue
                text = (
                    f"[oversized visible tool event raw_line={line.number} bytes={line.raw_bytes} "
                    f"raw_sha256={line.raw_sha256}; exact bytes remain in raw archive]\n"
                    f"lexical anchors: {line.lexical_sketch}"
                )[:MAX_EVENT_TEXT_CHARS]
                if text.strip():
                    yield ToolEvidence(
                        str(path.resolve()), session_id, "", line.number, 0, "oversized_tool", "", text,
                        line.raw_sha256,
                    )
                continue
            try:
                row = json.loads(line.raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(row, dict):
                continue
            payload, tool_kind, tool_name = _visible_tool_payload(row)
            if payload is None:
                continue
            timestamp = str(row.get("timestamp") or "")
            prefix = f"[visible tool evidence kind={tool_kind} name={tool_name or 'unnamed'} raw_line={line.number} raw_sha256={line.raw_sha256}]"
            navigation, routing_anchors = _tool_navigation_text(payload)
            if navigation.strip():
                yield ToolEvidence(
                    str(path.resolve()), session_id, timestamp, line.number, 0, tool_kind, tool_name,
                    prefix + "\n" + navigation, line.raw_sha256, routing_anchors,
                )


def evidence_text_from_row(row: dict[str, Any]) -> tuple[str, str]:
    """Render exact text fields plus verified attachment descriptors from a raw row."""

    role, text = _message_role_and_text(row)
    if role:
        return role.replace("-fallback", ""), text
    payload, _tool_kind, _tool_name = _visible_tool_payload(row)
    if payload is not None:
        return "tool", "\n".join(_tool_string_chunks(payload))
    if row.get("type") == "compacted" or (
        row.get("type") == "event_msg"
        and isinstance(row.get("payload"), dict)
        and row["payload"].get("type") == "context_compacted"
    ):
        return "compaction", _text_fragments(row.get("payload"), budget=40_000)
    if row.get("type") == "session_meta":
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        metadata = {
            key: payload.get(key) for key in ("id", "cwd", "project", "git_branch") if payload.get(key)
        }
        return "session", json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    return "", ""


def _session_metadata(row: dict[str, Any], state: ParserState) -> None:
    if row.get("type") != "session_meta":
        return
    # Fork archives can contain the new fork's session_meta followed by the
    # source task's complete history, including its older session_meta. The
    # outer archive identity is the first record and must never be overwritten.
    if state.session_id:
        return
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    state.session_id = str(payload.get("id") or state.session_id)
    state.forked_from_id = str(payload.get("forked_from_id") or state.forked_from_id)
    state.cwd = str(payload.get("cwd") or state.cwd)
    state.project = str(payload.get("project") or (Path(state.cwd).name if state.cwd else state.project))
    git = payload.get("git") if isinstance(payload.get("git"), dict) else {}
    state.git_branch = str(git.get("branch") or payload.get("git_branch") or state.git_branch)


def _exchange_from_pending(path: Path, state: ParserState, line_end: int) -> Exchange | None:
    if not state.pending_line_start or not state.pending_user.strip():
        return None
    if state.pending_user.lstrip().startswith(APPROVAL_REVIEW_PREFIX):
        return None
    session_id = state.session_id
    if not session_id:
        match = SESSION_ID_RE.search(path.name)
        session_id = match.group(1) if match else path.stem
    exchange_key = f"{path.resolve()}:{state.pending_line_start}"
    return Exchange(
        exchange_id=hashlib.sha256(exchange_key.encode("utf-8")).hexdigest()[:32],
        archive_path=str(path.resolve()),
        session_id=session_id,
        timestamp=state.pending_timestamp,
        line_start=state.pending_line_start,
        line_end=max(state.pending_line_start, line_end),
        user_message=state.pending_user,
        assistant_message=state.pending_assistant,
        project=state.project,
        cwd=state.cwd,
        git_branch=state.git_branch,
        turn_kind=state.pending_turn_kind,
    )


def parse_increment(
    path: Path,
    state: ParserState,
) -> tuple[list[Exchange], ParserState, list[tuple[int, int, str, int]], list[tuple[int, str]], int]:
    """Parse only bytes appended after state.next_byte_offset.

    Returns completed/current exchanges, updated parser state, seek points,
    compaction summaries, and bytes scanned. The final pending exchange is
    returned so callers can upsert it; its state is retained for future append.
    """

    exchanges: list[Exchange] = []
    seek_points: list[tuple[int, int, str, int]] = []
    compactions: list[tuple[int, str]] = []
    last_line = state.next_line_number - 1
    start_offset = state.next_byte_offset
    recent_canonical: tuple[str, str] | None = None

    with path.open("rb") as handle:
        for line in iter_bounded_lines(handle, state.next_byte_offset, state.next_line_number):
            last_line = line.number
            if (line.number - 1) % 128 == 0:
                seek_points.append((line.number, line.byte_offset, line.prefix_sha256, line.prefix_length))
            state.next_byte_offset = line.next_byte_offset
            state.next_line_number = line.number + 1

            if line.raw is None:
                oversized_role, oversized_text = oversized_evidence_text(line)
                if b'"compacted"' in line.prefix or b'"context_compacted"' in line.prefix:
                    compactions.append((line.number, oversized_text))
                if oversized_role == "user":
                    cleaned, turn_kind = _clean_user_text(oversized_text)
                    if turn_kind != "meta":
                        previous = _exchange_from_pending(path, state, line.number - 1)
                        if previous:
                            exchanges.append(previous)
                        state.pending_line_start = line.number
                        state.pending_timestamp = ""
                        state.pending_user = cleaned
                        state.pending_assistant = ""
                        state.pending_turn_kind = turn_kind
                        recent_canonical = ("user", cleaned)
                elif oversized_role == "assistant" and state.pending_line_start:
                    state.pending_assistant = _bounded_append(state.pending_assistant, oversized_text)
                    recent_canonical = ("assistant", oversized_text)
                elif state.pending_line_start:
                    state.pending_assistant = _bounded_append(state.pending_assistant, oversized_text)
                continue
            try:
                row = json.loads(line.raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(row, dict):
                continue
            _session_metadata(row, state)
            timestamp = str(row.get("timestamp") or "")
            role, text = _message_role_and_text(row)

            if role == "user":
                text, turn_kind = _clean_user_text(text)
                if turn_kind == "meta":
                    recent_canonical = ("user", "")
                    continue
                previous = _exchange_from_pending(path, state, line.number - 1)
                if previous:
                    exchanges.append(previous)
                state.pending_line_start = line.number
                state.pending_timestamp = timestamp
                state.pending_user = text
                state.pending_assistant = ""
                state.pending_turn_kind = turn_kind
                recent_canonical = ("user", text)
            elif role == "assistant" and state.pending_line_start:
                state.pending_assistant = _bounded_append(state.pending_assistant, text)
                recent_canonical = ("assistant", text)
            elif role == "user-fallback":
                text, turn_kind = _clean_user_text(text)
                if turn_kind == "meta":
                    recent_canonical = None
                    continue
                if recent_canonical != ("user", text):
                    previous = _exchange_from_pending(path, state, line.number - 1)
                    if previous:
                        exchanges.append(previous)
                    state.pending_line_start = line.number
                    state.pending_timestamp = timestamp
                    state.pending_user = text
                    state.pending_assistant = ""
                    state.pending_turn_kind = turn_kind
                recent_canonical = None
            elif role == "assistant-fallback" and state.pending_line_start:
                if recent_canonical != ("assistant", text):
                    state.pending_assistant = _bounded_append(state.pending_assistant, text)
                recent_canonical = None
            else:
                tool_text = _tool_text(row)
                if tool_text and state.pending_line_start:
                    state.pending_assistant = _bounded_append(state.pending_assistant, tool_text)
                if row.get("type") == "compacted" or (
                    row.get("type") == "event_msg"
                    and isinstance(row.get("payload"), dict)
                    and row["payload"].get("type") == "context_compacted"
                ):
                    summary = _text_fragments(row.get("payload"), budget=40_000)
                    compactions.append((line.number, summary))

    pending = _exchange_from_pending(path, state, last_line)
    if pending:
        exchanges.append(pending)
    return exchanges, state, seek_points, compactions, max(0, state.next_byte_offset - start_offset)
