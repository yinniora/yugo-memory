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
    oversized: bool


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


def _bounded_append(existing: str, addition: str, limit: int = MAX_PENDING_TEXT_CHARS) -> str:
    addition = (addition or "").strip()
    if not addition or len(existing) >= limit:
        return existing
    separator = "\n" if existing else ""
    return (existing + separator + addition)[:limit]


def _text_fragments(value: Any, budget: int = MAX_EVENT_TEXT_CHARS) -> str:
    """Extract human-visible strings from one already-bounded decoded event."""

    parts: list[str] = []
    used = 0

    def visit(item: Any, depth: int = 0) -> None:
        nonlocal used
        if used >= budget or depth > 8:
            return
        if isinstance(item, str):
            remaining = budget - used
            if remaining <= 0:
                return
            piece = item[:remaining]
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
                }:
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
        f"[oversized JSONL event at raw line {line.number}; bounded navigation preview]\n"
        f"{prefix}\n[...middle retained only in raw archive...]\n{suffix}"
    )[:MAX_EVENT_TEXT_CHARS]


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
                oversized=False,
            )
            number += 1
            continue

        prefix = first[:OVERSIZED_PREVIEW_BYTES]
        suffix = bytearray(first[-OVERSIZED_PREVIEW_BYTES:])
        while not first.endswith(b"\n"):
            first = handle.readline(READ_CHUNK_BYTES)
            if not first:
                break
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
            oversized=True,
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
    if row.get("type") != "response_item":
        return ""
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    if payload.get("type") not in {
        "custom_tool_call", "custom_tool_call_output", "function_call", "function_call_output",
    }:
        return ""
    return _text_fragments(payload)


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
                preview = _jsonish_preview(line)
                if b'"type":"compacted"' in line.prefix or b'"type":"context_compacted"' in line.prefix:
                    compactions.append((line.number, preview))
                if state.pending_line_start:
                    state.pending_assistant = _bounded_append(state.pending_assistant, preview)
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
