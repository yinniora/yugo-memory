#!/usr/bin/env python3
"""Standalone dependency-free stdio MCP server for long-memory recall."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from memory_control import (
    control_status,
    manage_experience,
    prepare_context,
    recall_experiences,
    sync_task,
    task_status,
)
from recall_index import default_paths, index_status, read_evidence, search_index


SERVER_VERSION = "1.4.2"
SESSION_ID_KEYS = (
    "session_id", "sessionId", "thread_id", "threadId", "conversation_id", "conversationId",
)
SESSION_CONTAINER_KEYS = ("session", "thread", "conversation", "context", "client")


def _validated_session_id(value: Any, source: str) -> str | None:
    if value is None or value == "":
        return None
    session_id = str(value).strip()
    if not session_id:
        return None
    if len(session_id) > 256 or any(character.isspace() or ord(character) < 32 for character in session_id):
        raise ValueError(f"invalid session id from {source}")
    return session_id


def _session_id_from_metadata(value: Any, depth: int = 0) -> str | None:
    if not isinstance(value, dict) or depth > 2:
        return None
    for key in SESSION_ID_KEYS:
        if key in value:
            candidate = _validated_session_id(value.get(key), f"MCP metadata {key}")
            if candidate:
                return candidate
    for key in SESSION_CONTAINER_KEYS:
        candidate = _session_id_from_metadata(value.get(key), depth + 1)
        if candidate:
            return candidate
    return None


def resolved_session_id(
    value: Any = None,
    current_value: Any = None,
    *metadata: Any,
    required: bool = True,
) -> str:
    """Resolve a session without guessing another active task.

    Explicit tool arguments win, then exact MCP metadata keys, then process
    environment supplied by an agent host. There is intentionally no fallback to
    the latest task, because that could leak constraints between conversations.
    """

    for source, candidate in (("session_id", value), ("current_session_id", current_value)):
        resolved = _validated_session_id(candidate, source)
        if resolved:
            return resolved
    for candidate in metadata:
        resolved = _session_id_from_metadata(candidate)
        if resolved:
            return resolved
    for key in ("CODEX_THREAD_ID", "QODER_SESSION_ID", "YUGO_MEMORY_SESSION_ID"):
        resolved = _validated_session_id(os.environ.get(key), key)
        if resolved:
            return resolved
    if required:
        raise ValueError(
            "session context unavailable; pass session_id/current_session_id from the current "
            "SessionStart context (Yugo Memory will not guess another session)"
        )
    return ""


def write_message(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "prepare_context",
            "description": (
                "Prepare a context-budgeted continuity packet for a substantive multi-step task. "
                "It automatically starts, amends, or replaces the ephemeral task checklist, recalls "
                "relevant versioned experience, and recalls conversation evidence only when the request "
                "depends on older history. Short standalone requests should skip this tool."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "minLength": 1},
                    "user_request": {"type": "string", "minLength": 2},
                    "current_session_id": {"type": "string"},
                    "context_window": {"type": "integer", "minimum": 1000},
                    "context_tokens_used": {"type": "integer", "minimum": 0},
                    "include_recall": {"type": "string", "enum": ["auto", "yes", "no"], "default": "auto"},
                },
                "required": ["user_request"],
                "additionalProperties": False,
            },
            "annotations": {
                "title": "Prepare Adaptive Memory Context",
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "task_update",
            "description": (
                "Update the ephemeral per-session task objective and optimized instruction checklist. "
                "Minimal auto mode is suitable for each substantive turn: acknowledgements do not mutate, "
                "follow-ups amend, clearly independent tasks replace, and ambiguity preserves the current "
                "objective. Complete, cancel, and clear permanently remove the checklist."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "minLength": 1},
                    "current_session_id": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["auto", "start", "replace", "amend", "complete", "cancel", "clear"],
                        "default": "auto",
                    },
                    "user_request": {"type": "string"},
                    "objective": {"type": "string"},
                    "items": {
                        "type": "array",
                        "maxItems": 18,
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string"},
                                        "kind": {"type": "string", "enum": ["requirement", "constraint", "acceptance", "action"]},
                                        "status": {"type": "string", "enum": ["active", "done", "dropped"]},
                                    },
                                    "required": ["text"],
                                    "additionalProperties": False,
                                },
                            ]
                        },
                    },
                    "source_refs": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                    "profile": {
                        "type": "string", "enum": ["minimal", "compact", "standard"],
                        "default": "minimal",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            "annotations": {
                "title": "Update Active Task Memory",
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "task_status",
            "description": (
                "Read the compact active task objective and instruction checklist for one session. "
                "Use session_id/current_session_id from the hook; exact MCP metadata or an agent-provided "
                "session environment is used only when available, and another task is never guessed."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "minLength": 1},
                    "current_session_id": {"type": "string"},
                    "profile": {"type": "string", "enum": ["minimal", "compact", "standard"], "default": "standard"},
                },
                "required": [],
                "additionalProperties": False,
            },
            "annotations": {
                "title": "Read Active Task Memory",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "experience_manage",
            "description": (
                "Create a new version of a reusable tool/platform workflow experience or permanently "
                "delete it. Conversation-derived experience must include raw evidence ranges, which are "
                "verified before storage; only the compact lesson and locators are stored."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["upsert", "delete"]},
                    "experience_key": {"type": "string", "minLength": 3},
                    "title": {"type": "string"},
                    "situation": {"type": "string"},
                    "guidance": {"type": "string"},
                    "outcome": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 16},
                    "evidence_refs": {"type": "array", "items": {"type": "object"}},
                    "source": {"type": "string", "enum": ["conversation", "user"], "default": "conversation"},
                },
                "required": ["action", "experience_key"],
                "additionalProperties": False,
            },
            "annotations": {
                "title": "Manage Reusable Experience",
                "readOnlyHint": False,
                "destructiveHint": True,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "experience_recall",
            "description": (
                "Recall versioned reusable experience for tools, platforms, and workflows. Returned lessons "
                "are compact navigation; verify linked raw evidence before reusing exact commands or claims."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 2},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 8, "default": 3},
                    "profile": {"type": "string", "enum": ["minimal", "compact", "standard"], "default": "standard"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "annotations": {
                "title": "Recall Reusable Experience",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "recall",
            "description": (
                "Recall exact evidence from long Codex or Qoder conversations with calibrated local hybrid retrieval: "
                "direct anchors, multilingual session/episode routing, multi-facet late interaction, LSH, "
                "sparse graph expansion, diverse evidence sets, deduplication, typed file descriptors, "
                "visible tool/code/command evidence nodes, and raw-line anchors. Use after compaction or whenever older exact "
                "decisions, commands, paths, results, or constraints matter. Read returned raw ranges "
                "with this server's bounded read_evidence tool before relying on details. No upstream "
                "memory plugin or remote server is used."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 2},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
                    "mode": {"type": "string", "enum": ["fast", "auto", "deep"], "default": "auto"},
                    "current_session_id": {"type": "string"},
                    "context_window": {"type": "integer", "minimum": 1000},
                    "context_tokens_used": {"type": "integer", "minimum": 0},
                    "response_profile": {
                        "type": "string",
                        "enum": ["auto", "minimal", "compact", "standard", "diagnostic"],
                        "default": "auto",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "annotations": {
                "title": "Recall Long Conversation Evidence",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "read_evidence",
            "description": (
                "Read a verified evidence range returned by recall without loading the entire conversation "
                "archive. Use view=media for image/PDF/large-attachment turns: user/assistant text and media "
                "tool events are read from raw JSONL while unrelated tool output and binary payloads are omitted. "
                "Use view=text for all visible text fields, including paginated tool commands/code/results, and view=raw "
                "when exact JSONL bytes are required. Paginate with offset_chars and next_offset_chars."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "archive_path": {"type": "string", "minLength": 1},
                    "line_start": {"type": "integer", "minimum": 1},
                    "line_end": {"type": "integer", "minimum": 1},
                    "offset_chars": {"type": "integer", "minimum": 0, "default": 0},
                    "max_chars": {"type": "integer", "minimum": 1000, "maximum": 250000, "default": 60000},
                    "view": {"type": "string", "enum": ["raw", "text", "media"], "default": "raw"},
                },
                "required": ["archive_path", "line_start", "line_end"],
                "additionalProperties": False,
            },
            "annotations": {
                "title": "Read Long Conversation Evidence",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "status",
            "description": "Inspect the private local Yugo Memory index without exposing transcript content.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            "annotations": {
                "title": "Long Memory Recall Status",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
    ]


def result_content(value: Any, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False, indent=2)}],
        "isError": is_error,
    }


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = request.get("method")
    if request_id is None:
        return None
    if method == "initialize":
        requested = (request.get("params") or {}).get("protocolVersion") or "2025-06-18"
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": requested,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "yugo-memory", "version": SERVER_VERSION},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": tool_definitions()}}
    if method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        metadata = (params.get("_meta"), request.get("_meta"))
        _archive_root, target = default_paths()
        try:
            if name == "status":
                value = index_status(target)
                value["continuity"] = control_status()
            elif name == "prepare_context":
                session_id = resolved_session_id(
                    arguments.get("session_id"), arguments.get("current_session_id"), *metadata,
                )
                value = prepare_context(
                    session_id=session_id,
                    user_request=str(arguments.get("user_request") or ""),
                    current_session_id=resolved_session_id(
                        arguments.get("current_session_id"), session_id, *metadata,
                    ),
                    context_window=arguments.get("context_window"),
                    context_tokens_used=arguments.get("context_tokens_used"),
                    include_recall=str(arguments.get("include_recall", "auto")),
                    index_path=target,
                )
            elif name == "task_update":
                value = sync_task(
                    session_id=resolved_session_id(
                        arguments.get("session_id"), arguments.get("current_session_id"), *metadata,
                    ),
                    user_request=str(arguments.get("user_request") or ""),
                    objective=str(arguments.get("objective") or ""),
                    items=arguments.get("items"),
                    action=str(arguments.get("action", "auto")),
                    source_refs=arguments.get("source_refs"),
                    profile=str(arguments.get("profile", "minimal")),
                )
            elif name == "task_status":
                value = task_status(
                    resolved_session_id(
                        arguments.get("session_id"), arguments.get("current_session_id"), *metadata,
                    ),
                    profile=str(arguments.get("profile", "standard")),
                )
            elif name == "experience_manage":
                value = manage_experience(
                    action=str(arguments.get("action") or ""),
                    experience_key=str(arguments.get("experience_key") or ""),
                    title=str(arguments.get("title") or ""),
                    situation=str(arguments.get("situation") or ""),
                    guidance=str(arguments.get("guidance") or ""),
                    outcome=str(arguments.get("outcome") or ""),
                    tags=arguments.get("tags"),
                    evidence_refs=arguments.get("evidence_refs"),
                    source=str(arguments.get("source", "conversation")),
                    index_path=target,
                )
            elif name == "experience_recall":
                value = recall_experiences(
                    str(arguments.get("query") or ""),
                    limit=int(arguments.get("limit", 3)),
                    profile=str(arguments.get("profile", "standard")),
                )
            elif name == "read_evidence":
                if not target.is_file():
                    raise FileNotFoundError(
                        "Yugo Memory index is not ready; no verified evidence can be read yet"
                    )
                value = read_evidence(
                    target,
                    str(arguments.get("archive_path") or ""),
                    int(arguments.get("line_start", 0)),
                    int(arguments.get("line_end", 0)),
                    int(arguments.get("offset_chars", 0)),
                    int(arguments.get("max_chars", 60_000)),
                    str(arguments.get("view", "raw")),
                )
            elif name == "recall":
                query = arguments.get("query")
                if not isinstance(query, str) or len(query.strip()) < 2:
                    raise ValueError("query must contain at least two characters")
                mode = arguments.get("mode", "auto")
                limit = int(arguments.get("limit", 8))
                if not 1 <= limit <= 20:
                    raise ValueError("limit must be between 1 and 20")
                if not target.is_file():
                    value = {
                        "query": query.strip(),
                        "answerability": "index_not_ready",
                        "safe_to_answer": False,
                        "index_snapshot_available": False,
                        "abstention_reason": (
                            "Background memory maintenance has not published an index snapshot yet; "
                            "do not infer an answer from unavailable history."
                        ),
                        "snapshot_policy": "last-committed-nonblocking",
                    }
                else:
                    value = search_index(
                        target, query.strip(), limit=limit, mode=mode,
                        current_session_id=resolved_session_id(
                            arguments.get("current_session_id"), None, *metadata, required=False,
                        ) or None,
                        context_window=arguments.get("context_window"),
                        context_tokens_used=arguments.get("context_tokens_used"),
                        response_profile=arguments.get("response_profile", "auto"),
                    )
            else:
                raise ValueError(f"unknown tool: {name}")
            result = result_content(value)
        except Exception as error:  # MCP should return a tool error, not terminate the server.
            result = result_content({"error": str(error)}, is_error=True)
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> int:
    for raw in sys.stdin:
        if not raw.strip():
            continue
        try:
            request = json.loads(raw)
            response = handle(request)
        except Exception as error:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {error}"},
            }
        if response is not None:
            write_message(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
