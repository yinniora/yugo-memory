#!/usr/bin/env python3
"""Standalone dependency-free stdio MCP server for long-memory recall."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from recall_index import default_paths, index_status, read_evidence, search_index, sync_index


SERVER_VERSION = "1.0.0"


def write_message(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "recall",
            "description": (
                "Recall exact evidence from long Codex conversations with calibrated local hybrid retrieval: "
                "direct anchors, multilingual session/episode routing, multi-facet late interaction, LSH, "
                "sparse graph expansion, diverse evidence sets, deduplication, and raw-line anchors. Use after compaction or whenever older exact "
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
                "Read an exact raw JSONL line range returned by recall without loading the entire "
                "conversation archive. Safe for very large archives; paginate with offset_chars and "
                "next_offset_chars. Only archives already present in the recall index are readable."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "archive_path": {"type": "string", "minLength": 1},
                    "line_start": {"type": "integer", "minimum": 1},
                    "line_end": {"type": "integer", "minimum": 1},
                    "offset_chars": {"type": "integer", "minimum": 0, "default": 0},
                    "max_chars": {"type": "integer", "minimum": 1000, "maximum": 250000, "default": 60000},
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


def ensure_index(archive_root: Path, target: Path) -> None:
    # sync_index is incremental: unchanged archives cost only directory/stat
    # checks, while append-only sessions resume at their stored byte offset.
    sync_index(archive_root, target)


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
        archive_root, target = default_paths()
        try:
            if name == "status":
                value = index_status(target)
            elif name == "read_evidence":
                if not target.is_file():
                    ensure_index(archive_root, target)
                value = read_evidence(
                    target,
                    str(arguments.get("archive_path") or ""),
                    int(arguments.get("line_start", 0)),
                    int(arguments.get("line_end", 0)),
                    int(arguments.get("offset_chars", 0)),
                    int(arguments.get("max_chars", 60_000)),
                )
            elif name == "recall":
                query = arguments.get("query")
                if not isinstance(query, str) or len(query.strip()) < 2:
                    raise ValueError("query must contain at least two characters")
                mode = arguments.get("mode", "auto")
                limit = int(arguments.get("limit", 8))
                if not 1 <= limit <= 20:
                    raise ValueError("limit must be between 1 and 20")
                ensure_index(archive_root, target)
                value = search_index(
                    target, query.strip(), limit=limit, mode=mode,
                    current_session_id=arguments.get("current_session_id"),
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
