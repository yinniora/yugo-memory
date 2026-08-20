#!/usr/bin/env python3
"""Ephemeral task continuity and versioned, evidence-linked experience memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_embedding import cosine, decode, embed, encode
from recall_common import extract_terms, normalize_text
from recall_index import adaptive_context_budget, default_paths, read_evidence, search_index


CONTROL_SCHEMA_VERSION = 1
MAX_TASK_ITEMS = 18
MAX_ITEM_CHARS = 420
HISTORY_DEPENDENCY_RE = re.compile(
    r"之前|上次|当时|最早|最近|最新|第\s*\d+\s*(?:次|轮|条)|另一个(?:分支|会话|窗口)|"
    r"previous|earlier|last time|another (?:thread|session|window)|\brecall\b",
    re.IGNORECASE,
)
CONTINUITY_RE = re.compile(
    r"^(?:继续|接着|然后|另外|补充|再|完成|修复|优化|更新|安装|发布)|"
    r"^(?:continue|also|then|next|finish|fix|update|install|publish)\b",
    re.IGNORECASE,
)
REPLACE_RE = re.compile(
    r"新任务|换个任务|改做|不要再做|停止当前|清除当前|重新开始|"
    r"new task|switch task|replace the task|stop the current",
    re.IGNORECASE,
)
CONSTRAINT_RE = re.compile(
    r"必须|禁止|不要|不得|只能|不能|务必|需要|保留|避免|严禁|"
    r"\bmust\b|\bnever\b|\bdo not\b|\bonly\b|\brequire",
    re.IGNORECASE,
)
ACCEPTANCE_RE = re.compile(
    r"确保|验证|测试|通过|完成后|最终|结果|交付|"
    r"\bverify\b|\btest\b|\bpass\b|\bensure\b|\bdeliver",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_control_path() -> Path:
    root = os.environ.get("YUGO_MEMORY_HOME")
    if root:
        base = Path(root).expanduser()
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = (Path(xdg).expanduser() if xdg else Path.home() / ".config") / "yugo-memory"
    return Path(os.environ.get("YUGO_MEMORY_CONTROL_DB", base / "control.sqlite")).expanduser()


def connect_control(path: Path | None = None) -> sqlite3.Connection:
    target = path or default_control_path()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    db = sqlite3.connect(target, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS active_tasks(
          session_id TEXT PRIMARY KEY,
          task_id TEXT NOT NULL,
          objective TEXT NOT NULL,
          objective_embedding BLOB NOT NULL,
          items_json TEXT NOT NULL,
          source_refs_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS experience_revisions(
          experience_key TEXT NOT NULL,
          version INTEGER NOT NULL,
          title TEXT NOT NULL,
          situation TEXT NOT NULL,
          guidance TEXT NOT NULL,
          outcome TEXT NOT NULL,
          tags_json TEXT NOT NULL,
          evidence_refs_json TEXT NOT NULL,
          embedding BLOB NOT NULL,
          state TEXT NOT NULL CHECK(state IN ('active', 'superseded')),
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY(experience_key, version)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS experience_current(
          experience_key TEXT PRIMARY KEY,
          version INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts USING fts5(
          experience_key UNINDEXED,
          version UNINDEXED,
          terms,
          tokenize='unicode61 remove_diacritics 2'
        );
        """
    )
    current = db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
    if current and int(current[0]) != CONTROL_SCHEMA_VERSION:
        db.close()
        raise RuntimeError(
            f"control database schema {current[0]} is incompatible with {CONTROL_SCHEMA_VERSION}"
        )
    db.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES('schema_version',?)",
        (str(CONTROL_SCHEMA_VERSION),),
    )
    db.commit()
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    return db


def control_status(control_path: Path | None = None) -> dict[str, Any]:
    target = control_path or default_control_path()
    db = connect_control(target)
    task_count = db.execute("SELECT count(*) FROM active_tasks").fetchone()[0]
    experience_count = db.execute("SELECT count(*) FROM experience_current").fetchone()[0]
    revision_count = db.execute("SELECT count(*) FROM experience_revisions").fetchone()[0]
    db.close()
    return {
        "ready": True,
        "control_path": str(target),
        "schema_version": CONTROL_SCHEMA_VERSION,
        "active_tasks": task_count,
        "active_experiences": experience_count,
        "experience_revisions": revision_count,
        "task_state_is_ephemeral": True,
    }


def _bounded(value: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _task_id(session_id: str, objective: str) -> str:
    return hashlib.sha256(f"{session_id}\0{utc_now()}\0{objective}".encode()).hexdigest()[:20]


def _item_id(kind: str, text: str) -> str:
    return hashlib.sha256(f"{kind}\0{normalize_text(text)}".encode()).hexdigest()[:16]


def extract_task_items(user_request: str) -> list[dict[str, str]]:
    cleaned = re.sub(r"<[^>]+>.*?</[^>]+>", " ", user_request or "", flags=re.DOTALL)
    clauses = re.split(r"(?:\r?\n+|[。！？!?；;]+)", cleaned)
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for clause in clauses:
        text = _bounded(clause.strip(" -•\t,，"), MAX_ITEM_CHARS)
        if len(text) < 3:
            continue
        kind = "constraint" if CONSTRAINT_RE.search(text) else (
            "acceptance" if ACCEPTANCE_RE.search(text) else "requirement"
        )
        key = _item_id(kind, text)
        if key in seen:
            continue
        seen.add(key)
        result.append({"id": key, "kind": kind, "text": text, "status": "active"})
        if len(result) >= MAX_TASK_ITEMS:
            break
    return result


def _task_payload(row: sqlite3.Row | None, profile: str = "standard") -> dict[str, Any] | None:
    if row is None:
        return None
    items = json.loads(row["items_json"])
    limit = {"minimal": 4, "compact": 8, "standard": 18, "diagnostic": 18}.get(profile, 18)
    return {
        "task_id": row["task_id"],
        "session_id": row["session_id"],
        "objective": row["objective"][: (240 if profile == "minimal" else 600)],
        "items": items[:limit],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "ephemeral": True,
    }


def task_status(session_id: str, control_path: Path | None = None, profile: str = "standard") -> dict[str, Any]:
    db = connect_control(control_path)
    row = db.execute("SELECT * FROM active_tasks WHERE session_id=?", (session_id,)).fetchone()
    db.close()
    return {"active_task": _task_payload(row, profile), "session_id": session_id}


def _normalize_items(items: list[Any] | None, user_request: str) -> list[dict[str, str]]:
    if not items:
        return extract_task_items(user_request)
    result: list[dict[str, str]] = []
    for raw in items[:MAX_TASK_ITEMS]:
        if isinstance(raw, str):
            text = _bounded(raw, MAX_ITEM_CHARS)
            kind = "requirement"
            status = "active"
        elif isinstance(raw, dict):
            text = _bounded(raw.get("text"), MAX_ITEM_CHARS)
            kind = str(raw.get("kind") or "requirement")
            status = str(raw.get("status") or "active")
        else:
            continue
        if not text:
            continue
        if kind not in {"requirement", "constraint", "acceptance", "action"}:
            kind = "requirement"
        if status not in {"active", "done", "dropped"}:
            status = "active"
        result.append({"id": _item_id(kind, text), "kind": kind, "text": text, "status": status})
    return result


def _optimized_objective(explicit_objective: str, items: list[dict[str, str]]) -> str:
    if explicit_objective.strip():
        return _bounded(explicit_objective, 800)
    # Retain a short actionable route, not another copy of the full prompt.
    # Exact constraints and acceptance criteria remain as separately bounded
    # checklist items below.
    preferred = [item["text"] for item in items if item["kind"] == "requirement"]
    if not preferred:
        preferred = [item["text"] for item in items]
    return _bounded("；".join(preferred[:2]), 360)


def _task_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return cosine(embed(left), encode(right))


def sync_task(
    session_id: str,
    user_request: str = "",
    objective: str = "",
    items: list[Any] | None = None,
    action: str = "auto",
    source_refs: list[dict[str, Any]] | None = None,
    control_path: Path | None = None,
) -> dict[str, Any]:
    if action not in {"auto", "start", "replace", "amend", "complete", "cancel", "clear"}:
        raise ValueError("unsupported task action")
    if not session_id.strip():
        raise ValueError("session_id is required")
    db = connect_control(control_path)
    current = db.execute("SELECT * FROM active_tasks WHERE session_id=?", (session_id,)).fetchone()
    if action in {"complete", "cancel", "clear"}:
        removed_task_id = current["task_id"] if current else None
        with db:
            db.execute("DELETE FROM active_tasks WHERE session_id=?", (session_id,))
        db.close()
        return {
            "session_id": session_id,
            "cleared": current is not None,
            "previous_task_id": removed_task_id,
            "reason": action,
        }

    proposed_items = _normalize_items(items, user_request or objective)
    proposed_objective = _optimized_objective(objective, proposed_items)
    if not proposed_objective:
        db.close()
        raise ValueError("user_request or objective is required")
    should_replace = action in {"start", "replace"} or current is None
    reason = "explicit" if action in {"start", "replace", "amend"} else "auto"
    similarity = None
    if current is not None and action == "auto":
        similarity = _task_similarity(current["objective"], proposed_objective)
        should_replace = bool(
            REPLACE_RE.search(user_request)
            or (not CONTINUITY_RE.search(user_request.strip()) and similarity < 0.14)
        )
    if current is not None and action == "amend":
        should_replace = False

    now = utc_now()
    if should_replace:
        task_id = _task_id(session_id, proposed_objective)
        refs = source_refs or []
        with db:
            db.execute("DELETE FROM active_tasks WHERE session_id=?", (session_id,))
            db.execute(
                """INSERT INTO active_tasks(
                     session_id,task_id,objective,objective_embedding,items_json,source_refs_json,
                     created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    session_id, task_id, proposed_objective, encode(proposed_objective),
                    json.dumps(proposed_items, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(refs, ensure_ascii=False, separators=(",", ":")), now, now,
                ),
            )
        transition = "started" if current is None else "replaced"
    else:
        task_id = current["task_id"]
        existing = json.loads(current["items_json"])
        by_id = {item["id"]: item for item in existing}
        for item in proposed_items:
            by_id[item["id"]] = item
        merged = list(by_id.values())[-MAX_TASK_ITEMS:]
        refs = json.loads(current["source_refs_json"])
        for ref in source_refs or []:
            if ref not in refs:
                refs.append(ref)
        with db:
            db.execute(
                "UPDATE active_tasks SET items_json=?,source_refs_json=?,updated_at=? WHERE session_id=?",
                (
                    json.dumps(merged, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(refs[-16:], ensure_ascii=False, separators=(",", ":")),
                    now, session_id,
                ),
            )
        transition = "amended"
    row = db.execute("SELECT * FROM active_tasks WHERE session_id=?", (session_id,)).fetchone()
    db.close()
    return {
        "transition": transition,
        "transition_reason": reason,
        "objective_similarity": round(similarity, 4) if similarity is not None else None,
        "active_task": _task_payload(row),
    }


def _verify_refs(index_path: Path, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for ref in refs:
        archive_path = str(ref.get("archive_path") or "")
        line_start = int(ref.get("line_start") or ref.get("context_line_start") or 0)
        line_end = int(ref.get("line_end") or ref.get("context_line_end") or 0)
        if not archive_path or line_start < 1 or line_end < line_start:
            raise ValueError("each evidence ref needs archive_path and a valid line range")
        # A tiny verified read checks both index membership and raw integrity;
        # no transcript text is retained in the experience database.
        read_evidence(index_path, archive_path, line_start, line_end, 0, 1_000, "text")
        verified.append({
            "archive_path": archive_path,
            "line_start": line_start,
            "line_end": line_end,
        })
    return verified


def manage_experience(
    action: str,
    experience_key: str,
    title: str = "",
    situation: str = "",
    guidance: str = "",
    outcome: str = "",
    tags: list[str] | None = None,
    evidence_refs: list[dict[str, Any]] | None = None,
    source: str = "conversation",
    control_path: Path | None = None,
    index_path: Path | None = None,
) -> dict[str, Any]:
    if action not in {"upsert", "delete"}:
        raise ValueError("experience action must be upsert or delete")
    key = _bounded(experience_key, 120)
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,119}", key):
        raise ValueError("experience_key must be a stable lowercase slug")
    db = connect_control(control_path)
    current = db.execute(
        "SELECT version FROM experience_current WHERE experience_key=?", (key,)
    ).fetchone()
    if action == "delete":
        with db:
            db.execute("DELETE FROM experiences_fts WHERE experience_key=?", (key,))
            db.execute("DELETE FROM experience_current WHERE experience_key=?", (key,))
            db.execute("DELETE FROM experience_revisions WHERE experience_key=?", (key,))
        db.close()
        return {"experience_key": key, "deleted": current is not None, "hard_deleted": True}

    title = _bounded(title, 240)
    situation = _bounded(situation, 1_200)
    guidance = _bounded(guidance, 2_400)
    outcome = _bounded(outcome, 1_200)
    if not title or not situation or not guidance or not outcome:
        db.close()
        raise ValueError("title, situation, guidance, and outcome are required")
    refs = evidence_refs or []
    if source == "conversation":
        if not refs:
            db.close()
            raise ValueError("conversation-derived experience requires raw evidence refs")
        _, resolved_index = default_paths()
        refs = _verify_refs(index_path or resolved_index, refs)
    elif source != "user":
        db.close()
        raise ValueError("source must be conversation or user")
    normalized_tags = sorted({_bounded(tag, 80) for tag in (tags or []) if _bounded(tag, 80)})[:16]
    combined = "\n".join((title, situation, guidance, outcome, " ".join(normalized_tags)))
    version = int(current["version"] + 1) if current else 1
    now = utc_now()
    with db:
        if current:
            db.execute(
                "UPDATE experience_revisions SET state='superseded',updated_at=? "
                "WHERE experience_key=? AND version=?",
                (now, key, current["version"]),
            )
            db.execute(
                "DELETE FROM experiences_fts WHERE experience_key=? AND version=?",
                (key, current["version"]),
            )
        db.execute(
            """INSERT INTO experience_revisions(
                 experience_key,version,title,situation,guidance,outcome,tags_json,
                 evidence_refs_json,embedding,state,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,'active',?,?)""",
            (
                key, version, title, situation, guidance, outcome,
                json.dumps(normalized_tags, ensure_ascii=False, separators=(",", ":")),
                json.dumps(refs, ensure_ascii=False, separators=(",", ":")),
                encode(combined), now, now,
            ),
        )
        db.execute(
            "INSERT OR REPLACE INTO experience_current(experience_key,version) VALUES(?,?)",
            (key, version),
        )
        db.execute(
            "INSERT INTO experiences_fts(experience_key,version,terms) VALUES(?,?,?)",
            (key, version, " ".join(extract_terms(combined, max_terms=1_200))),
        )
    db.close()
    return {
        "experience_key": key,
        "version": version,
        "updated": current is not None,
        "evidence_refs_verified": len(refs),
        "source": source,
    }


def recall_experiences(
    query: str,
    limit: int = 3,
    control_path: Path | None = None,
    profile: str = "standard",
) -> dict[str, Any]:
    db = connect_control(control_path)
    rows = db.execute(
        """SELECT r.* FROM experience_current c
             JOIN experience_revisions r
               ON r.experience_key=c.experience_key AND r.version=c.version"""
    ).fetchall()
    query_terms = set(extract_terms(query, max_terms=96))
    query_vector = embed(query)
    ranked = []
    for row in rows:
        combined = "\n".join((row["title"], row["situation"], row["guidance"], row["outcome"]))
        terms = set(extract_terms(combined, max_terms=1_200))
        lexical = (len(query_terms & terms) / len(query_terms)) if query_terms else 0.0
        semantic = max(0.0, cosine(query_vector, row["embedding"]))
        exact = normalize_text(query) in normalize_text(combined) or normalize_text(query) == row["experience_key"]
        score = (0.62 * lexical) + (0.38 * semantic) + (0.5 if exact else 0.0)
        if score >= 0.12:
            ranked.append((score, row))
    ranked.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
    cap = min(limit, 1 if profile == "minimal" else (2 if profile == "compact" else limit))
    results = []
    for score, row in ranked[:cap]:
        results.append({
            "experience_key": row["experience_key"],
            "version": row["version"],
            "title": row["title"],
            "situation": row["situation"][: (220 if profile in {"minimal", "compact"} else 1_200)],
            "guidance": row["guidance"][: (360 if profile == "minimal" else (700 if profile == "compact" else 2_400))],
            "outcome": row["outcome"][: (220 if profile in {"minimal", "compact"} else 1_200)],
            "tags": json.loads(row["tags_json"]),
            "evidence_refs": json.loads(row["evidence_refs_json"]),
            "score": round(score, 4),
            "summary_is_navigation_not_raw_evidence": True,
        })
    db.close()
    return {
        "query": query,
        "results": results,
        "answerability": "experience_found" if results else "insufficient_evidence",
        "must_verify_conversation_evidence_before_exact_reuse": True,
    }


def prepare_context(
    session_id: str,
    user_request: str,
    current_session_id: str | None = None,
    context_window: int | None = None,
    context_tokens_used: int | None = None,
    include_recall: str = "auto",
    control_path: Path | None = None,
    index_path: Path | None = None,
) -> dict[str, Any]:
    if include_recall not in {"auto", "yes", "no"}:
        raise ValueError("include_recall must be auto, yes, or no")
    archive_root, resolved_index = default_paths()
    target = index_path or resolved_index
    budget = adaptive_context_budget(
        target, current_session_id or session_id, context_window, context_tokens_used, "auto"
    )
    task = sync_task(
        session_id=session_id,
        user_request=user_request,
        action="auto",
        control_path=control_path,
    )
    profile = budget["profile"]
    experience = recall_experiences(user_request, 3, control_path, profile)
    should_recall = include_recall == "yes" or (
        include_recall == "auto" and bool(HISTORY_DEPENDENCY_RE.search(user_request))
    )
    recall = None
    if should_recall:
        if not target.is_file() and not archive_root.is_dir():
            recall = {"answerability": "insufficient_evidence", "safe_to_answer": False}
        else:
            recall = search_index(
                target, user_request, limit=4, mode="auto",
                current_session_id=current_session_id or session_id,
                context_window=context_window,
                context_tokens_used=context_tokens_used,
                response_profile=profile,
            )
    active = task.get("active_task")
    if active:
        item_cap = {"minimal": 4, "compact": 8, "standard": 18}.get(profile, 18)
        active["items"] = active["items"][:item_cap]
    return {
        "context_budget": budget,
        "active_task": active,
        "task_transition": task.get("transition"),
        "experience_memory": experience,
        "conversation_recall": recall,
        "response_contract": {
            "raw_conversation_is_source_of_truth": True,
            "task_ledger_is_ephemeral": True,
            "experience_summaries_require_evidence_verification_for_exact_details": True,
        },
    }


def compact_hint(session_id: str, control_path: Path | None = None) -> str:
    payload = task_status(session_id, control_path, "minimal")["active_task"]
    base = (
        f"Yugo Memory continuity for session {session_id}; current_session_id={session_id}. "
        "Use prepare_context when hidden history or a "
        "multi-step task matters; it budgets output automatically. Verify exact facts with read_evidence."
    )
    if not payload:
        return base
    items = "; ".join(item["text"] for item in payload["items"] if item["status"] == "active")
    return f"{base}\nActive task: {payload['objective']}\nActive instructions: {items}"[:1_600]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status")
    status.add_argument("--session-id")
    clear = sub.add_parser("clear-task")
    clear.add_argument("--session-id", required=True)
    hint = sub.add_parser("compact-hint")
    hint.add_argument("--session-id", required=True)
    args = parser.parse_args()
    if args.command == "status":
        result = control_status()
        if args.session_id:
            result.update(task_status(args.session_id))
    elif args.command == "clear-task":
        result = sync_task(args.session_id, action="clear")
    else:
        result = {"additional_context": compact_hint(args.session_id)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
