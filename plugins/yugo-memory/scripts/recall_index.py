#!/usr/bin/env python3
"""Standalone incremental index and adaptive recall for Codex transcripts."""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from archive_parser import (
    Exchange,
    ParserState,
    evidence_text_from_row,
    iter_tool_evidence,
    iter_bounded_lines,
    oversized_evidence_text,
    parse_increment,
)
from local_embedding import cosine, decode, embed, encode
from recall_common import (
    QueryFeatures,
    anchor_sets,
    content_fingerprint,
    evidence_coverage,
    indexable_anchor,
    normalize_text,
    query_features,
    routing_terms,
    salient_terms,
)
from retrieval_layers import (
    bands_from_simhash,
    exchange_facets,
    facet_payload,
    late_interaction_score,
    lsh_bands,
    query_facets,
)


SCHEMA_VERSION = 13
EPISODE_EXCHANGES = 10
EPISODE_OVERLAP = 2
EXCHANGE_CHARS = 12_000
EXCHANGE_OVERLAP = 800
RRF_K = 32.0
DEFAULT_EVIDENCE_CHARS = 60_000
MAX_EVIDENCE_CHARS = 250_000
STREAM_CHUNK_BYTES = 64 * 1024
MEDIA_TOOL_EVENT_LIMIT = 8
MEDIA_EVENT_TEXT_CHARS = 6_000


def _compact_media_event_text(text: str) -> tuple[str, bool]:
    """Keep exact leading/trailing evidence while bounding one media-view event."""

    if len(text) <= MEDIA_EVENT_TEXT_CHARS:
        return text, False
    half = MEDIA_EVENT_TEXT_CHARS // 2
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    marker = (
        f"\n[media view omitted {len(text) - (half * 2)} middle chars; "
        f"full_text_sha256={digest}; use view=text or view=raw]\n"
    )
    return text[:half] + marker + text[-half:], True


@dataclass
class SearchHit:
    node_id: str
    session_id: str
    archive_path: str
    line_start: int
    line_end: int
    context_line_start: int
    context_line_end: int
    timestamp: str
    score: float
    confidence: str
    routes: list[str]
    term_coverage: float
    concept_coverage: float
    evidence_coverage: float
    local_vector_similarity: float
    late_interaction_similarity: float
    exact_match: bool
    matched_anchors: list[str]
    all_structured_anchors_matched: bool
    covered_query_terms: list[str]
    snippet: str
    content_hash: str


def default_memory_root() -> Path:
    configured = os.environ.get("YUGO_MEMORY_HOME")
    if configured:
        return Path(configured).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "yugo-memory"


def default_paths() -> tuple[Path, Path]:
    root = default_memory_root()
    archive = Path(os.environ.get("YUGO_MEMORY_ARCHIVE_DIR", root / "archives")).expanduser()
    index = Path(os.environ.get("YUGO_MEMORY_INDEX_DB", root / "index.sqlite")).expanduser()
    return archive, index


def connect_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    new_database = not path.exists()
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    if new_database:
        connection.execute("PRAGMA auto_vacuum=INCREMENTAL")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        create_schema(connection)
    except Exception:
        connection.close()
        raise
    return connection


def create_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS indexed_files (
          archive_path TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          source_size INTEGER NOT NULL,
          source_mtime_ns INTEGER NOT NULL,
          next_line_number INTEGER NOT NULL,
          tail_guard_offset INTEGER NOT NULL,
          tail_guard_sha256 TEXT NOT NULL,
          parser_state TEXT NOT NULL,
          exchange_count INTEGER NOT NULL,
          indexed_at TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS archive_seek_points (
          archive_path TEXT NOT NULL,
          line_number INTEGER NOT NULL,
          byte_offset INTEGER NOT NULL,
          prefix_sha256 TEXT NOT NULL,
          prefix_length INTEGER NOT NULL,
          PRIMARY KEY(archive_path, line_number)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_seek_archive_line
          ON archive_seek_points(archive_path, line_number);
        CREATE TABLE IF NOT EXISTS compactions (
          archive_path TEXT NOT NULL,
          line_number INTEGER NOT NULL,
          summary_text TEXT NOT NULL,
          PRIMARY KEY(archive_path, line_number)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS exchanges (
          id TEXT PRIMARY KEY,
          archive_path TEXT NOT NULL,
          session_id TEXT NOT NULL,
          timestamp TEXT NOT NULL,
          line_start INTEGER NOT NULL,
          line_end INTEGER NOT NULL,
          user_message TEXT NOT NULL,
          assistant_message TEXT NOT NULL,
          project TEXT,
          cwd TEXT,
          git_branch TEXT,
          turn_kind TEXT NOT NULL CHECK(turn_kind IN ('user', 'delegation'))
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_exchanges_archive_line
          ON exchanges(archive_path, line_start, line_end);
        CREATE INDEX IF NOT EXISTS idx_exchanges_session_line
          ON exchanges(session_id, line_start, line_end);
        CREATE TABLE IF NOT EXISTS nodes (
          id TEXT PRIMARY KEY,
          level TEXT NOT NULL CHECK(level IN ('session', 'episode', 'exchange')),
          session_id TEXT NOT NULL,
          archive_path TEXT NOT NULL,
          parent_id TEXT,
          line_start INTEGER NOT NULL,
          line_end INTEGER NOT NULL,
          timestamp TEXT NOT NULL,
          project TEXT,
          cwd TEXT,
          git_branch TEXT,
          text TEXT NOT NULL,
          terms TEXT NOT NULL,
          embedding BLOB NOT NULL,
          content_hash TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_nodes_level ON nodes(level);
        CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(session_id, line_start, line_end);
        CREATE INDEX IF NOT EXISTS idx_nodes_archive ON nodes(archive_path, line_start, line_end);
        CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
        CREATE INDEX IF NOT EXISTS idx_nodes_hash ON nodes(content_hash);
        CREATE TABLE IF NOT EXISTS node_facets (
          node_id TEXT NOT NULL,
          facet_kind TEXT NOT NULL,
          facet_ordinal INTEGER NOT NULL,
          embedding BLOB NOT NULL,
          simhash INTEGER NOT NULL,
          PRIMARY KEY(node_id, facet_ordinal)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_facets_node ON node_facets(node_id);
        CREATE TABLE IF NOT EXISTS node_lsh (
          band TEXT NOT NULL,
          node_id TEXT NOT NULL,
          facet_ordinal INTEGER NOT NULL,
          PRIMARY KEY(band, node_id, facet_ordinal)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_lsh_band ON node_lsh(band);
        CREATE TABLE IF NOT EXISTS node_anchors (
          anchor TEXT NOT NULL,
          node_id TEXT NOT NULL,
          decisive INTEGER NOT NULL,
          PRIMARY KEY(anchor, node_id)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_anchors_node ON node_anchors(node_id);
        CREATE TABLE IF NOT EXISTS node_edges (
          source_id TEXT NOT NULL,
          target_id TEXT NOT NULL,
          relation TEXT NOT NULL,
          weight REAL NOT NULL,
          PRIMARY KEY(source_id, target_id, relation)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_edges_source ON node_edges(source_id);
        CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
          node_id UNINDEXED,
          terms,
          tokenize='unicode61 remove_diacritics 2'
        );
        """
    )
    current = db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
    if current and int(current[0]) != SCHEMA_VERSION:
        raise RuntimeError(f"recall index schema {current[0]} is incompatible with {SCHEMA_VERSION}; rebuild it")
    db.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    db.commit()


def _chunk_text(text: str, limit: int = EXCHANGE_CHARS, overlap: int = EXCHANGE_OVERLAP) -> Iterable[str]:
    if len(text) <= limit:
        yield text
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + limit)
        yield text[start:end]
        if end == len(text):
            return
        start = max(start + 1, end - overlap)


def _node(
    node_id: str,
    level: str,
    session_id: str,
    archive_path: str,
    parent_id: str | None,
    line_start: int,
    line_end: int,
    timestamp: str,
    project: str,
    cwd: str,
    git_branch: str,
    text: str,
    terms_limit: int = 1600,
) -> tuple[Any, ...]:
    terms = " ".join(routing_terms(text, max_terms=terms_limit))
    return (
        node_id, level, session_id, archive_path, parent_id, line_start, line_end,
        timestamp, project, cwd, git_branch, text, terms, encode(text), content_fingerprint(text),
    )


def _replace_file_nodes(
    db: sqlite3.Connection,
    archive_path: str,
    nodes: list[tuple[Any, ...]],
    facets: list[tuple[Any, ...]] | None = None,
    anchors: list[tuple[Any, ...]] | None = None,
) -> None:
    old_ids = [row[0] for row in db.execute("SELECT id FROM nodes WHERE archive_path=?", (archive_path,))]
    if old_ids:
        placeholders = ",".join("?" for _ in old_ids)
        db.execute(f"DELETE FROM node_edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})", (*old_ids, *old_ids))
        db.executemany("DELETE FROM node_lsh WHERE node_id=?", ((node_id,) for node_id in old_ids))
        db.executemany("DELETE FROM node_facets WHERE node_id=?", ((node_id,) for node_id in old_ids))
        db.executemany("DELETE FROM node_anchors WHERE node_id=?", ((node_id,) for node_id in old_ids))
        db.executemany("DELETE FROM nodes_fts WHERE node_id=?", ((node_id,) for node_id in old_ids))
        db.execute("DELETE FROM nodes WHERE archive_path=?", (archive_path,))
    if not nodes:
        return
    db.executemany(
        """INSERT INTO nodes(
          id, level, session_id, archive_path, parent_id, line_start, line_end,
          timestamp, project, cwd, git_branch, text, terms, embedding, content_hash
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        nodes,
    )
    db.executemany(
        "INSERT INTO nodes_fts(node_id, terms) VALUES(?, ?)",
        ((row[0], row[12]) for row in nodes),
    )
    facets = facets or []
    anchors = anchors or []
    db.executemany(
        """INSERT INTO node_facets(
             node_id, facet_kind, facet_ordinal, embedding, simhash
           ) VALUES(?, ?, ?, ?, ?)""",
        facets,
    )
    lsh_rows = []
    for row in nodes:
        if row[1] == "exchange":
            for band in lsh_bands(decode(row[13])):
                lsh_rows.append((band, row[0], -1))
    for node_id, _, facet_ordinal, _embedding_blob, signature in facets:
        for band in bands_from_simhash(signature):
            lsh_rows.append((band, node_id, facet_ordinal))
    db.executemany("INSERT INTO node_lsh(band, node_id, facet_ordinal) VALUES(?, ?, ?)", lsh_rows)
    db.executemany(
        "INSERT INTO node_anchors(anchor, node_id, decisive) VALUES(?, ?, ?)",
        anchors,
    )


def _upsert_exchanges(db: sqlite3.Connection, exchanges: list[Exchange]) -> None:
    db.executemany(
        """INSERT OR REPLACE INTO exchanges(
          id, archive_path, session_id, timestamp, line_start, line_end,
          user_message, assistant_message, project, cwd, git_branch, turn_kind
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            (
                row.exchange_id, row.archive_path, row.session_id, row.timestamp,
                row.line_start, row.line_end, row.user_message, row.assistant_message,
                row.project, row.cwd, row.git_branch, row.turn_kind,
            )
            for row in exchanges
        ),
    )


def _episode_groups(rows: list[sqlite3.Row], compact_lines: list[int]) -> list[list[sqlite3.Row]]:
    boundary_groups: list[list[sqlite3.Row]] = []
    current: list[sqlite3.Row] = []
    boundary_index = 0
    for row in rows:
        while boundary_index < len(compact_lines) and row["line_start"] > compact_lines[boundary_index]:
            if current:
                boundary_groups.append(current)
                current = []
            boundary_index += 1
        current.append(row)
    if current:
        boundary_groups.append(current)

    episodes: list[list[sqlite3.Row]] = []
    step = max(1, EPISODE_EXCHANGES - EPISODE_OVERLAP)
    for group in boundary_groups:
        for start in range(0, len(group), step):
            chunk = group[start:start + EPISODE_EXCHANGES]
            if chunk:
                episodes.append(chunk)
            if start + EPISODE_EXCHANGES >= len(group):
                break
    return episodes


def _build_file_nodes(
    db: sqlite3.Connection,
    archive_path: str,
    forked_from_id: str = "",
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    rows = db.execute(
        "SELECT * FROM exchanges WHERE archive_path=? ORDER BY line_start", (archive_path,)
    ).fetchall()
    if not rows:
        return [], [], []
    route_rows = rows
    inherited_count = 0
    if forked_from_id:
        parent_rows = db.execute(
            """SELECT user_message, assistant_message FROM exchanges
                 WHERE session_id=? AND archive_path<>?""",
            (forked_from_id, archive_path),
        ).fetchall()
        parent_hashes = {content_fingerprint(row["user_message"]) for row in parent_rows}
        unique_rows = [
            row for row in rows
            if content_fingerprint(row["user_message"]) not in parent_hashes
        ]
        if unique_rows:
            route_rows = unique_rows
            inherited_count = len(rows) - len(unique_rows)

    compact_rows = db.execute(
        "SELECT line_number, summary_text FROM compactions WHERE archive_path=? ORDER BY line_number",
        (archive_path,),
    ).fetchall()
    compact_lines = [row["line_number"] for row in compact_rows]
    route_compact_rows = compact_rows
    if forked_from_id and route_rows:
        route_compact_rows = [
            row for row in compact_rows if row["line_number"] >= route_rows[0]["line_start"]
        ]
    route_compact_lines = [row["line_number"] for row in route_compact_rows]
    session_id = rows[0]["session_id"]
    archive_key = content_fingerprint(archive_path)[:12]
    session_node_id = f"session:{session_id}:{archive_key}"
    prompts = [row["user_message"] for row in route_rows]
    route_prompts = prompts[:4] + prompts[-4:] + prompts[4:-4:12]
    summary_text = "\n".join(
        row["summary_text"][:12_000] for row in route_compact_rows[-8:] if row["summary_text"]
    )
    top_terms = salient_terms(
        prompts + [row["assistant_message"] for row in route_rows], limit=420,
    )
    session_text = "\n".join(
        part for part in (
            f"Project: {rows[-1]['project'] or ''}",
            f"Working directory: {rows[-1]['cwd'] or ''}",
            f"Git branch: {rows[-1]['git_branch'] or ''}",
            f"Compaction epochs: {len(compact_lines) + 1}",
            f"Forked from: {forked_from_id}" if forked_from_id else "",
            f"Inherited exchanges excluded from session routing: {inherited_count}" if inherited_count else "",
            summary_text,
            "\n".join(route_prompts),
            "Key terms: " + " ".join(top_terms),
        ) if part
    )[:180_000]
    nodes: list[tuple[Any, ...]] = [
        _node(
            session_node_id, "session", session_id, archive_path, None,
            rows[0]["line_start"], rows[-1]["line_end"], rows[-1]["timestamp"],
            rows[-1]["project"] or "", rows[-1]["cwd"] or "",
            rows[-1]["git_branch"] or "", session_text, 2400,
        )
    ]

    for episode_number, episode in enumerate(_episode_groups(route_rows, route_compact_lines)):
        episode_id = f"episode:{session_id}:{archive_key}:{episode_number:05d}"
        episode_text = "\n\n".join(
            f"User: {row['user_message'][:3000]}\nAssistant: {row['assistant_message'][:5000]}"
            for row in episode
        )
        nodes.append(
            _node(
                episode_id, "episode", session_id, archive_path, session_node_id,
                episode[0]["line_start"], episode[-1]["line_end"], episode[-1]["timestamp"],
                episode[-1]["project"] or "", episode[-1]["cwd"] or "",
                episode[-1]["git_branch"] or "", episode_text, 1800,
            )
        )

    facets: list[tuple[Any, ...]] = []
    anchors: list[tuple[Any, ...]] = []
    for row in rows:
        combined = f"User: {row['user_message']}\n\nAssistant: {row['assistant_message']}"
        for part_number, part in enumerate(_chunk_text(combined)):
            node_id = f"exchange:{row['id']}:{part_number:03d}"
            node_tuple = _node(
                node_id, "exchange", row["session_id"],
                archive_path, None, row["line_start"], row["line_end"], row["timestamp"],
                row["project"] or "", row["cwd"] or "", row["git_branch"] or "", part,
            )
            nodes.append(node_tuple)
            facet_source = (
                exchange_facets(row["user_message"], row["assistant_message"], part)
                if part_number == 0 else []
            )
            for kind, ordinal, embedding_blob, simhash in facet_payload(facet_source):
                facets.append((node_id, kind, ordinal, embedding_blob, simhash))
            all_anchors, decisive_anchors = anchor_sets(part)
            decisive = set(decisive_anchors)
            stable_anchors = [
                anchor for anchor in all_anchors
                if anchor in decisive and indexable_anchor(anchor)
            ][:128]
            anchors.extend(
                (anchor, node_id, 1)
                for anchor in stable_anchors
            )
    # Tool calls/results are separate navigation nodes. This prevents a long
    # command, patch, log, or tool result from disappearing behind the compact
    # assistant-message budget while the raw JSONL remains the source of truth.
    for evidence in iter_tool_evidence(Path(archive_path), session_id):
        parent = next(
            (
                row for row in reversed(rows)
                if row["line_start"] <= evidence.line_number <= row["line_end"]
            ),
            rows[-1],
        )
        digest = hashlib.sha256(
            f"{archive_path}:{evidence.line_number}:{evidence.ordinal}:{evidence.raw_sha256}".encode("utf-8")
        ).hexdigest()[:32]
        node_id = f"exchange:tool:{digest}"
        node_tuple = _node(
            node_id, "exchange", session_id, archive_path,
            f"exchange:{parent['id']}:000", evidence.line_number, evidence.line_number,
            evidence.timestamp or parent["timestamp"], parent["project"] or "",
            parent["cwd"] or "", parent["git_branch"] or "", evidence.text, 2200,
        )
        nodes.append(node_tuple)
        all_anchors, decisive_anchors = anchor_sets(evidence.text)
        decisive = set(decisive_anchors)
        direct_tool_anchors = {
            normalize_text(anchor) for anchor in evidence.routing_anchors
        }
        combined_tool_anchors = {
            anchor for anchor in all_anchors
            if anchor in decisive and indexable_anchor(anchor)
        } | {
            anchor for anchor in direct_tool_anchors
            if indexable_anchor(anchor)
        }
        anchors.extend(
            (anchor, node_id, 1)
            for anchor in combined_tool_anchors
        )
    return nodes, facets, anchors


def _refresh_graph(db: sqlite3.Connection) -> int:
    """Rebuild a sparse deterministic graph from temporal and exact-anchor links."""

    db.execute("DELETE FROM node_edges")
    edges: set[tuple[str, str, str, float]] = set()
    rows = db.execute(
        """SELECT id, session_id, line_start FROM nodes
             WHERE level='exchange' ORDER BY session_id, line_start, id"""
    ).fetchall()
    by_session: defaultdict[str, list[sqlite3.Row]] = defaultdict(list)
    by_exchange: defaultdict[tuple[str, int], list[str]] = defaultdict(list)
    for row in rows:
        by_session[row["session_id"]].append(row)
        by_exchange[(row["session_id"], row["line_start"])].append(row["id"])

    for part_ids in by_exchange.values():
        for left, right in zip(part_ids, part_ids[1:]):
            edges.add((left, right, "same-exchange", 1.0))
            edges.add((right, left, "same-exchange", 1.0))
    for session_rows in by_session.values():
        primary = []
        seen_lines: set[int] = set()
        for row in session_rows:
            if row["line_start"] not in seen_lines:
                primary.append(row["id"])
                seen_lines.add(row["line_start"])
        for left, right in zip(primary, primary[1:]):
            edges.add((left, right, "temporal", 0.62))
            edges.add((right, left, "temporal", 0.62))

    anchor_groups = db.execute(
        """SELECT anchor, group_concat(node_id) AS node_ids, count(*) AS count
             FROM node_anchors WHERE decisive=1
             GROUP BY anchor HAVING count(*) BETWEEN 2 AND 24"""
    ).fetchall()
    for group in anchor_groups:
        node_ids = list(dict.fromkeys(group["node_ids"].split(",")))
        hub = node_ids[0]
        for target in node_ids[1:]:
            edges.add((hub, target, "shared-anchor", 0.92))
            edges.add((target, hub, "shared-anchor", 0.92))
    db.executemany(
        "INSERT INTO node_edges(source_id, target_id, relation, weight) VALUES(?, ?, ?, ?)",
        sorted(edges),
    )
    return len(edges)


def _guard(path: Path, source_size: int) -> tuple[int, str]:
    length = min(4096, source_size)
    offset = max(0, source_size - length)
    with path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(length)
    return offset, hashlib.sha256(raw).hexdigest()


def _append_only(path: Path, previous: sqlite3.Row) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= previous["source_size"]:
        return False
    length = previous["source_size"] - previous["tail_guard_offset"]
    with path.open("rb") as handle:
        handle.seek(previous["tail_guard_offset"])
        raw = handle.read(length)
    return hashlib.sha256(raw).hexdigest() == previous["tail_guard_sha256"]


def _remove_archive(db: sqlite3.Connection, archive_path: str) -> None:
    _replace_file_nodes(db, archive_path, [])
    db.execute("DELETE FROM exchanges WHERE archive_path=?", (archive_path,))
    db.execute("DELETE FROM compactions WHERE archive_path=?", (archive_path,))
    db.execute("DELETE FROM archive_seek_points WHERE archive_path=?", (archive_path,))
    db.execute("DELETE FROM indexed_files WHERE archive_path=?", (archive_path,))


def _archive_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path.resolve() for path in root.rglob("*.jsonl") if path.is_file())


def sync_index(archive_root: Path, index_path: Path, force: bool = False) -> dict[str, Any]:
    """Incrementally synchronize local raw archives into the standalone index."""

    started = time.perf_counter()
    if force:
        for candidate in (index_path, Path(f"{index_path}-wal"), Path(f"{index_path}-shm")):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
    try:
        db = connect_index(index_path)
    except RuntimeError as error:
        if "incompatible" not in str(error):
            raise
        for candidate in (index_path, Path(f"{index_path}-wal"), Path(f"{index_path}-shm")):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        db = connect_index(index_path)

    files = _archive_files(archive_root)
    paths = {str(path) for path in files}
    existing = {row["archive_path"]: row for row in db.execute("SELECT * FROM indexed_files")}
    changed = incremental = unchanged = removed = bytes_scanned = 0
    with db:
        for stale in sorted(set(existing) - paths):
            _remove_archive(db, stale)
            removed += 1

        for path in files:
            archive_path = str(path)
            stat = path.stat()
            previous = existing.get(archive_path)
            if previous and previous["source_size"] == stat.st_size and previous["source_mtime_ns"] == stat.st_mtime_ns:
                unchanged += 1
                continue

            can_resume = bool(previous and _append_only(path, previous))
            if can_resume:
                state = ParserState.from_json(previous["parser_state"])
                incremental += 1
            else:
                _remove_archive(db, archive_path)
                state = ParserState()
                changed += 1

            exchanges, state, seeks, compactions, scanned = parse_increment(path, state)
            bytes_scanned += scanned
            _upsert_exchanges(db, exchanges)
            db.executemany(
                """INSERT OR REPLACE INTO archive_seek_points(
                     archive_path, line_number, byte_offset, prefix_sha256, prefix_length
                   ) VALUES(?, ?, ?, ?, ?)""",
                ((archive_path, line, offset, digest, length) for line, offset, digest, length in seeks),
            )
            db.executemany(
                "INSERT OR REPLACE INTO compactions(archive_path, line_number, summary_text) VALUES(?, ?, ?)",
                ((archive_path, line, summary) for line, summary in compactions),
            )
            nodes, facets, anchors = _build_file_nodes(db, archive_path, state.forked_from_id)
            _replace_file_nodes(db, archive_path, nodes, facets, anchors)
            count = db.execute(
                "SELECT count(*) FROM exchanges WHERE archive_path=?", (archive_path,)
            ).fetchone()[0]
            source_size = state.next_byte_offset
            guard_offset, guard_sha = _guard(path, source_size)
            db.execute(
                """INSERT OR REPLACE INTO indexed_files(
                     archive_path, session_id, source_size, source_mtime_ns, next_line_number,
                     tail_guard_offset, tail_guard_sha256, parser_state, exchange_count, indexed_at
                   ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                (
                    archive_path, state.session_id or path.stem, source_size, path.stat().st_mtime_ns,
                    state.next_line_number, guard_offset, guard_sha, state.to_json(), count,
                ),
            )

        edge_count = _refresh_graph(db) if changed or incremental or removed else db.execute(
            "SELECT count(*) FROM node_edges"
        ).fetchone()[0]
        db.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES('archive_root', ?)", (str(archive_root),))
        db.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES('last_indexed_at', datetime('now'))")
        db.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES('runtime_dependency', 'none')")

    if removed:
        db.execute("PRAGMA incremental_vacuum")
    db.execute("PRAGMA optimize")
    counts = {
        row["level"]: row["count"]
        for row in db.execute("SELECT level, count(*) AS count FROM nodes GROUP BY level")
    }
    exchange_count = db.execute("SELECT count(*) FROM exchanges").fetchone()[0]
    page_size = db.execute("PRAGMA page_size").fetchone()[0]
    page_count = db.execute("PRAGMA page_count").fetchone()[0]
    freelist_count = db.execute("PRAGMA freelist_count").fetchone()[0]
    db.close()
    return {
        "schema_version": SCHEMA_VERSION,
        "full_reindexed_files": changed,
        "incrementally_indexed_files": incremental,
        "unchanged_files": unchanged,
        "removed_files": removed,
        "bytes_scanned": bytes_scanned,
        "exchanges": exchange_count,
        "nodes": counts,
        "graph_edges": edge_count,
        "index_bytes": page_size * page_count,
        "reclaimable_bytes": page_size * freelist_count,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "archive_root": str(archive_root),
        "index_path": str(index_path),
        "runtime_dependency": "none",
    }


# Backward-compatible function name for local callers; the first argument is now
# an archive directory, never an upstream database.
rebuild_index = sync_index


def _fts_expression(features: QueryFeatures) -> str:
    safe = []
    for term in dict.fromkeys((*features.anchors, *features.terms)):
        cleaned = term.replace('"', '""').strip()
        if cleaned:
            safe.append(f'"{cleaned}"')
    return " OR ".join(safe[:96])


def _ranked_nodes(
    db: sqlite3.Connection,
    features: QueryFeatures,
    level: str,
    limit: int,
    session_ids: set[str] | None = None,
    include_text: bool = False,
) -> list[sqlite3.Row]:
    expression = _fts_expression(features)
    if not expression:
        return []
    params: list[Any] = [expression, level]
    session_clause = ""
    if session_ids:
        placeholders = ",".join("?" for _ in session_ids)
        session_clause = f" AND n.session_id IN ({placeholders})"
        params.extend(sorted(session_ids))
    params.append(limit)
    columns = "n.*" if include_text else (
        "n.id, n.level, n.session_id, n.archive_path, n.parent_id, n.line_start, n.line_end, "
        "n.timestamp, n.project, n.cwd, n.git_branch, n.terms, n.embedding, n.content_hash"
    )
    return db.execute(
        f"""SELECT {columns}, bm25(nodes_fts, 0.0, 2.3) AS lexical_rank
             FROM nodes_fts JOIN nodes n ON n.id=nodes_fts.node_id
             WHERE nodes_fts MATCH ? AND n.level=? {session_clause}
             ORDER BY lexical_rank, n.timestamp DESC LIMIT ?""",
        params,
    ).fetchall()


def _dense_nodes(
    db: sqlite3.Connection,
    query_vector: tuple[float, ...],
    level: str,
    limit: int,
    session_ids: set[str] | None = None,
    line_ranges: list[tuple[str, int, int]] | None = None,
) -> list[tuple[sqlite3.Row, float]]:
    params: list[Any] = [level]
    clause = ""
    if line_ranges:
        clauses = []
        for session_id, line_start, line_end in line_ranges:
            clauses.append("(session_id=? AND line_end>=? AND line_start<=?)")
            params.extend((session_id, line_start, line_end))
        clause = " AND (" + " OR ".join(clauses) + ")"
    elif session_ids:
        placeholders = ",".join("?" for _ in session_ids)
        clause = f" AND session_id IN ({placeholders})"
        params.extend(sorted(session_ids))
    rows = db.execute(
        f"SELECT id, timestamp, embedding FROM nodes WHERE level=? {clause}", params,
    ).fetchall()
    ranked = sorted(
        ((row, cosine(query_vector, row["embedding"])) for row in rows),
        key=lambda item: (item[1], item[0]["timestamp"]),
        reverse=True,
    )
    selected = [(row["id"], score) for row, score in ranked[:limit] if score > 0.04]
    if not selected:
        return []
    by_id = {
        row["id"]: row
        for row in db.execute(
            f"SELECT * FROM nodes WHERE id IN ({','.join('?' for _ in selected)})",
            [node_id for node_id, _ in selected],
        )
    }
    return [(by_id[node_id], score) for node_id, score in selected if node_id in by_id]


def _exact_anchor_nodes(
    db: sqlite3.Connection,
    features: QueryFeatures,
    limit: int,
) -> list[sqlite3.Row]:
    if not features.decisive_anchors:
        return []
    anchors = list(dict.fromkeys(features.decisive_anchors))
    placeholders = ",".join("?" for _ in anchors)
    return db.execute(
        f"""SELECT n.*, count(DISTINCT a.anchor) AS anchor_matches
              FROM node_anchors a JOIN nodes n ON n.id=a.node_id
             WHERE a.decisive=1 AND a.anchor IN ({placeholders}) AND n.level='exchange'
             GROUP BY n.id HAVING anchor_matches=?
             ORDER BY n.timestamp DESC LIMIT ?""",
        (*anchors, len(anchors), limit),
    ).fetchall()


def _lsh_nodes(
    db: sqlite3.Connection,
    query_vector: tuple[float, ...],
    limit: int,
) -> list[sqlite3.Row]:
    bands = lsh_bands(query_vector)
    placeholders = ",".join("?" for _ in bands)
    return db.execute(
        f"""SELECT n.*, count(DISTINCT l.band) AS band_matches
              FROM node_lsh l JOIN nodes n ON n.id=l.node_id
             WHERE l.band IN ({placeholders}) AND n.level='exchange'
             GROUP BY n.id ORDER BY band_matches DESC, n.timestamp DESC LIMIT ?""",
        (*bands, limit),
    ).fetchall()


def _late_interaction_rows(
    db: sqlite3.Connection,
    query: str,
    features: QueryFeatures,
    node_ids: list[str],
) -> list[tuple[sqlite3.Row, float]]:
    if not node_ids:
        return []
    q_vectors = [embed(value) for value in query_facets(query, features)]
    if not q_vectors:
        return []
    placeholders = ",".join("?" for _ in node_ids)
    node_rows = db.execute(
        f"SELECT * FROM nodes WHERE id IN ({placeholders})",
        node_ids,
    ).fetchall()
    rows = {row["id"]: row for row in node_rows}
    facet_rows = db.execute(
        f"SELECT node_id, embedding FROM node_facets WHERE node_id IN ({placeholders})",
        node_ids,
    ).fetchall()
    by_node: defaultdict[str, list[tuple[float, ...]]] = defaultdict(list)
    for row in node_rows:
        by_node[row["id"]].append(decode(row["embedding"]))
    for row in facet_rows:
        by_node[row["node_id"]].append(decode(row["embedding"]))
    scores = {
        node_id: late_interaction_score(q_vectors, vectors)
        for node_id, vectors in by_node.items()
    }
    ranked_ids = [
        node_id for node_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if score >= 0.05
    ]
    if not ranked_ids:
        return []
    return [(rows[node_id], scores[node_id]) for node_id in ranked_ids if node_id in rows]


def _expand_graph(
    db: sqlite3.Connection,
    scores: dict[str, dict[str, Any]],
    seed_limit: int = 16,
) -> int:
    seeds = sorted(scores.values(), key=lambda entry: entry["score"], reverse=True)[:seed_limit]
    if not seeds:
        return 0
    seed_ids = [entry["row"]["id"] for entry in seeds]
    seed_rank = {node_id: rank for rank, node_id in enumerate(seed_ids, 1)}
    placeholders = ",".join("?" for _ in seed_ids)
    edges = db.execute(
        f"""SELECT e.source_id, e.target_id, e.relation, e.weight, n.*
              FROM node_edges e JOIN nodes n ON n.id=e.target_id
             WHERE e.source_id IN ({placeholders})""",
        seed_ids,
    ).fetchall()
    added = 0
    for edge in edges:
        before = edge["id"] in scores
        relation = edge["relation"]
        _add_route(
            scores,
            edge,
            f"graph-{relation}",
            seed_rank[edge["source_id"]],
            0.38 * float(edge["weight"]),
        )
        if not before:
            added += 1
    return added


def _add_route(
    scores: dict[str, dict[str, Any]],
    row: sqlite3.Row,
    route: str,
    rank: int,
    weight: float,
    vector_similarity: float = 0.0,
) -> None:
    entry = scores.setdefault(
        row["id"], {"row": row, "score": 0.0, "routes": [], "vector_similarity": 0.0}
    )
    entry["score"] += weight / (RRF_K + rank)
    entry["vector_similarity"] = max(entry["vector_similarity"], vector_similarity)
    if route not in entry["routes"]:
        entry["routes"].append(route)


def _neighbor_range(db: sqlite3.Connection, row: sqlite3.Row) -> tuple[int, int]:
    neighbors = db.execute(
        """SELECT line_start, line_end FROM nodes
             WHERE level='exchange' AND session_id=?
               AND line_start BETWEEN ? AND ?
             ORDER BY line_start""",
        (row["session_id"], max(1, row["line_start"] - 200), row["line_end"] + 200),
    ).fetchall()
    unique = sorted({(item["line_start"], item["line_end"]) for item in neighbors})
    if not unique:
        return row["line_start"], row["line_end"]
    position = min(range(len(unique)), key=lambda index: abs(unique[index][0] - row["line_start"]))
    return unique[max(0, position - 1)][0], unique[min(len(unique) - 1, position + 1)][1]


def _evidence_plan(hits: list[SearchHit], features: QueryFeatures) -> dict[str, Any]:
    """Select a small diverse raw-evidence set and calibrate multi-turn support."""

    if not hits:
        return {
            "node_ids": [], "ranges": [], "combined_term_coverage": 0.0,
            "all_structured_anchors_matched": not features.anchors,
            "multi_evidence_supported": False,
        }
    required_terms = set(features.terms)
    chosen: list[SearchHit] = []
    covered: set[str] = set()
    remaining = hits[: min(16, len(hits))]
    while remaining and len(chosen) < 4:
        def utility(hit: SearchHit) -> tuple[float, float, float]:
            new_terms = len(set(hit.covered_query_terms) - covered)
            session_bonus = 0.75 if chosen and hit.session_id == chosen[0].session_id else 0.0
            graph_bonus = 0.35 if any(route.startswith("graph-") for route in hit.routes) else 0.0
            confidence = {"high": 1.0, "medium": 0.55, "low": 0.0}[hit.confidence]
            return (new_terms + session_bonus + graph_bonus + confidence, hit.evidence_coverage, hit.score)

        best = max(remaining, key=utility)
        if chosen and utility(best)[0] <= 0.55:
            break
        chosen.append(best)
        covered.update(best.covered_query_terms)
        remaining.remove(best)
        if required_terms and covered >= required_terms:
            break
    anchor_union = {anchor for hit in chosen for anchor in hit.matched_anchors}
    combined = len(covered & required_terms) / len(required_terms) if required_terms else 0.0
    same_session = len({hit.session_id for hit in chosen}) == 1
    supported_hits = [
        hit for hit in chosen
        if (
            (hit.confidence in {"medium", "high"} and (hit.evidence_coverage >= 0.14 or hit.exact_match))
            or ("exact-anchor" in hit.routes and bool(hit.matched_anchors))
        )
    ]
    all_anchors = not features.anchors or set(features.anchors).issubset(anchor_union)
    coverage_gate = 0.50 if len(features.decisive_anchors) >= 2 else 0.68
    multi_supported = bool(
        len(supported_hits) >= 2
        and same_session
        and combined >= coverage_gate
        and all_anchors
        and len(covered) >= min(3, len(required_terms))
    )
    raw_ranges = sorted(
        (
            hit.archive_path,
            hit.context_line_start,
            hit.context_line_end,
        )
        for hit in chosen
    )
    merged_ranges: list[dict[str, Any]] = []
    for archive_path, line_start, line_end in raw_ranges:
        if (
            merged_ranges
            and merged_ranges[-1]["archive_path"] == archive_path
            and line_start <= merged_ranges[-1]["line_end"] + 1
        ):
            merged_ranges[-1]["line_end"] = max(merged_ranges[-1]["line_end"], line_end)
        else:
            merged_ranges.append({
                "archive_path": archive_path,
                "line_start": line_start,
                "line_end": line_end,
            })
    return {
        "node_ids": [hit.node_id for hit in chosen],
        "ranges": merged_ranges,
        "combined_term_coverage": round(combined, 4),
        "covered_query_terms": sorted(covered),
        "all_structured_anchors_matched": all_anchors,
        "same_session": same_session,
        "multi_evidence_supported": multi_supported,
    }


def search_index(
    index_path: Path,
    query: str,
    limit: int = 8,
    mode: str = "auto",
    current_session_id: str | None = None,
) -> dict[str, Any]:
    if mode not in {"fast", "auto", "deep"}:
        raise ValueError("mode must be fast, auto, or deep")
    started = time.perf_counter()
    features = query_features(query)
    query_vector = embed(query)
    db = connect_index(index_path)
    timings: dict[str, float] = {}
    scores: dict[str, dict[str, Any]] = {}
    use_vector = mode != "fast"

    stage = time.perf_counter()
    exact_anchor_rows = _exact_anchor_nodes(db, features, max(64, limit * 8))
    for rank, row in enumerate(exact_anchor_rows, 1):
        _add_route(scores, row, "exact-anchor-direct", rank, 4.5)
        _add_route(scores, row, "exact-anchor", rank, 1.0)
        contextual_coverage = evidence_coverage(row["terms"], features)[2]
        if contextual_coverage:
            scores[row["id"]]["score"] += 0.9 * contextual_coverage
            scores[row["id"]]["routes"].append("exact-anchor-context-rerank")
    timings["exact_anchor_ms"] = (time.perf_counter() - stage) * 1000
    direct_stable_anchors = [anchor for anchor in features.decisive_anchors if indexable_anchor(anchor)]
    found_stable_anchors: set[str] = set()
    if direct_stable_anchors:
        placeholders = ",".join("?" for _ in direct_stable_anchors)
        found_stable_anchors = {
            row[0] for row in db.execute(
                f"SELECT DISTINCT anchor FROM node_anchors WHERE anchor IN ({placeholders})",
                direct_stable_anchors,
            )
        }
    missing_direct_anchors = [
        anchor for anchor in direct_stable_anchors if anchor not in found_stable_anchors
    ]
    direct_absent = bool(
        missing_direct_anchors
        and len(direct_stable_anchors) == len(features.decisive_anchors)
    )
    direct_only = bool(exact_anchor_rows and features.decisive_anchors and features.ordinal_index is None)
    if direct_only or direct_absent:
        use_vector = False

    stage = time.perf_counter()
    session_rows = [] if direct_only or direct_absent else _ranked_nodes(db, features, "session", 16)
    dense_sessions = _dense_nodes(db, query_vector, "session", 12) if use_vector else []
    session_route_scores: defaultdict[str, float] = defaultdict(float)
    for rank, row in enumerate(session_rows, 1):
        session_route_scores[row["session_id"]] += 1.0 / (RRF_K + rank)
    for rank, (row, similarity) in enumerate(dense_sessions, 1):
        session_route_scores[row["session_id"]] += (0.8 + 0.2 * similarity) / (RRF_K + rank)
    ordered_session_routes = sorted(session_route_scores.items(), key=lambda item: item[1], reverse=True)
    route_sessions = {session_id for session_id, _ in ordered_session_routes[:6]}
    session_ranks = {row["session_id"]: rank for rank, row in enumerate(session_rows, 1)}
    dense_session_ranks = {row["session_id"]: (rank, similarity) for rank, (row, similarity) in enumerate(dense_sessions, 1)}
    timings["session_route_ms"] = (time.perf_counter() - stage) * 1000

    stage = time.perf_counter()
    episode_rows = [] if direct_only or direct_absent or features.ordinal_index is not None else _ranked_nodes(
        db, features, "episode", 32, route_sessions or None,
    )
    dense_episodes = (
        _dense_nodes(db, query_vector, "episode", 32, route_sessions or None)
        if use_vector and features.ordinal_index is None else []
    )
    episode_ranges = [
        (row["session_id"], row["line_start"], row["line_end"], rank, 0.0)
        for rank, row in enumerate(episode_rows, 1)
    ] + [
        (row["session_id"], row["line_start"], row["line_end"], rank, similarity)
        for rank, (row, similarity) in enumerate(dense_episodes, 1)
    ]
    timings["episode_route_ms"] = (time.perf_counter() - stage) * 1000

    stage = time.perf_counter()
    exchange_rows = [] if direct_only or direct_absent or features.ordinal_index is not None else _ranked_nodes(
        db, features, "exchange", max(64, limit * 10), include_text=True,
    )
    for rank, row in enumerate(exchange_rows, 1):
        if row["session_id"] in session_ranks:
            _add_route(scores, row, "session-route", session_ranks[row["session_id"]], 0.55)
        if row["session_id"] in dense_session_ranks:
            dense_rank, similarity = dense_session_ranks[row["session_id"]]
            _add_route(scores, row, "local-vector-session", dense_rank, 0.35, similarity)
        matching = [
            (episode_rank, similarity)
            for session_id, line_start, line_end, episode_rank, similarity in episode_ranges
            if session_id == row["session_id"] and row["line_end"] >= line_start and row["line_start"] <= line_end
        ]
        if matching:
            lexical_episode = [rank_ for rank_, similarity in matching if similarity == 0.0]
            dense_episode = [(rank_, similarity) for rank_, similarity in matching if similarity > 0.0]
            if lexical_episode:
                _add_route(scores, row, "episode-route", min(lexical_episode), 0.95)
            if dense_episode:
                rank_, similarity = max(dense_episode, key=lambda item: item[1])
                _add_route(scores, row, "local-vector-episode", rank_, 0.65, similarity)
        weight = 1.75
        normalized = normalize_text(row["text"])
        if features.query and features.query in normalized:
            weight += 2.5
            _add_route(scores, row, "exact-phrase", rank, 2.5)
        matched_decisive = [anchor for anchor in features.decisive_anchors if anchor in normalized]
        matched_contextual = [
            anchor for anchor in features.anchors
            if anchor not in features.decisive_anchors and anchor in normalized
        ]
        if matched_decisive:
            weight += 2.2
            _add_route(scores, row, "exact-anchor", rank, 2.2)
        if matched_contextual:
            weight += 0.35
            _add_route(scores, row, "contextual-anchor", rank, 0.35)
        _add_route(scores, row, "sparse-exchange", rank, weight)
        matched_anchor_count = sum(anchor in normalized for anchor in features.anchors)
        if features.anchors and matched_anchor_count:
            scores[row["id"]]["score"] += 0.03 * matched_anchor_count / len(features.anchors)
            if matched_decisive and len(matched_decisive) == len(features.decisive_anchors):
                scores[row["id"]]["score"] += 0.08
        scores[row["id"]]["score"] += 0.12 * evidence_coverage(row["terms"], features)[2]
    timings["sparse_evidence_ms"] = (time.perf_counter() - stage) * 1000

    stage = time.perf_counter()
    if use_vector and features.ordinal_index is None:
        dense_scope = None if mode == "deep" else (route_sessions or None)
        dense_ranges = None
        if mode == "auto":
            ranges = [
                (row["session_id"], row["line_start"], row["line_end"])
                for row in episode_rows[:12]
            ] + [
                (row["session_id"], row["line_start"], row["line_end"])
                for row, _ in dense_episodes[:12]
            ]
            dense_ranges = list(dict.fromkeys(ranges)) or None
        for rank, (row, similarity) in enumerate(
            _dense_nodes(
                db, query_vector, "exchange", max(48, limit * 8),
                dense_scope, dense_ranges,
            ), 1
        ):
            _add_route(scores, row, "local-vector-exchange", rank, 1.15, similarity)
    timings["local_vector_ms"] = (time.perf_counter() - stage) * 1000

    stage = time.perf_counter()
    if use_vector and features.ordinal_index is None:
        lsh_rows = _lsh_nodes(db, query_vector, max(96, limit * 16))
        candidate_ids = list(dict.fromkeys(
            [entry["row"]["id"] for entry in sorted(
                scores.values(), key=lambda entry: entry["score"], reverse=True,
            )[:160]]
            + [row["id"] for row in lsh_rows]
        ))
        for rank, (row, similarity) in enumerate(
            _late_interaction_rows(db, query, features, candidate_ids), 1
        ):
            _add_route(scores, row, "late-interaction", rank, 1.35, similarity)
            scores[row["id"]]["score"] += 0.07 * similarity
            scores[row["id"]]["late_similarity"] = similarity
    timings["lsh_late_interaction_ms"] = (time.perf_counter() - stage) * 1000

    stage = time.perf_counter()
    graph_candidates = _expand_graph(db, scores) if features.ordinal_index is None else 0
    timings["graph_expansion_ms"] = (time.perf_counter() - stage) * 1000

    ordinal_ambiguous = False
    if features.ordinal_index is not None:
        eligible_session_ids = {
            row["session_id"]
            for row in db.execute(
                """SELECT session_id FROM exchanges WHERE turn_kind='user'
                     GROUP BY session_id HAVING count(*)>=?""",
                (features.ordinal_index,),
            )
        }
        eligible_routes = [item for item in ordered_session_routes if item[0] in eligible_session_ids]
        session_by_id = {row["session_id"]: row for row in session_rows}
        if current_session_id and current_session_id in eligible_session_ids:
            eligible_routes = [
                (current_session_id, dict(eligible_routes).get(current_session_id, 0.0) + 1.0),
                *(item for item in eligible_routes if item[0] != current_session_id),
            ]
        elif len(eligible_routes) >= 2:
            top_row = session_by_id.get(eligible_routes[0][0])
            second_row = session_by_id.get(eligible_routes[1][0])
            top_coverage = evidence_coverage(top_row["terms"], features)[2] if top_row else 0.0
            second_coverage = evidence_coverage(second_row["terms"], features)[2] if second_row else 0.0
            ordinal_ambiguous = bool(
                second_coverage >= 0.18
                and second_coverage >= top_coverage * 0.80
            )
        ordinal_session_rows = [
            session_by_id[session_id]
            for session_id, _ in eligible_routes[:5]
            if session_id in session_by_id
        ]
        for session_rank, session_row in enumerate(ordinal_session_rows, 1):
            exchange = db.execute(
                """SELECT * FROM exchanges WHERE session_id=? AND turn_kind='user' ORDER BY line_start
                     LIMIT 1 OFFSET ?""",
                (session_row["session_id"], features.ordinal_index - 1),
            ).fetchone()
            if exchange is None:
                continue
            row = db.execute(
                """SELECT * FROM nodes WHERE level='exchange' AND archive_path=? AND line_start=?
                     ORDER BY id LIMIT 1""",
                (exchange["archive_path"], exchange["line_start"]),
            ).fetchone()
            if row is None:
                continue
            session_coverage = evidence_coverage(session_row["terms"], features)[2]
            if session_coverage < 0.10:
                continue
            _add_route(scores, row, f"ordinal-{features.ordinal_index}", session_rank, 1.9)
            _add_route(scores, row, "session-route", session_rank, 0.75)
            scores[row["id"]]["ordinal_support"] = max(
                scores[row["id"]].get("ordinal_support", 0.0), session_coverage,
            )
            scores[row["id"]]["ordinal_priority"] = max(
                scores[row["id"]].get("ordinal_priority", 0.0), 1.0 / session_rank,
            )
            if session_coverage >= 0.10:
                # Ordinality is the user's requested selector, not a weak
                # topical hint. Once a session is lexically established, its
                # exact Nth exchange should outrank an earlier topical mention.
                scores[row["id"]]["score"] += 0.35 / session_rank

    if current_session_id:
        for entry in scores.values():
            if entry["row"]["session_id"] == current_session_id:
                entry["score"] += 0.025
                entry["routes"].append("current-session")

    if features.temporal_direction in {"earliest", "latest"} and scores:
        maximum_coverage = max(evidence_coverage(entry["row"]["terms"], features)[2] for entry in scores.values())
        eligible = [
            entry for entry in scores.values()
            if evidence_coverage(entry["row"]["terms"], features)[2] >= max(0.2, maximum_coverage * 0.65)
        ]
        reverse = features.temporal_direction == "latest"
        ordered = sorted(eligible, key=lambda entry: entry["row"]["timestamp"], reverse=reverse)
        count = max(1, len(ordered))
        for rank, entry in enumerate(ordered, 1):
            entry["score"] += 0.055 * (count - rank + 1) / count
            entry["routes"].append(f"time-{features.temporal_direction}")

    ranked = sorted(
        scores.values(),
        key=lambda entry: (
            entry.get("ordinal_priority", 0.0) if features.ordinal_index is not None else 0.0,
            entry["score"],
            entry["row"]["timestamp"],
        ),
        reverse=True,
    )
    hits: list[SearchHit] = []
    seen_hashes: set[str] = set()
    for entry in ranked:
        row = entry["row"]
        if row["content_hash"] in seen_hashes:
            continue
        seen_hashes.add(row["content_hash"])
        normalized = normalize_text(row["text"])
        matched_anchors = [anchor for anchor in features.anchors if anchor in normalized]
        all_anchors = not features.anchors or len(matched_anchors) == len(features.anchors)
        matched_decisive = [anchor for anchor in features.decisive_anchors if anchor in normalized]
        all_decisive = bool(features.decisive_anchors and len(matched_decisive) == len(features.decisive_anchors))
        exact = bool((features.query and features.query in normalized) or all_decisive)
        lexical_coverage, group_coverage, coverage = evidence_coverage(row["terms"], features)
        available_terms = set((row["terms"] or "").split())
        covered_terms = [term for term in features.terms if term in available_terms]
        route_count = len(set(entry["routes"]))
        similarity = float(entry["vector_similarity"])
        ordinal_support = float(entry.get("ordinal_support", 0.0))
        confidence = (
            "high" if exact or (coverage >= 0.65 and route_count >= 2)
            else "medium" if (
                (coverage >= 0.30 and route_count >= 2)
                or (coverage >= 0.12 and similarity >= 0.28 and route_count >= 3)
                or (ordinal_support >= 0.10 and route_count >= 2)
            ) else "low"
        )
        context_start, context_end = _neighbor_range(db, row)
        snippet = re.sub(r"\s+", " ", row["text"]).strip()
        if len(snippet) > 360:
            snippet = snippet[:357] + "..."
        hits.append(
            SearchHit(
                node_id=row["id"], session_id=row["session_id"], archive_path=row["archive_path"],
                line_start=row["line_start"], line_end=row["line_end"],
                context_line_start=context_start, context_line_end=context_end,
                timestamp=row["timestamp"], score=round(entry["score"], 8), confidence=confidence,
                routes=list(dict.fromkeys(entry["routes"])), term_coverage=round(lexical_coverage, 4),
                concept_coverage=round(group_coverage, 4), evidence_coverage=round(coverage, 4),
                local_vector_similarity=round(similarity, 4),
                late_interaction_similarity=round(float(entry.get("late_similarity", 0.0)), 4),
                exact_match=exact,
                matched_anchors=matched_anchors, all_structured_anchors_matched=all_anchors,
                covered_query_terms=covered_terms,
                snippet=snippet, content_hash=row["content_hash"],
            )
        )
        if len(hits) >= limit:
            break

    timings["total_ms"] = (time.perf_counter() - started) * 1000
    db.close()
    top = hits[0] if hits else None
    evidence_plan = _evidence_plan(hits, features)
    top_routes = set(top.routes) if top else set()
    supported = bool(top and (
        top.confidence == "high"
        or (top.evidence_coverage >= 0.45 and len(top_routes) >= 2)
        or (
            top.evidence_coverage >= 0.12
            and top.local_vector_similarity >= 0.28
            and len(top_routes) >= 3
        )
        or (
            features.ordinal_index is not None
            and f"ordinal-{features.ordinal_index}" in top_routes
            and top.confidence in {"medium", "high"}
        )
        or (
            any(route.startswith("time-") for route in top_routes)
            and top.evidence_coverage >= 0.25
            and len(top_routes) >= 3
        )
    ))
    if features.ordinal_index is not None and ordinal_ambiguous:
        supported = False
    structured_guard = bool(not features.has_structured_anchor or (top and top.all_structured_anchors_matched))
    single_supported = supported and structured_guard
    multi_supported = bool(evidence_plan["multi_evidence_supported"])
    answerability = (
        "evidence_found" if single_supported
        else "multi_evidence_found" if multi_supported
        else "insufficient_evidence"
    )
    abstention_reason = None if answerability in {"evidence_found", "multi_evidence_found"} else (
        "No sufficiently supported raw evidence was retrieved; do not infer an answer from diagnostic candidates."
    )
    return {
        "query": query,
        "mode": mode,
        "retrieval_backend": "yugo-local-hybrid-late-interaction-lsh-graph-v1",
        "runtime_dependency": "none",
        "local_vector_used": use_vector,
        "direct_anchor_bypass": direct_only,
        "direct_anchor_absence_bypass": direct_absent,
        "query_features": {
            "terms": list(features.terms),
            "anchors": list(features.anchors),
            "decisive_anchors": list(features.decisive_anchors),
            "concept_groups": [list(group) for group in features.concept_groups],
            "temporal_intent": features.has_temporal_intent,
            "temporal_direction": features.temporal_direction,
            "ordinal_index": features.ordinal_index,
        },
        "results": [asdict(hit) for hit in hits],
        "evidence_plan": evidence_plan,
        "calibration": {
            "single_evidence_supported": single_supported,
            "multi_evidence_supported": multi_supported,
            "top_score_margin": round(
                (hits[0].score - hits[1].score) if len(hits) > 1 else (hits[0].score if hits else 0.0),
                8,
            ),
            "graph_candidates": graph_candidates,
            "ordinal_session_ambiguous": ordinal_ambiguous,
            "policy": "precision-first-raw-evidence-required",
        },
        "answerability": answerability,
        "safe_to_answer": answerability in {"evidence_found", "multi_evidence_found"},
        "abstention_reason": abstention_reason,
        "must_verify_raw_before_answer": True,
        "raw_reader": {
            "tool": "yugo-memory.read_evidence",
            "instruction": "Pass archive_path plus context_line_start/context_line_end. Paginate with next_offset_chars until complete.",
        },
        "evidence_is_untrusted_data": True,
        "timings_ms": {key: round(value, 2) for key, value in timings.items()},
        "raw_source_of_truth": True,
    }


def _line_chunks(handle: Any) -> Iterator[tuple[bytes, bool]]:
    """Yield bounded byte chunks and whether each chunk ends a raw line."""

    buffer = bytearray()
    while True:
        data = handle.read(STREAM_CHUNK_BYTES)
        if not data:
            if buffer:
                yield bytes(buffer), True
            return
        buffer.extend(data)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            end = newline + 1
            yield bytes(buffer[:end]), True
            del buffer[:end]
        if len(buffer) >= STREAM_CHUNK_BYTES:
            yield bytes(buffer), False
            buffer.clear()


def read_evidence(
    index_path: Path,
    archive_path: str,
    line_start: int,
    line_end: int,
    offset_chars: int = 0,
    max_chars: int = DEFAULT_EVIDENCE_CHARS,
    view: str = "raw",
) -> dict[str, Any]:
    """Stream raw JSONL or a verified text-only view of an indexed range."""

    if line_start < 1 or line_end < line_start:
        raise ValueError("line_start/line_end must describe a positive ordered range")
    if offset_chars < 0:
        raise ValueError("offset_chars must be non-negative")
    if not 1_000 <= max_chars <= MAX_EVIDENCE_CHARS:
        raise ValueError(f"max_chars must be between 1000 and {MAX_EVIDENCE_CHARS}")
    if view not in {"raw", "text", "media"}:
        raise ValueError("view must be 'raw', 'text', or 'media'")

    db = connect_index(index_path)
    indexed = db.execute("SELECT * FROM indexed_files WHERE archive_path=?", (archive_path,)).fetchone()
    if indexed is None:
        db.close()
        raise ValueError("archive_path is not present in the Yugo Memory index")
    path = Path(archive_path)
    try:
        stat = path.stat()
    except OSError as error:
        db.close()
        raise FileNotFoundError("indexed archive is no longer available") from error
    if stat.st_size < indexed["source_size"]:
        db.close()
        raise RuntimeError("archive shrank after indexing; run recall again after memory sync")
    if stat.st_size == indexed["source_size"] and stat.st_mtime_ns != indexed["source_mtime_ns"]:
        db.close()
        raise RuntimeError("archive changed after indexing; run recall again after memory sync")
    if stat.st_size > indexed["source_size"]:
        guard_length = indexed["source_size"] - indexed["tail_guard_offset"]
        with path.open("rb") as guard_handle:
            guard_handle.seek(indexed["tail_guard_offset"])
            guard_raw = guard_handle.read(guard_length)
        if hashlib.sha256(guard_raw).hexdigest() != indexed["tail_guard_sha256"]:
            db.close()
            raise RuntimeError("archive was not append-only after indexing; run recall again after memory sync")
    seek = db.execute(
        """SELECT line_number, byte_offset, prefix_sha256, prefix_length
             FROM archive_seek_points WHERE archive_path=? AND line_number<=?
             ORDER BY line_number DESC LIMIT 1""",
        (archive_path, line_start),
    ).fetchone()
    db.close()
    if seek is None:
        raise RuntimeError("archive seek index is missing; rebuild the Yugo Memory index")

    if view in {"text", "media"}:
        parts: list[str] = []
        seen_chars = 0
        emitted = 0
        last_line = seek["line_number"] - 1
        complete = True
        seen_media_events: set[str] = set()
        media_tool_events = 0
        media_duplicates_omitted = 0
        media_tool_events_omitted = 0
        media_omission_marker_emitted = False
        media_text_events_compacted = 0
        with path.open("rb") as handle:
            handle.seek(seek["byte_offset"])
            prefix = handle.read(seek["prefix_length"])
            if hashlib.sha256(prefix).hexdigest() != seek["prefix_sha256"]:
                raise RuntimeError("archive content changed after indexing; run recall again after memory sync")
            handle.seek(seek["byte_offset"])
            for line in iter_bounded_lines(handle, seek["byte_offset"], seek["line_number"]):
                last_line = line.number
                if line.number > line_end:
                    break
                if line.number < line_start:
                    continue
                if line.raw is None:
                    role, text = oversized_evidence_text(line)
                    timestamp = ""
                    row_type = "oversized_event"
                    integrity = f"bytes={line.raw_bytes} raw_sha256={line.raw_sha256}"
                else:
                    try:
                        row = json.loads(line.raw)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        role = "event"
                        text = "[undecodable JSONL event; use raw view]"
                        timestamp = ""
                        row_type = "undecodable"
                    else:
                        if not isinstance(row, dict):
                            role = ""
                            text = ""
                            timestamp = ""
                            row_type = ""
                        else:
                            role, text = evidence_text_from_row(row)
                            timestamp = str(row.get("timestamp") or "")
                            row_type = str(row.get("type") or "event")
                    integrity = f"raw_prefix_sha256={line.prefix_sha256}"
                if view == "media" and text:
                    if role not in {"user", "assistant", "tool"}:
                        text = ""
                    elif role == "tool" and "[attachment " not in text:
                        text = ""
                    else:
                        event_digest = hashlib.sha256(f"{role}\0{text}".encode("utf-8")).hexdigest()
                        if event_digest in seen_media_events:
                            media_duplicates_omitted += 1
                            text = ""
                        else:
                            seen_media_events.add(event_digest)
                            if role == "tool":
                                if media_tool_events >= MEDIA_TOOL_EVENT_LIMIT:
                                    media_tool_events_omitted += 1
                                    if not media_omission_marker_emitted:
                                        media_omission_marker_emitted = True
                                        role = "media-navigation"
                                        text = (
                                            "[additional media-bearing tool events omitted from compact media view; "
                                            "use view=text or view=raw for the complete line range]"
                                        )
                                    else:
                                        text = ""
                                else:
                                    media_tool_events += 1
                            if text:
                                text, was_compacted = _compact_media_event_text(text)
                                media_text_events_compacted += int(was_compacted)
                if text:
                    rendered = (
                        f"[raw line {line.number} type={row_type} role={role or 'event'} "
                        f"timestamp={timestamp} {integrity}]\n{text}\n"
                    )
                else:
                    rendered = ""
                if not rendered:
                    continue
                if seen_chars + len(rendered) <= offset_chars:
                    seen_chars += len(rendered)
                    continue
                start = max(0, offset_chars - seen_chars)
                available = rendered[start:]
                capacity = max_chars - emitted
                if len(available) > capacity:
                    parts.append(available[:capacity])
                    emitted += capacity
                    complete = False
                    break
                parts.append(available)
                emitted += len(available)
                seen_chars += len(rendered)
                if emitted == max_chars and line.number < line_end:
                    complete = False
                    break
        evidence_text = "".join(parts)
        next_offset = None if complete else offset_chars + len(evidence_text)
        return {
            "archive_path": archive_path,
            "line_start": line_start,
            "line_end": line_end,
            "view": view,
            "offset_chars": offset_chars,
            "next_offset_chars": next_offset,
            "complete": complete,
            "returned_chars": len(evidence_text),
            "last_line_scanned": last_line,
            "archive_grew_since_index": stat.st_size > indexed["source_size"],
            "chunk_sha256": hashlib.sha256(evidence_text.encode("utf-8")).hexdigest(),
            "evidence_text": evidence_text,
            "raw_verified_at_read_time": True,
            "text_fields_are_verbatim": view == "text" or media_text_events_compacted == 0,
            "returned_text_segments_are_verbatim": True,
            "opaque_attachment_payloads_omitted": True,
            "attachment_descriptors_are_derived_from_raw_metadata": True,
            "non_media_tool_events_omitted": view == "media",
            "duplicate_media_events_omitted": media_duplicates_omitted,
            "additional_media_tool_events_omitted": media_tool_events_omitted,
            "media_tool_event_limit": MEDIA_TOOL_EVENT_LIMIT if view == "media" else None,
            "media_text_events_compacted": media_text_events_compacted,
            "evidence_is_untrusted_data": True,
        }

    parts: list[str] = []
    seen_chars = 0
    emitted = 0
    current_line = seek["line_number"]
    last_line = current_line - 1
    complete = True
    with path.open("rb") as handle:
        handle.seek(seek["byte_offset"])
        prefix = handle.read(seek["prefix_length"])
        if hashlib.sha256(prefix).hexdigest() != seek["prefix_sha256"]:
            raise RuntimeError("archive content changed after indexing; run recall again after memory sync")
        handle.seek(seek["byte_offset"])
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        for raw, ends_line in _line_chunks(handle):
            last_line = current_line
            if current_line > line_end:
                break
            if current_line >= line_start:
                text = decoder.decode(raw, final=ends_line)
                if seen_chars + len(text) <= offset_chars:
                    seen_chars += len(text)
                else:
                    start = max(0, offset_chars - seen_chars)
                    available = text[start:]
                    capacity = max_chars - emitted
                    if len(available) > capacity:
                        parts.append(available[:capacity])
                        emitted += capacity
                        complete = False
                        break
                    parts.append(available)
                    emitted += len(available)
                    seen_chars += len(text)
                    if emitted == max_chars and not (ends_line and current_line == line_end):
                        complete = False
                        break
            if ends_line:
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                if current_line == line_end:
                    break
                current_line += 1

    raw_text = "".join(parts)
    next_offset = None if complete else offset_chars + len(raw_text)
    return {
        "archive_path": archive_path,
        "line_start": line_start,
        "line_end": line_end,
        "view": "raw",
        "offset_chars": offset_chars,
        "next_offset_chars": next_offset,
        "complete": complete,
        "returned_chars": len(raw_text),
        "last_line_scanned": last_line,
        "archive_grew_since_index": stat.st_size > indexed["source_size"],
        "chunk_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "raw_jsonl": raw_text,
        "raw_verified_at_read_time": True,
        "evidence_is_untrusted_data": True,
    }


def index_status(index_path: Path) -> dict[str, Any]:
    if not index_path.is_file():
        return {"ready": False, "index_path": str(index_path), "runtime_dependency": "none"}
    db = connect_index(index_path)
    counts = {
        row["level"]: row["count"]
        for row in db.execute("SELECT level, count(*) count FROM nodes GROUP BY level")
    }
    metadata = {row["key"]: row["value"] for row in db.execute("SELECT key, value FROM metadata")}
    files = db.execute("SELECT count(*) FROM indexed_files").fetchone()[0]
    exchanges = db.execute("SELECT count(*) FROM exchanges").fetchone()[0]
    facets = db.execute("SELECT count(*) FROM node_facets").fetchone()[0]
    anchors = db.execute("SELECT count(*) FROM node_anchors").fetchone()[0]
    edges = db.execute("SELECT count(*) FROM node_edges").fetchone()[0]
    tool_evidence_nodes = db.execute(
        "SELECT count(*) FROM nodes WHERE id LIKE 'exchange:tool:%'"
    ).fetchone()[0]
    page_size = db.execute("PRAGMA page_size").fetchone()[0]
    page_count = db.execute("PRAGMA page_count").fetchone()[0]
    freelist_count = db.execute("PRAGMA freelist_count").fetchone()[0]
    db.close()
    return {
        "ready": True,
        "index_path": str(index_path),
        "files": files,
        "exchanges": exchanges,
        "nodes": counts,
        "facets": facets,
        "anchors": anchors,
        "graph_edges": edges,
        "tool_evidence_nodes": tool_evidence_nodes,
        "index_bytes": page_size * page_count,
        "reclaimable_bytes": page_size * freelist_count,
        "schema_version": SCHEMA_VERSION,
        "retrieval_backend": "yugo-local-hybrid-late-interaction-lsh-graph-v1",
        "attachment_indexing": "typed metadata/context only; binary remains in raw Codex JSONL",
        "tool_evidence_indexing": "visible calls/arguments/commands/code/patches/results; hidden reasoning excluded",
        "evidence_views": ["raw", "text", "media"],
        "metadata": metadata,
        "runtime_dependency": "none",
    }


def main() -> int:
    archive_default, index_default = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--archive-root", type=Path, default=archive_default)
    index_parser.add_argument("--output", type=Path, default=index_default)
    index_parser.add_argument("--force", action="store_true")
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--index", type=Path, default=index_default)
    search_parser.add_argument("--limit", type=int, default=8)
    search_parser.add_argument("--mode", choices=("fast", "auto", "deep"), default="auto")
    search_parser.add_argument("--current-session-id")
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--index", type=Path, default=index_default)
    args = parser.parse_args()

    if args.command == "index":
        result = sync_index(args.archive_root, args.output, force=args.force)
    elif args.command == "search":
        result = search_index(args.index, args.query, args.limit, args.mode, args.current_session_id)
    else:
        result = index_status(args.index)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
