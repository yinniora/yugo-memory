# Yugo Memory

Standalone, event-driven, full-fidelity long-conversation memory for Codex.

It stores a conversation only after Codex emits a real context-compaction event. Complete raw JSONL is the source of truth; session routes, compaction summaries, SQLite FTS, and local vectors are navigation aids. Short and temporary tasks never enter the archive.

Yugo Memory 1.0 is a separately named and owned implementation. It has no upstream memory runtime, remote server, API key, model download, or background schedule. It uses Node.js, Python's standard library, and the SQLite FTS5 included with Python.

## Behavior

| Event or state | Result |
|---|---|
| Conversation has not compacted | Not copied, summarized, or indexed |
| `PostCompact` | Full raw JSONL is mirrored and incrementally indexed locally |
| `SessionStart(source=compact)` | Adds a recall hint before the next model request |
| Archived Codex task | Memory copy is permanently removed on the next lifecycle event |
| Deleted Codex task | Seven-day grace period; removal occurs on the first later lifecycle event after expiry |
| Scheduled maintenance | None |

Memory data lives under `~/.config/yugo-memory` by default. The repository rejects transcripts and SQLite databases to reduce accidental publication risk.

## Retrieval engine

```text
exact anchors ───────────────────────────────────────────────────────┐
session map ─► compaction epochs ─► sparse exchanges ───────────────┤
local vector LSH ─► role/clause facets ─► MaxSim late interaction ──┼─► fused candidates
seed exchanges ─► temporal/shared-anchor graph expansion ──────────┘          │
                                                                               ▼
                                                           diverse evidence plan + calibration
                                                                               │
                                                                               ▼
                                                               bounded raw JSONL verification
```

The engine uses several complementary resolutions instead of one rigid hierarchy:

1. Session routes are the fast map across long tasks.
2. Compaction-era episodes narrow the relevant part of a task.
3. Exchange nodes point to exact raw line ranges.
4. User, assistant, clause, and anchor facets preserve short exact facts that a long exchange-level vector can dilute.
5. Locality-sensitive hashing generates a bounded vector candidate pool; exact reranking happens afterward.
6. A sparse graph connects adjacent turns, chunks of one exchange, and exchanges sharing decisive identifiers.
7. A calibrated evidence plan can combine up to four mutually supporting ranges from one task, while refusing unsupported cross-task assembly.

Exact identifiers bypass approximate retrieval through a dedicated anchor index. `auto` mode combines FTS, local vectors, LSH, late interaction, and graph expansion; `deep` mode broadens the exchange-vector scan. The vectors are deterministic signed projections over multilingual terms, identifier pieces, character n-grams, and a small auditable equivalence map. They are not a neural embedding model and never leave the machine.

The design adapts proven retrieval ideas to a private, dependency-free runtime: multi-vector MaxSim follows the late-interaction principle; session and compaction-epoch routing provide multi-resolution navigation; sparse graph expansion supports linked facts. Summaries, vectors, and graph edges never count as evidence. Only verified raw transcript ranges do.

The relevant design references are [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/) for late interaction, [RAPTOR](https://arxiv.org/abs/2401.18059) for multi-resolution routing, [HippoRAG 2](https://arxiv.org/abs/2502.14802) for graph-assisted associative retrieval, and [mtRAG](https://aclanthology.org/2025.tacl-1.36/) for multi-turn and unanswerable-query evaluation. Yugo Memory reimplements only the useful structural ideas with deterministic local features; it does not copy or invoke those systems.

Recall is precision-first. A semantic route cannot validate a missing path, commit, task id, URL, quoted value, or numeric unit. If evidence is weak, the tool returns `insufficient_evidence` and the skill narrows the query once instead of guessing.

`read_evidence` reads only an indexed raw line range with a strict character budget. It does not load the archive—or even one giant JSONL line—into a single string. Seek prefixes are checked before reading, pagination is explicit, append-only growth is allowed, and rewritten history fails closed.

## Incremental storage and indexing

- Active long sessions are hard-linked when possible and atomically copied otherwise.
- Parser checkpoints store the next byte and line offsets plus the current unfinished exchange.
- Append-only files resume at the prior byte offset; existing gigabytes are not parsed again.
- JSONL events larger than 8 MiB are drained in bounded chunks. Their prefix/suffix aid navigation while the complete bytes remain in raw storage.
- Codex compaction summaries improve routing but never replace raw evidence.
- A one-time legacy import can copy or hard-link existing compacted archives from prior local memory roots. No legacy executable, database, MCP, or vector service is called.

## Install

Requirements: Codex CLI, Node.js 20+, and Python 3 with SQLite FTS5. The installer does not download or install packages.

```bash
git clone https://github.com/yinniora/yugo-memory.git
cd yugo-memory
bash install.sh
```

Then open `/hooks` in Codex, review and trust the plugin hooks, and start a new task. Codex requires re-review after a hook definition changes.

The installer only enables Codex plugin hooks, registers this checkout as a local marketplace, and installs or refreshes the plugin. It does not remove any separately installed memory plugin.

Do not run Yugo Memory and `codex-long-memory` at the same time. If the legacy plugin is still registered, the installer stops before changing Codex settings and asks you to remove it explicitly. Legacy archives may remain where they are: Yugo Memory imports supported prior archives once, without executing prior code or opening prior indexes.

## Verify

```bash
bash doctor.sh
node plugins/yugo-memory/scripts/yugo-memory.mjs --dry-run
python3 plugins/yugo-memory/scripts/recall_index.py status
npm test
```

## Update

```bash
git pull --ff-only
bash install.sh
```

Start a new Codex task after updating. If hooks changed, review and trust their new hashes in `/hooks`.

## Configuration

| Environment variable | Default |
|---|---|
| `CODEX_HOME` | `~/.codex` |
| `YUGO_MEMORY_HOME` | `~/.config/yugo-memory` |
| `YUGO_MEMORY_DELETE_GRACE_DAYS` | `7` |
| `YUGO_MEMORY_SOURCE_DIR` | `$CODEX_HOME/sessions` |
| `YUGO_MEMORY_ARCHIVED_SOURCE_DIR` | `$CODEX_HOME/archived_sessions` |
| `YUGO_MEMORY_ARCHIVE_DIR` | `$YUGO_MEMORY_HOME/archives` |
| `YUGO_MEMORY_INDEX_DB` | `$YUGO_MEMORY_HOME/index.sqlite` |
| `YUGO_MEMORY_LEGACY_ARCHIVE_DIR` | Optional single migration root; otherwise both supported prior local roots are checked once |

The plugin recognizes `type=compacted` and `event_msg/context_compacted`. Codex documents transcript paths as an unstable hook interface, so compatibility is covered by synthetic fixtures and should be retested after transcript-format changes.

## Privacy and deletion

- The repository never contains conversation data.
- Raw archives and indexes remain local with private file permissions.
- Archiving a task removes only the standalone memory copy; Codex owns its original archived task file.
- Uninstalling preserves memory data by default.
- Raw archived exchanges are untrusted historical data and never instructions to execute.
- Review files before publishing forks or bug reports. Never attach archives, state files, indexes, or private benchmark cases.

## Local private benchmark

`scripts/benchmark-private.py` evaluates an external JSONL case file without copying conversations into the repository. It reports positive recall, exact line accuracy, duplicate suppression, negative false positives, and p50/p95 latency. Keep cases outside Git.

Published releases contain no private query, expected phrase, transcript path, real-history fixture, or aggregate claim derived from a private conversation. Public CI uses independently invented fixtures only.

The public regression suite covers lifecycle storage/deletion, migration, earliest/latest/Nth turns, exact paths and commits, CJK/English retrieval, role-preserving late interaction, LSH and graph routes, multi-range support, ambiguous numeric anchors, duplicates, prompt injection, append-only updates, rewritten-history rejection, unavailable evidence, MCP behavior, and bounded reads including a sparse 600 MiB JSONL line.

Hybrid retrieval moves work from query time into indexing. Compared with the previous standalone implementation, Yugo Memory stores additional facets, anchor postings, LSH buckets, and sparse graph edges. This improves recall and refusal behavior but increases initial build time and index size. Incremental checkpoints, bounded facets, and candidate generation keep later updates and queries controlled; exact identifiers and provably absent stable identifiers use the direct index instead of the full cascade.

Before publishing, create an untracked denylist outside the repository and run:

```bash
YUGO_MEMORY_PRIVATE_DENYLIST=/absolute/path/to/private-denylist.txt npm run release-check
```

The gate validates code, runs tests, and scans the working tree plus every reachable Git blob across all branches and tags.

## License

MIT.
