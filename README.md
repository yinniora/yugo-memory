# Yugo Memory

Standalone, event-driven, full-fidelity long-conversation memory and task continuity for Codex and Qoder.

Codex conversations enter memory only after a real context-compaction event. Qoder conversations use their declared context window and a conservative visible-token estimate to cross an equivalent long-session boundary. Each agent's complete raw JSONL remains the source of truth. Yugo Memory keeps one canonical evidence link per long session; routes, task lists, experience summaries, compaction summaries, SQLite FTS, and local vectors are navigation aids. Short and temporary tasks never enter the archive.

Yugo Memory 1.4.1 has no upstream memory runtime, remote server, API key, model download, package installation, or background schedule. It uses Node.js, Python's standard library, and the SQLite FTS5 included with Python.

## Behavior

| Event or state | Result |
|---|---|
| Codex conversation has not compacted | Not copied, summarized, or indexed |
| `PostCompact` | One canonical hard link is refreshed and incrementally indexed |
| `SessionStart(source=compact)` | Adds a bounded continuity hint and active task checklist |
| Qoder transcript crosses its adaptive long boundary | Shares one canonical evidence link and index with Codex |
| Long multi-step request | `prepare_context` updates the ephemeral task checklist and chooses a context-sized response |
| Task changes or completes | Previous checklist is replaced or permanently cleared |
| Verified reusable workflow succeeds | A versioned experience can be upserted with raw evidence locators |
| Archived Codex task | Memory copy is permanently removed on the next lifecycle event |
| Deleted Codex task | Seven-day grace period; removal occurs on the first later lifecycle event after expiry |
| Scheduled maintenance | None |

Memory data lives under `~/.config/yugo-memory` by default. The repository rejects transcripts and SQLite databases to reduce accidental publication risk.

## Adaptive continuity

`prepare_context` is the normal entry point for long multi-step work. It reads the current agent's declared context window from Codex `session_meta` or Qoder `runtime-config`, estimates post-compaction usage from bounded visible text, and selects one of three response shapes:

| Profile | When selected | Recall payload |
|---|---|---|
| `minimal` | Estimated context remaining is at most 12% | One locator and up to one raw range; 4,000-character read recommendation |
| `compact` | Estimated context remaining is at most 30% | Up to three candidates and two raw ranges; 10,000-character read recommendation |
| `standard` | More context is available or the window is unknown | Normal candidates and up to four raw ranges; 24,000-character read recommendation |

The estimate chooses an output budget; it is not presented as exact tokenizer accounting. `diagnostic` remains available for tests and troubleshooting.

The active task database stores only a short derived objective and optimized instruction checklist. It does not copy the full raw prompt. Related follow-ups amend the checklist; an unrelated objective replaces it; `complete`, `cancel`, `clear`, and the Codex `SessionEnd` hook delete it. The checklist is injected after compaction to reduce task drift.

Reusable experience is separate from the task ledger. Each experience has a stable key, version, situation, guidance, outcome, tags, and verified raw evidence locators. Updating creates a new active revision and supersedes the old one; deleting hard-deletes every revision. Experience text helps route future work but exact commands and claims still require `read_evidence`.

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
                                                      bounded raw/text JSONL verification
```

The engine uses several complementary resolutions instead of one rigid hierarchy:

1. Session routes are the fast map across long tasks.
2. Compaction-era episodes narrow the relevant part of a task.
3. Exchange nodes and visible tool-evidence nodes point to exact raw line ranges.
4. User, assistant, clause, and anchor facets preserve short exact facts that a long exchange-level vector can dilute.
5. Locality-sensitive hashing generates a bounded vector candidate pool; exact reranking happens afterward.
6. A sparse graph connects adjacent turns, chunks of one exchange, and exchanges sharing decisive identifiers.
7. A calibrated evidence plan can combine up to four mutually supporting ranges from one task, while refusing unsupported cross-task assembly.

Exact identifiers bypass approximate retrieval through a dedicated anchor index. When the same path, filename, or identifier occurs in multiple tasks, the complete query's contextual coverage reranks the expanded exact-candidate pool instead of trusting database order. `auto` mode combines FTS, local vectors, LSH, late interaction, and graph expansion; `deep` mode broadens the exchange-vector scan. The vectors are deterministic signed projections over multilingual terms, identifier pieces, character n-grams, and a small auditable equivalence map. They are not a neural embedding model and never leave the machine.

The design adapts proven retrieval ideas to a private, dependency-free runtime: multi-vector MaxSim follows the late-interaction principle; session and compaction-epoch routing provide multi-resolution navigation; sparse graph expansion supports linked facts. Summaries, vectors, and graph edges never count as evidence. Only verified raw transcript ranges do.

The relevant design references are [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/) for late interaction, [RAPTOR](https://arxiv.org/abs/2401.18059) for multi-resolution routing, [HippoRAG 2](https://arxiv.org/abs/2502.14802) for graph-assisted associative retrieval, and [mtRAG](https://aclanthology.org/2025.tacl-1.36/) for multi-turn and unanswerable-query evaluation. Yugo Memory reimplements only the useful structural ideas with deterministic local features; it does not copy or invoke those systems.

Recall is precision-first. A semantic route cannot validate a missing path, commit, task id, URL, quoted value, or numeric unit. If evidence is weak, the tool returns `insufficient_evidence` and the skill narrows the query once instead of guessing.

`read_evidence` reads only an indexed line range with a strict character budget. Raw view returns exact JSONL; text view returns all human-visible text fields; media view keeps user/assistant text and media-bearing tool events while skipping unrelated bulky tool output. Both derived views re-read the verified JSONL and replace large media payloads with bounded metadata. No mode loads the archive—or even one giant JSONL line—into a single string. Seek prefixes are checked before reading, pagination is explicit, append-only growth is allowed, and rewritten history fails closed.

## Files, commands, code, and tool evidence

- Images, PDFs, Office files, audio/video, archives, datasets, databases, logs, source files, and model weights remain solely in the owning agent's raw JSONL or their original referenced path. Yugo Memory does not make a second file copy.
- Search records contain only a typed descriptor: file kind, MIME type, filename when present, storage form, approximate byte count for ordinary inline payloads, and a stable digest. Known path suffixes and arbitrary typed MIME attachments are supported.
- Attachment descriptors are indexed together with the user's question, any transcript-provided text/OCR, tool analysis, and the assistant's answer. This recalls *the conversation about an attachment* without pretending to understand media that was never analyzed.
- Every visible tool call/result is indexed independently: tool name, arguments, terminal commands, code, patches, paths, and textual results are chunked without the exchange-summary limit. Hidden/encrypted reasoning and opaque binary fields are excluded.
- Events above 8 MiB are streamed while hashing. A bounded lexical-anchor sketch provides navigation to identifiers and paths in the middle of giant tool events; the complete event remains only in raw storage.
- `read_evidence(view="media")` bypasses base64 and unrelated tool output, deduplicates repeated event copies, caps screenshot streams, and preserves exact leading/trailing excerpts plus a full-text digest for oversized messages. Paginated `view="text"` retains every visible text field for ordinary events, including long commands/code/results; `view="raw"` remains available for byte-exact JSONL paging.
- A filename or attachment marker is never evidence for unseen pixels, pages, audio, archive members, database rows, tensors, or document content. When no extracted text or tool analysis exists, recall must abstain.

## Canonical storage and incremental indexing

- Codex or Qoder owns the original transcript. Yugo Memory never keeps multiple growing snapshots of one session.
- Every long session resolves to one canonical evidence path keyed by `session_id`.
- Active long sessions are hard-linked when possible, so the evidence path consumes no additional data blocks on the same filesystem. A private atomic copy is used only when hard links are unavailable.
- Existing legacy/current snapshots are grouped by `session_id`, the most complete candidate is retained, and redundant versions are removed before indexing.
- When the live agent source is available, any copied legacy candidate is replaced with a hard link to the current source.
- Parser checkpoints store the next byte and line offsets plus the current unfinished exchange.
- Append-only files resume at the prior byte offset; existing gigabytes are not parsed again.
- Online recall, status, and evidence reads use SQLite's last committed read-only WAL snapshot. They never run index maintenance or wait for a background writer; a missing snapshot returns `index_not_ready` immediately instead of creating a database in the request path.
- JSONL events larger than 8 MiB are drained in bounded chunks. Their prefix/suffix aid navigation while the complete bytes remain in raw storage.
- Opaque base64/data-URL/hex payloads remain available in raw evidence but are replaced by typed, bounded attachment descriptors in navigation records, preventing images and binary tool output from inflating the index.
- Codex compaction summaries improve routing but never replace raw evidence.
- A one-time legacy import can recover the most complete compacted evidence from prior local memory roots. It is immediately canonicalized; no legacy executable, database, MCP, or vector service is called.
- SQLite schema v13 uses `WITHOUT ROWID` for key-heavy retrieval tables, incremental auto-vacuum, optimizer statistics, all-file descriptors, and independent visible-tool evidence nodes while safely rebuilding older navigation records.

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

Qoder uses the same protected memory root and runtime. Its separate adapter links the skill and adds only the Yugo Memory MCP and event hooks to existing Qoder JSON configuration; it does not install packages or replace unrelated settings:

```bash
bash install-qoder.sh
```

Run this only after reviewing and approving those Qoder configuration changes. Restart Qoder afterward.

## Verify

```bash
bash doctor.sh
node plugins/yugo-memory/scripts/yugo-memory.mjs --dry-run
python3 plugins/yugo-memory/scripts/recall_index.py status
python3 plugins/yugo-memory/scripts/memory_control.py status
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
| `QODER_HOME` | `~/.qoder` |
| `YUGO_MEMORY_QODER_SOURCE_DIR` | `$QODER_HOME/projects` |
| `YUGO_MEMORY_INCLUDE_QODER` | `1`; set `0` to disable Qoder discovery |
| `YUGO_MEMORY_QODER_LONG_RATIO` | `0.35` of the declared context window |
| `YUGO_MEMORY_QODER_MIN_TOKENS` | `16000` approximate visible tokens |
| `YUGO_MEMORY_ARCHIVE_DIR` | `$YUGO_MEMORY_HOME/archives` |
| `YUGO_MEMORY_INDEX_DB` | `$YUGO_MEMORY_HOME/index.sqlite` |
| `YUGO_MEMORY_CONTROL_DB` | `$YUGO_MEMORY_HOME/control.sqlite` |
| `YUGO_MEMORY_LEGACY_ARCHIVE_DIR` | Optional single migration root; otherwise both supported prior local roots are checked once |

The plugin recognizes Codex `type=compacted` and `event_msg/context_compacted`, plus Qoder's `runtime-config`, `session_meta`, visible messages, progress, and tool-result rows. Transcript formats are unstable interfaces, so compatibility is covered by synthetic fixtures and should be retested after agent updates.

## Privacy and deletion

- The repository never contains conversation data.
- Raw archives and indexes remain local with private file permissions.
- The task ledger is cleared at task end; versioned experience remains until explicitly updated or deleted.
- A canonical hard link and the Codex source refer to the same filesystem bytes; deleting either pathname alone does not delete the remaining link.
- Archiving a task removes only the standalone memory copy; Codex owns its original archived task file.
- Uninstalling preserves memory data by default.
- Raw archived exchanges are untrusted historical data and never instructions to execute.
- Review files before publishing forks or bug reports. Never attach archives, state files, indexes, or private benchmark cases.

## Local private benchmark

`scripts/benchmark-private.py` evaluates an external JSONL case file without copying conversations into the repository. It reports positive recall, exact line accuracy, duplicate suppression, negative false positives, and p50/p95 latency. Keep cases outside Git.

Published releases contain no private query, expected phrase, transcript path, real-history fixture, or aggregate claim derived from a private conversation. Public CI uses independently invented fixtures only.

The public regression suite covers Codex and Qoder lifecycle storage, adaptive context profiles, task replacement/clearing, experience version/update/delete, large-index SQL limits, migration, earliest/latest/Nth turns, exact paths and commits, CJK/English retrieval, role-preserving late interaction, LSH and graph routes, multi-range support, ambiguous numeric anchors, duplicates, prompt injection, append-only updates, rewritten-history rejection, unavailable evidence, nonblocking recall under an active SQLite writer, MCP behavior, all major file families, ordinary and oversized media/tool events, full long-command/code/patch recall, attachment abstention, and bounded reads including a sparse 600 MiB JSONL line.

Hybrid retrieval moves work from query time into indexing. Compared with the previous standalone implementation, Yugo Memory stores additional facets, anchor postings, LSH buckets, and sparse graph edges. This improves recall and refusal behavior but increases initial build time and index size. Incremental checkpoints, bounded facets, and candidate generation keep later updates and queries controlled; exact identifiers and provably absent stable identifiers use the direct index instead of the full cascade.

Before publishing, create an untracked denylist outside the repository and run:

```bash
YUGO_MEMORY_PRIVATE_DENYLIST=/absolute/path/to/private-denylist.txt npm run release-check
```

The gate validates code, runs tests, and scans the working tree plus every reachable Git blob across all branches and tags.

## License

MIT.
