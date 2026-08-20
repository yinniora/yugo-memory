#!/usr/bin/env node

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';

const DAY_MS = 24 * 60 * 60 * 1000;
const args = process.argv.slice(2);
const background = args.includes('--background');
const dryRun = args.includes('--dry-run');
const quiet = args.includes('--quiet');
const doctor = args.includes('--doctor');
const skipIndex = process.env.YUGO_MEMORY_SKIP_INDEX === '1';
const now = Date.now();

const codexHome = process.env.CODEX_HOME || path.join(os.homedir(), '.codex');
const configBase = process.env.XDG_CONFIG_HOME || path.join(os.homedir(), '.config');
const memoryRoot = process.env.YUGO_MEMORY_HOME || path.join(configBase, 'yugo-memory');
const sourceRoot = process.env.YUGO_MEMORY_SOURCE_DIR || path.join(codexHome, 'sessions');
const archivedSourceRoot = process.env.YUGO_MEMORY_ARCHIVED_SOURCE_DIR || path.join(codexHome, 'archived_sessions');
const qoderHome = process.env.QODER_HOME || path.join(os.homedir(), '.qoder');
const qoderSourceRoot = process.env.YUGO_MEMORY_QODER_SOURCE_DIR || path.join(qoderHome, 'projects');
const includeQoder = process.env.YUGO_MEMORY_INCLUDE_QODER !== '0';
const archiveRoot = process.env.YUGO_MEMORY_ARCHIVE_DIR || path.join(memoryRoot, 'archives');
const statePath = process.env.YUGO_MEMORY_STATE_PATH || path.join(memoryRoot, 'state.json');
const indexDb = process.env.YUGO_MEMORY_INDEX_DB || path.join(memoryRoot, 'index.sqlite');
const codexStateDb = process.env.YUGO_MEMORY_CODEX_STATE_DB || path.join(codexHome, 'state_5.sqlite');
const configuredLegacyRoot = process.env.YUGO_MEMORY_LEGACY_ARCHIVE_DIR;
const legacyArchiveRoots = configuredLegacyRoot
  ? [configuredLegacyRoot]
  : [
      path.join(configBase, 'codex-long-memory', 'archives'),
      path.join(configBase, 'superpowers', 'conversation-archive'),
    ];
const indexScript = path.join(path.dirname(process.argv[1]), 'recall_index.py');
const parserScript = path.join(path.dirname(process.argv[1]), 'archive_parser.py');
const deletedRetentionDays = Number(process.env.YUGO_MEMORY_DELETE_GRACE_DAYS || 7);
const deletedRetentionMs = deletedRetentionDays * DAY_MS;
const qoderLongRatio = Number(process.env.YUGO_MEMORY_QODER_LONG_RATIO || 0.35);
const qoderMinimumTokens = Number(process.env.YUGO_MEMORY_QODER_MIN_TOKENS || 16000);
const logDir = path.join(memoryRoot, 'logs');
const logPath = path.join(logDir, 'yugo-memory.log');
const lockPath = path.join(logDir, 'yugo-memory.lock');

function output(value) {
  if (!quiet) process.stdout.write(`${typeof value === 'string' ? value : JSON.stringify(value, null, 2)}\n`);
}

function ensurePrivateDir(dir) {
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  fs.chmodSync(dir, 0o700);
}

function walkJsonl(root) {
  const result = [];
  if (!fs.existsSync(root)) return result;
  const stack = [root];
  while (stack.length) {
    const dir = stack.pop();
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) stack.push(full);
      else if (entry.isFile() && entry.name.endsWith('.jsonl')) result.push(full);
    }
  }
  return result.sort();
}

function relativeMap(root) {
  return new Map(walkJsonl(root).map(file => [path.relative(root, file), file]));
}

function sessionIdFromFile(file) {
  const match = path.basename(file).match(/([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\.jsonl$/i);
  return match ? match[1] : null;
}

function archivedSessionMap(root) {
  const result = new Map();
  for (const file of walkJsonl(root)) {
    const sessionId = sessionIdFromFile(file);
    if (!sessionId) continue;
    const previous = result.get(sessionId);
    if (!previous) {
      result.set(sessionId, file);
      continue;
    }
    const candidateStat = fs.statSync(file);
    const previousStat = fs.statSync(previous);
    if (
      candidateStat.size > previousStat.size
      || (candidateStat.size === previousStat.size && candidateStat.mtimeMs > previousStat.mtimeMs)
    ) result.set(sessionId, file);
  }
  return result;
}

function canonicalArchivePath(sessionId) {
  return path.join(archiveRoot, 'by-session', sessionId.slice(0, 2), `${sessionId}.jsonl`);
}

function parseThreadRows(text) {
  const states = new Map();
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    const [id, archived] = line.split('\t');
    if (id) states.set(id, { archived: archived === '1' });
  }
  return states;
}

function loadThreadStates() {
  if (!fs.existsSync(codexStateDb)) return null;
  const sqlite = spawnSync('sqlite3', [
    '-readonly', '-separator', '\t', codexStateDb, 'select id, archived from threads;',
  ], { encoding: 'utf8' });
  if (sqlite.status === 0) return parseThreadRows(sqlite.stdout);

  const pythonCode = [
    'import sqlite3, sys',
    'db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)',
    'for row in db.execute("select id, archived from threads"):',
    '    print(f"{row[0]}\\t{int(bool(row[1]))}")',
  ].join('\n');
  const python = spawnSync('python3', ['-c', pythonCode, codexStateDb], { encoding: 'utf8' });
  return python.status === 0 ? parseThreadRows(python.stdout) : null;
}

function crossedCompactionBoundary(file) {
  // Search exact JSON keys in bounded binary chunks. Quoted JSON embedded in a
  // message is escaped, so it does not match these unescaped event markers.
  const needles = [Buffer.from('"type":"compacted"'), Buffer.from('"type":"context_compacted"')];
  const buffer = Buffer.allocUnsafe(64 * 1024);
  const fd = fs.openSync(file, 'r');
  let overlap = Buffer.alloc(0);
  try {
    while (true) {
      const read = fs.readSync(fd, buffer, 0, buffer.length, null);
      if (!read) return false;
      const combined = Buffer.concat([overlap, buffer.subarray(0, read)]);
      if (needles.some(needle => combined.includes(needle))) return true;
      overlap = combined.subarray(Math.max(0, combined.length - 64));
    }
  } finally {
    fs.closeSync(fd);
  }
}

function loadState() {
  if (!fs.existsSync(statePath)) {
    return { schemaVersion: 5, missingSince: {}, sourceStatus: {}, legacyMigrations: {} };
  }
  try {
    const state = JSON.parse(fs.readFileSync(statePath, 'utf8'));
    const compatible = Number(state?.schemaVersion || 0) >= 4;
    return {
      schemaVersion: 5,
      // v3 keyed these maps by relative paths. Resetting only delays final
      // deletion; it can never remove evidence before the configured grace.
      missingSince: compatible && state?.missingSince && typeof state.missingSince === 'object'
        ? state.missingSince : {},
      sourceStatus: compatible && state?.sourceStatus && typeof state.sourceStatus === 'object'
        ? state.sourceStatus : {},
      legacyMigrations: state?.legacyMigrations && typeof state.legacyMigrations === 'object'
        ? state.legacyMigrations : {},
    };
  } catch {
    return { schemaVersion: 5, missingSince: {}, sourceStatus: {}, legacyMigrations: {} };
  }
}

function probeLongSource(file) {
  const result = spawnSync('python3', [
    parserScript, 'probe', '--input', file,
    '--long-ratio', String(qoderLongRatio),
    '--minimum-tokens', String(qoderMinimumTokens),
  ], { encoding: 'utf8', env: process.env });
  if (result.status !== 0) {
    throw new Error(`conversation source probe failed (${result.status}): ${result.stderr}`);
  }
  return JSON.parse(result.stdout);
}

function saveState(state) {
  ensurePrivateDir(path.dirname(statePath));
  const temp = `${statePath}.tmp.${process.pid}`;
  fs.writeFileSync(temp, `${JSON.stringify(state, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 });
  fs.renameSync(temp, statePath);
  fs.chmodSync(statePath, 0o600);
}

function removeFile(file, removed) {
  if (!fs.existsSync(file)) return;
  const stat = fs.lstatSync(file);
  if (!stat.isFile() && !stat.isSymbolicLink()) throw new Error(`refusing to remove non-file path: ${file}`);
  if (!dryRun) fs.unlinkSync(file);
  removed.push(file);
}

function pruneEmptyParents(start, stop) {
  if (dryRun) return;
  let current = path.dirname(start);
  const resolvedStop = path.resolve(stop);
  while (path.resolve(current).startsWith(`${resolvedStop}${path.sep}`)) {
    try {
      fs.rmdirSync(current);
    } catch {
      return;
    }
    current = path.dirname(current);
  }
}

function copyOrLink(source, destination) {
  ensurePrivateDir(path.dirname(destination));
  if (fs.existsSync(destination)) {
    const sourceStat = fs.statSync(source);
    const destinationStat = fs.statSync(destination);
    if (sourceStat.dev === destinationStat.dev && sourceStat.ino === destinationStat.ino) return false;
    if (sourceStat.size === destinationStat.size && sourceStat.mtimeMs === destinationStat.mtimeMs) return false;
    const temp = `${destination}.tmp.${process.pid}`;
    fs.copyFileSync(source, temp);
    fs.chmodSync(temp, 0o600);
    fs.renameSync(temp, destination);
    return true;
  }
  let linked = true;
  try {
    fs.linkSync(source, destination);
  } catch (error) {
    if (!['EXDEV', 'EPERM', 'EACCES'].includes(error?.code)) throw error;
    linked = false;
    fs.copyFileSync(source, destination);
  }
  if (!linked) fs.chmodSync(destination, 0o600);
  return true;
}

function replaceWithSourceLink(source, destination) {
  ensurePrivateDir(path.dirname(destination));
  if (fs.existsSync(destination)) {
    const sourceStat = fs.statSync(source);
    const destinationStat = fs.statSync(destination);
    if (sourceStat.dev === destinationStat.dev && sourceStat.ino === destinationStat.ino) {
      return { changed: false, mode: 'hardlink' };
    }
  }
  const temp = `${destination}.tmp.${process.pid}`;
  try {
    fs.unlinkSync(temp);
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
  let mode = 'hardlink';
  try {
    fs.linkSync(source, temp);
  } catch (error) {
    if (!['EXDEV', 'EPERM', 'EACCES'].includes(error?.code)) throw error;
    mode = 'copy';
    fs.copyFileSync(source, temp);
    fs.chmodSync(temp, 0o600);
  }
  fs.renameSync(temp, destination);
  return { changed: true, mode };
}

function canonicalizeArchives() {
  const groups = new Map();
  for (const file of walkJsonl(archiveRoot)) {
    const sessionId = sessionIdFromFile(file);
    if (!sessionId) continue;
    if (!groups.has(sessionId)) groups.set(sessionId, []);
    groups.get(sessionId).push(file);
  }
  let canonicalized = 0;
  let duplicatesRemoved = 0;
  for (const [sessionId, files] of groups) {
    const canonical = canonicalArchivePath(sessionId);
    const best = [...files].sort((left, right) => {
      const leftStat = fs.statSync(left);
      const rightStat = fs.statSync(right);
      return rightStat.size - leftStat.size || rightStat.mtimeMs - leftStat.mtimeMs;
    })[0];
    if (dryRun) {
      if (path.resolve(best) !== path.resolve(canonical)) canonicalized += 1;
      duplicatesRemoved += files.length - 1;
      continue;
    }
    if (path.resolve(best) !== path.resolve(canonical)) {
      copyOrLink(best, canonical);
      canonicalized += 1;
    }
    for (const file of files) {
      if (path.resolve(file) === path.resolve(canonical)) continue;
      removeFile(file, []);
      pruneEmptyParents(file, archiveRoot);
      duplicatesRemoved += 1;
    }
  }
  return { canonicalized, duplicatesRemoved };
}

function refreshArchives(longSources) {
  if (dryRun) return { refreshed: longSources.size, hardlinked: 0, copied: 0 };
  ensurePrivateDir(archiveRoot);
  let refreshed = 0;
  let hardlinked = 0;
  let copied = 0;
  for (const [sessionId, source] of longSources) {
    const result = replaceWithSourceLink(source, canonicalArchivePath(sessionId));
    if (result.changed) refreshed += 1;
    if (result.mode === 'hardlink') hardlinked += 1;
    else copied += 1;
  }
  return { refreshed, hardlinked, copied };
}

async function migrateLegacyArchives(state) {
  let count = 0;
  for (const legacyRoot of legacyArchiveRoots) {
    const key = path.resolve(legacyRoot);
    if (state.legacyMigrations[key] || key === path.resolve(archiveRoot) || !fs.existsSync(key)) continue;
    for (const [relative, source] of relativeMap(key)) {
      if (!(await crossedCompactionBoundary(source))) continue;
      const destination = path.join(archiveRoot, relative);
      if (dryRun || copyOrLink(source, destination)) count += 1;
    }
    if (!dryRun) state.legacyMigrations[key] = true;
  }
  return count;
}

function runIndex() {
  if (skipIndex || dryRun) return null;
  const python = spawnSync('which', ['python3'], { encoding: 'utf8' });
  const pythonBin = python.status === 0 ? python.stdout.trim() : '';
  if (!pythonBin) throw new Error('python3 not found; it is required for standalone local recall');
  const result = spawnSync(pythonBin, [
    indexScript,
    'index',
    '--archive-root', archiveRoot,
    '--output', indexDb,
  ], {
    encoding: 'utf8',
    env: process.env,
  });
  if (result.status !== 0) throw new Error(`standalone recall indexing failed (${result.status}): ${result.stderr}`);
  if (!quiet && result.stdout.trim()) process.stdout.write(result.stdout);
  return result.stdout.trim() ? JSON.parse(result.stdout) : null;
}

function acquireLock() {
  ensurePrivateDir(logDir);
  try {
    const fd = fs.openSync(lockPath, 'wx', 0o600);
    fs.writeFileSync(fd, String(process.pid));
    fs.closeSync(fd);
    return true;
  } catch (error) {
    if (error?.code !== 'EEXIST') throw error;
    const age = now - fs.statSync(lockPath).mtimeMs;
    if (age <= 6 * 60 * 60 * 1000) return false;
    fs.unlinkSync(lockPath);
    const fd = fs.openSync(lockPath, 'wx', 0o600);
    fs.writeFileSync(fd, String(process.pid));
    fs.closeSync(fd);
    return true;
  }
}

function doctorReport() {
  const python = spawnSync('which', ['python3'], { encoding: 'utf8' });
  const fts5 = python.status === 0 ? spawnSync(python.stdout.trim(), [
    '-c', 'import sqlite3; db=sqlite3.connect(":memory:"); db.execute("create virtual table probe using fts5(text)")',
  ], { encoding: 'utf8' }) : { status: 1 };
  const threadStates = loadThreadStates();
  return {
    healthy: Boolean(python.status === 0 && fts5.status === 0 && fs.existsSync(sourceRoot)),
    node: process.version,
    python3: python.status === 0 ? python.stdout.trim() : null,
    sqliteFts5: fts5.status === 0,
    codexHome,
    sourceRoot,
    sourceSessions: walkJsonl(sourceRoot).length,
    qoderSourceRoot,
    qoderSourceSessions: includeQoder ? walkJsonl(qoderSourceRoot).length : 0,
    qoderEnabled: includeQoder,
    archivedSourceSessions: walkJsonl(archivedSourceRoot).length,
    codexStateDatabase: fs.existsSync(codexStateDb),
    codexThreadStateReadable: threadStates !== null,
    memoryRoot,
    archiveRoot,
    archivedLongSessions: walkJsonl(archiveRoot).length,
    indexDatabase: indexDb,
    indexReady: fs.existsSync(indexDb),
    runtimeDependency: 'none',
    remoteServerRequired: false,
    deleteGraceDays: deletedRetentionDays,
    eventDrivenOnly: true,
    legacyArchiveRoots,
  };
}

async function main() {
  const activeSources = archivedSessionMap(sourceRoot);
  const archivedSources = archivedSessionMap(archivedSourceRoot);
  const qoderSources = includeQoder ? archivedSessionMap(qoderSourceRoot) : new Map();
  const threadStates = loadThreadStates();
  const sources = new Map();
  for (const [sessionId, file] of activeSources) {
    const threadState = sessionId ? threadStates?.get(sessionId) : null;
    if (!threadStates || (threadState && !threadState.archived)) {
      sources.set(sessionId, { file, sourceAgent: 'codex' });
    }
  }
  for (const [sessionId, file] of qoderSources) {
    // UUID collisions across independent agents are vanishingly unlikely. If
    // one occurs, keep the Codex source and fail closed rather than merge two
    // raw conversations under one identity.
    if (!sources.has(sessionId)) sources.set(sessionId, { file, sourceAgent: 'qoder' });
  }

  const state = loadState();
  const legacyMigrations = await migrateLegacyArchives(state);
  const canonicalization = canonicalizeArchives();
  const longSources = new Map();
  for (const [sessionId, sourceInfo] of sources) {
    const { file, sourceAgent } = sourceInfo;
    const stat = fs.statSync(file);
    const statusKey = `${sourceAgent}:${sessionId}`;
    const cached = state.sourceStatus[statusKey] || state.sourceStatus[sessionId];
    let isLong;
    if (cached?.long === true) isLong = true;
    else if (cached?.size === stat.size && cached?.mtimeMs === stat.mtimeMs) isLong = false;
    else if (sourceAgent === 'codex') isLong = await crossedCompactionBoundary(file);
    else isLong = Boolean(probeLongSource(file).long);
    state.sourceStatus[statusKey] = {
      path: file, size: stat.size, mtimeMs: stat.mtimeMs, long: isLong, sourceAgent,
    };
    delete state.sourceStatus[sessionId];
    if (isLong) longSources.set(sessionId, file);
  }
  for (const statusKey of Object.keys(state.sourceStatus)) {
    const split = statusKey.indexOf(':');
    const sourceAgent = split >= 0 ? statusKey.slice(0, split) : 'codex';
    const sessionId = split >= 0 ? statusKey.slice(split + 1) : statusKey;
    if (sourceAgent === 'qoder' && !includeQoder) continue;
    // Preserve the Qoder source marker while its canonical archive is inside
    // the deletion grace period. Qoder does not expose deletion state through
    // Codex's state database and its long boundary need not be a compaction.
    if (sourceAgent === 'qoder' && fs.existsSync(canonicalArchivePath(sessionId))) continue;
    if (!sources.has(sessionId)) delete state.sourceStatus[statusKey];
  }

  const immediateDeletes = [];
  const expiredDeletes = [];
  const pendingDeletes = [];
  for (const [sessionId, archive] of archivedSessionMap(archiveRoot)) {
    if (!includeQoder && state.sourceStatus[`qoder:${sessionId}`]) {
      delete state.missingSince[sessionId];
      continue;
    }
    const threadState = sessionId ? threadStates?.get(sessionId) : null;
    const isKnownQoder = Boolean(state.sourceStatus[`qoder:${sessionId}`]);
    const archivedByFallback = !threadStates && sessionId && archivedSources.has(sessionId);
    if (threadState?.archived || archivedByFallback) {
      immediateDeletes.push({ sessionId, archive, reason: 'codex_thread_archived' });
      delete state.missingSince[sessionId];
      continue;
    }
    const source = sources.get(sessionId)?.file;
    if (source && !longSources.has(sessionId)) {
      immediateDeletes.push({ sessionId, archive, reason: 'below_compaction_boundary' });
      delete state.missingSince[sessionId];
      continue;
    }
    if (source || (threadStates && threadState && !threadState.archived)) {
      delete state.missingSince[sessionId];
      continue;
    }
    if (!isKnownQoder && !(await crossedCompactionBoundary(archive))) {
      immediateDeletes.push({ sessionId, archive, reason: 'legacy_short_archive' });
      delete state.missingSince[sessionId];
      continue;
    }
    const missingSince = Number(state.missingSince[sessionId] || now);
    state.missingSince[sessionId] = missingSince;
    if (now - missingSince >= deletedRetentionMs) {
      expiredDeletes.push({ sessionId, archive, missingSince });
    }
    else pendingDeletes.push({
      sessionId,
      missingSince: new Date(missingSince).toISOString(),
      deleteAfter: new Date(missingSince + deletedRetentionMs).toISOString(),
    });
  }

  output({
    mode: dryRun ? 'dry-run' : 'apply',
    runtimeDependency: 'none',
    remoteServerRequired: false,
    activeSourceSessions: activeSources.size,
    qoderSourceSessions: qoderSources.size,
    archivedSourceSessions: archivedSources.size,
    codexStateDatabaseAvailable: threadStates !== null,
    compactedLongSessions: longSources.size,
    belowBoundarySessions: sources.size - longSources.size,
    existingArchives: archivedSessionMap(archiveRoot).size,
    legacyMigrations,
    canonicalization,
    immediateDeletes: immediateDeletes.map(({ sessionId, reason }) => ({ sessionId, reason })),
    expiredDeletes: expiredDeletes.map(({ sessionId, missingSince }) => ({
      sessionId, missingSince: new Date(missingSince).toISOString(),
    })),
    pendingDeletes,
  });

  const removed = [];
  for (const item of [...immediateDeletes, ...expiredDeletes]) {
    removeFile(item.archive, removed);
    pruneEmptyParents(item.archive, archiveRoot);
    delete state.missingSince[item.sessionId];
    delete state.sourceStatus[`qoder:${item.sessionId}`];
  }
  const archiveRefresh = refreshArchives(longSources);
  const indexReport = runIndex();
  if (!dryRun) saveState(state);
  output({
    completed: true,
    archivedLongSessions: longSources.size,
    archiveRefresh,
    [dryRun ? 'plannedMemoryFileDeletes' : 'permanentlyDeletedMemoryFiles']: removed.length,
    pendingDeletedSessions: pendingDeletes.length,
    indexReport,
  });
}

if (doctor) {
  const report = doctorReport();
  output(report);
  process.exit(report.healthy ? 0 : 1);
}

if (background) {
  ensurePrivateDir(logDir);
  const logFd = fs.openSync(logPath, 'a', 0o600);
  const child = spawn(process.execPath, [process.argv[1], ...args.filter(arg => arg !== '--background')], {
    detached: true,
    stdio: ['ignore', logFd, logFd],
    env: process.env,
  });
  child.unref();
  output(`Long-memory maintenance started. Log: ${logPath}`);
  process.exit(0);
}

if (!acquireLock()) {
  output('Long-memory maintenance is already running; skipped.');
  process.exit(0);
}

try {
  await main();
} finally {
  try {
    fs.unlinkSync(lockPath);
  } catch {}
}
