import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

const repoRoot = path.resolve(import.meta.dirname, '..');
const memoryScript = path.join(repoRoot, 'plugins', 'yugo-memory', 'scripts', 'yugo-memory.mjs');
const compactScript = path.join(repoRoot, 'plugins', 'yugo-memory', 'scripts', 'compact-recall-context.mjs');

function writeJsonl(file, rows) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${rows.map(row => JSON.stringify(row)).join('\n')}\n`);
}

function createStateDb(file, rows) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const code = [
    'import json, sqlite3, sys',
    'db = sqlite3.connect(sys.argv[1])',
    'db.execute("create table threads (id text primary key, archived integer not null)")',
    'db.executemany("insert into threads values (?, ?)", json.loads(sys.argv[2]))',
    'db.commit()',
  ].join('\n');
  const result = spawnSync('python3', ['-c', code, file, JSON.stringify(rows)], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
}

function canonicalArchive(memoryRoot, sessionId) {
  return path.join(memoryRoot, 'archives', 'by-session', sessionId.slice(0, 2), `${sessionId}.jsonl`);
}

test('stores only compacted sessions and applies lifecycle deletion policy', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yugo-memory-'));
  const codexHome = path.join(root, 'codex');
  const memoryRoot = path.join(root, 'memory');
  const day = path.join('2026', '08', '12');
  const longId = '11111111-1111-4111-8111-111111111111';
  const shortId = '22222222-2222-4222-8222-222222222222';
  const archivedId = '33333333-3333-4333-8333-333333333333';
  const deletedId = '44444444-4444-4444-8444-444444444444';
  const longRel = path.join(day, `rollout-long-${longId}.jsonl`);
  const shortRel = path.join(day, `rollout-short-${shortId}.jsonl`);
  const archivedRel = path.join(day, `rollout-archived-${archivedId}.jsonl`);
  const deletedRel = path.join(day, `rollout-deleted-${deletedId}.jsonl`);
  const compactedRows = [
    { type: 'response_item', payload: { role: 'user', content: 'full original detail' } },
    { type: 'compacted', payload: { replacement: 'summary used only as a boundary signal' } },
  ];

  writeJsonl(path.join(codexHome, 'sessions', longRel), compactedRows);
  writeJsonl(path.join(codexHome, 'sessions', shortRel), [
    { type: 'response_item', payload: { role: 'user', content: 'short task' } },
  ]);
  writeJsonl(path.join(codexHome, 'archived_sessions', `rollout-${archivedId}.jsonl`), compactedRows);
  writeJsonl(path.join(memoryRoot, 'archives', archivedRel), compactedRows);
  writeJsonl(path.join(memoryRoot, 'archives', deletedRel), compactedRows);
  createStateDb(path.join(codexHome, 'state_5.sqlite'), [
    [longId, 0],
    [shortId, 0],
    [archivedId, 1],
  ]);
  fs.mkdirSync(memoryRoot, { recursive: true });
  fs.writeFileSync(path.join(memoryRoot, 'state.json'), JSON.stringify({
    schemaVersion: 4,
    missingSince: {
      [deletedId]: Date.now() - 8 * 24 * 60 * 60 * 1000,
    },
    sourceStatus: {},
    legacyMigrations: {},
  }));

  const result = spawnSync(process.execPath, [memoryScript], {
    encoding: 'utf8',
    env: {
      ...process.env,
      YUGO_MEMORY_INCLUDE_QODER: '0',
      CODEX_HOME: codexHome,
      YUGO_MEMORY_HOME: memoryRoot,
      YUGO_MEMORY_SKIP_INDEX: '1',
      YUGO_MEMORY_LEGACY_ARCHIVE_DIR: path.join(root, 'no-legacy'),
    },
  });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /codex_thread_archived/);
  const longArchive = canonicalArchive(memoryRoot, longId);
  assert.equal(fs.existsSync(longArchive), true);
  assert.equal(fs.statSync(longArchive).ino, fs.statSync(path.join(codexHome, 'sessions', longRel)).ino);
  assert.equal(fs.existsSync(path.join(memoryRoot, 'archives', shortRel)), false);
  assert.equal(fs.existsSync(path.join(memoryRoot, 'archives', archivedRel)), false);
  assert.equal(fs.existsSync(path.join(memoryRoot, 'archives', deletedRel)), false);
  assert.equal(fs.readFileSync(path.join(codexHome, 'sessions', longRel), 'utf8').includes('full original detail'), true);
});

test('shares a context-qualified Qoder long session through the canonical archive', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yugo-memory-qoder-'));
  const codexHome = path.join(root, 'codex');
  const qoderHome = path.join(root, 'qoder');
  const memoryRoot = path.join(root, 'memory');
  const sessionId = '88888888-8888-4888-8888-888888888888';
  const source = path.join(qoderHome, 'projects', 'fictional', `${sessionId}.jsonl`);
  writeJsonl(source, [
    { type: 'runtime-config', sessionId, contextWindow: 4_000, model: 'fictional-model' },
    { type: 'session_meta', sessionId, cwd: '/demo/fictional', data: {} },
    { type: 'user', sessionId, message: { role: 'user', content: 'violet route '.repeat(1_200) } },
    { type: 'assistant', sessionId, message: { role: 'assistant', content: 'fictional complete' } },
  ]);
  const result = spawnSync(process.execPath, [memoryScript], {
    encoding: 'utf8',
    env: {
      ...process.env,
      YUGO_MEMORY_INCLUDE_QODER: '1',
      CODEX_HOME: codexHome,
      QODER_HOME: qoderHome,
      YUGO_MEMORY_HOME: memoryRoot,
      YUGO_MEMORY_SKIP_INDEX: '1',
      YUGO_MEMORY_QODER_LONG_RATIO: '0.05',
      YUGO_MEMORY_QODER_MIN_TOKENS: '1000',
      YUGO_MEMORY_LEGACY_ARCHIVE_DIR: path.join(root, 'no-legacy'),
    },
  });
  assert.equal(result.status, 0, result.stderr);
  const canonical = canonicalArchive(memoryRoot, sessionId);
  assert.equal(fs.existsSync(canonical), true);
  assert.equal(fs.statSync(canonical).ino, fs.statSync(source).ino);
  assert.match(result.stdout, /"qoderSourceSessions": 1/);

  fs.unlinkSync(source);
  const disabled = spawnSync(process.execPath, [memoryScript], {
    encoding: 'utf8',
    env: {
      ...process.env,
      CODEX_HOME: codexHome,
      QODER_HOME: qoderHome,
      YUGO_MEMORY_HOME: memoryRoot,
      YUGO_MEMORY_INCLUDE_QODER: '0',
      YUGO_MEMORY_SKIP_INDEX: '1',
      YUGO_MEMORY_DELETE_GRACE_DAYS: '0',
      YUGO_MEMORY_LEGACY_ARCHIVE_DIR: path.join(root, 'no-legacy'),
    },
  });
  assert.equal(disabled.status, 0, disabled.stderr);
  assert.equal(fs.existsSync(canonical), true);

  const expired = spawnSync(process.execPath, [memoryScript], {
    encoding: 'utf8',
    env: {
      ...process.env,
      CODEX_HOME: codexHome,
      QODER_HOME: qoderHome,
      YUGO_MEMORY_HOME: memoryRoot,
      YUGO_MEMORY_INCLUDE_QODER: '1',
      YUGO_MEMORY_SKIP_INDEX: '1',
      YUGO_MEMORY_DELETE_GRACE_DAYS: '0',
      YUGO_MEMORY_LEGACY_ARCHIVE_DIR: path.join(root, 'no-legacy'),
    },
  });
  assert.equal(expired.status, 0, expired.stderr);
  assert.equal(fs.existsSync(canonical), false);
});

test('compact hook injects automatic recall context as valid JSON', () => {
  const result = spawnSync(process.execPath, [compactScript], {
    encoding: 'utf8',
    input: JSON.stringify({ session_id: 'test-session' }),
  });
  assert.equal(result.status, 0, result.stderr);
  const payload = JSON.parse(result.stdout);
  assert.equal(payload.hookSpecificOutput.hookEventName, 'SessionStart');
  assert.match(payload.hookSpecificOutput.additionalContext, /Yugo Memory continuity/);
  assert.match(payload.hookSpecificOutput.additionalContext, /current_session_id=test-session/);
  assert.match(payload.hookSpecificOutput.additionalContext, /read_evidence/);
  assert.match(payload.hookSpecificOutput.additionalContext, /budgets output automatically/);
  assert.doesNotMatch(payload.hookSpecificOutput.additionalContext, /episodic-memory/i);
});

test('compact hook resolves exact nested thread metadata', () => {
  const result = spawnSync(process.execPath, [compactScript], {
    encoding: 'utf8',
    input: JSON.stringify({ _meta: { context: { threadId: 'fictional-nested-thread' } } }),
  });
  assert.equal(result.status, 0, result.stderr);
  const payload = JSON.parse(result.stdout);
  assert.match(payload.hookSpecificOutput.additionalContext, /current_session_id=fictional-nested-thread/);
  assert.match(payload.hookSpecificOutput.additionalContext, /task_update\(action=auto, profile=minimal\)/);
});

test('imports compacted archives from prior local memory roots without an upstream executable', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yugo-memory-migration-'));
  const configBase = path.join(root, 'config');
  const memoryRoot = path.join(root, 'yugo');
  const sessionId = '55555555-5555-4555-8555-555555555555';
  const relative = path.join('2038', '01', `fictional-migration-${sessionId}.jsonl`);
  const legacy = path.join(configBase, 'codex-long-memory', 'archives', relative);
  writeJsonl(legacy, [
    { type: 'response_item', payload: { role: 'user', content: 'fictional migration detail' } },
    { type: 'compacted', payload: { replacement: 'routing summary only' } },
  ]);
  const result = spawnSync(process.execPath, [memoryScript], {
    encoding: 'utf8',
    env: {
      ...process.env,
      YUGO_MEMORY_INCLUDE_QODER: '0',
      CODEX_HOME: path.join(root, 'codex'),
      XDG_CONFIG_HOME: configBase,
      YUGO_MEMORY_HOME: memoryRoot,
      YUGO_MEMORY_SOURCE_DIR: path.join(root, 'empty-sessions'),
      YUGO_MEMORY_CODEX_STATE_DB: path.join(root, 'missing-state.sqlite'),
      YUGO_MEMORY_SKIP_INDEX: '1',
    },
  });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(fs.existsSync(canonicalArchive(memoryRoot, sessionId)), true);
  assert.equal(
    fs.readdirSync(path.join(memoryRoot, 'archives', 'by-session', sessionId.slice(0, 2))).length,
    1,
  );
  assert.match(result.stdout, /"legacyMigrations": 1/);
});

test('collapses growing legacy snapshots to one canonical session and relinks the live source', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yugo-memory-canonical-'));
  const codexHome = path.join(root, 'codex');
  const memoryRoot = path.join(root, 'memory');
  const sessionId = '66666666-6666-4666-8666-666666666666';
  const source = path.join(
    codexHome, 'sessions', '2038', '02', `rollout-live-${sessionId}.jsonl`,
  );
  const rows = [
    { type: 'response_item', payload: { role: 'user', content: 'fictional canonical detail' } },
    { type: 'compacted', payload: { replacement: 'routing summary' } },
    { type: 'response_item', payload: { role: 'assistant', content: 'latest source tail' } },
  ];
  writeJsonl(source, rows);
  writeJsonl(
    path.join(memoryRoot, 'archives', 'legacy-a', `snapshot-${sessionId}.jsonl`),
    rows.slice(0, 2),
  );
  writeJsonl(
    path.join(memoryRoot, 'archives', 'legacy-b', `snapshot-${sessionId}.jsonl`),
    rows,
  );
  createStateDb(path.join(codexHome, 'state_5.sqlite'), [[sessionId, 0]]);

  const result = spawnSync(process.execPath, [memoryScript], {
    encoding: 'utf8',
    env: {
      ...process.env,
      YUGO_MEMORY_INCLUDE_QODER: '0',
      CODEX_HOME: codexHome,
      YUGO_MEMORY_HOME: memoryRoot,
      YUGO_MEMORY_SKIP_INDEX: '1',
      YUGO_MEMORY_LEGACY_ARCHIVE_DIR: path.join(root, 'no-legacy'),
    },
  });
  assert.equal(result.status, 0, result.stderr);
  const canonical = canonicalArchive(memoryRoot, sessionId);
  assert.equal(fs.statSync(canonical).ino, fs.statSync(source).ino);
  assert.equal(
    fs.readdirSync(path.join(memoryRoot, 'archives', 'by-session', sessionId.slice(0, 2))).length,
    1,
  );
  assert.equal(fs.existsSync(path.join(memoryRoot, 'archives', 'legacy-a')), false);
  assert.equal(fs.existsSync(path.join(memoryRoot, 'archives', 'legacy-b')), false);
  assert.match(result.stdout, /"duplicatesRemoved": 2/);
  assert.match(result.stdout, /"hardlinked": 1/);
});

test('canonical hard link preserves the seven-day deletion grace without duplicate live blocks', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yugo-memory-grace-'));
  const codexHome = path.join(root, 'codex');
  const memoryRoot = path.join(root, 'memory');
  const sessionId = '77777777-7777-4777-8777-777777777777';
  const source = path.join(
    codexHome, 'sessions', '2038', '03', `rollout-grace-${sessionId}.jsonl`,
  );
  writeJsonl(source, [
    { type: 'response_item', payload: { role: 'user', content: 'fictional grace detail' } },
    { type: 'compacted', payload: { replacement: 'routing summary' } },
  ]);
  const stateDb = path.join(codexHome, 'state_5.sqlite');
  createStateDb(stateDb, [[sessionId, 0]]);
  const env = {
    ...process.env,
    YUGO_MEMORY_INCLUDE_QODER: '0',
    CODEX_HOME: codexHome,
    YUGO_MEMORY_HOME: memoryRoot,
    YUGO_MEMORY_SKIP_INDEX: '1',
    YUGO_MEMORY_LEGACY_ARCHIVE_DIR: path.join(root, 'no-legacy'),
  };
  let result = spawnSync(process.execPath, [memoryScript], { encoding: 'utf8', env });
  assert.equal(result.status, 0, result.stderr);
  const canonical = canonicalArchive(memoryRoot, sessionId);
  assert.equal(fs.statSync(canonical).ino, fs.statSync(source).ino);

  fs.unlinkSync(source);
  fs.unlinkSync(stateDb);
  createStateDb(stateDb, []);
  fs.writeFileSync(path.join(memoryRoot, 'state.json'), JSON.stringify({
    schemaVersion: 4,
    missingSince: { [sessionId]: Date.now() - 8 * 24 * 60 * 60 * 1000 },
    sourceStatus: {},
    legacyMigrations: {},
  }));
  result = spawnSync(process.execPath, [memoryScript], { encoding: 'utf8', env });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /"expiredDeletes"/);
  assert.equal(fs.existsSync(canonical), false);
});
