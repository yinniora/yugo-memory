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
    schemaVersion: 2,
    missingSince: {
      [deletedRel]: Date.now() - 8 * 24 * 60 * 60 * 1000,
    },
    sourceStatus: {},
    legacyMigrations: {},
  }));

  const result = spawnSync(process.execPath, [memoryScript], {
    encoding: 'utf8',
    env: {
      ...process.env,
      CODEX_HOME: codexHome,
      YUGO_MEMORY_HOME: memoryRoot,
      YUGO_MEMORY_SKIP_INDEX: '1',
      YUGO_MEMORY_LEGACY_ARCHIVE_DIR: path.join(root, 'no-legacy'),
    },
  });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /codex_thread_archived/);
  assert.equal(fs.existsSync(path.join(memoryRoot, 'archives', longRel)), true);
  assert.equal(fs.existsSync(path.join(memoryRoot, 'archives', shortRel)), false);
  assert.equal(fs.existsSync(path.join(memoryRoot, 'archives', archivedRel)), false);
  assert.equal(fs.existsSync(path.join(memoryRoot, 'archives', deletedRel)), false);
  assert.equal(fs.readFileSync(path.join(codexHome, 'sessions', longRel), 'utf8').includes('full original detail'), true);
});

test('compact hook injects automatic recall context as valid JSON', () => {
  const result = spawnSync(process.execPath, [compactScript], {
    encoding: 'utf8',
    input: JSON.stringify({ session_id: 'test-session' }),
  });
  assert.equal(result.status, 0, result.stderr);
  const payload = JSON.parse(result.stdout);
  assert.equal(payload.hookSpecificOutput.hookEventName, 'SessionStart');
  assert.match(payload.hookSpecificOutput.additionalContext, /Yugo Memory recall tool/);
  assert.match(payload.hookSpecificOutput.additionalContext, /current_session_id=test-session/);
  assert.match(payload.hookSpecificOutput.additionalContext, /read_evidence/);
  assert.match(payload.hookSpecificOutput.additionalContext, /standalone/);
  assert.doesNotMatch(payload.hookSpecificOutput.additionalContext, /episodic-memory/i);
});

test('imports compacted archives from prior local memory roots without an upstream executable', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yugo-memory-migration-'));
  const configBase = path.join(root, 'config');
  const memoryRoot = path.join(root, 'yugo');
  const relative = path.join('2038', '01', 'fictional-migration.jsonl');
  const legacy = path.join(configBase, 'codex-long-memory', 'archives', relative);
  writeJsonl(legacy, [
    { type: 'response_item', payload: { role: 'user', content: 'fictional migration detail' } },
    { type: 'compacted', payload: { replacement: 'routing summary only' } },
  ]);
  const result = spawnSync(process.execPath, [memoryScript], {
    encoding: 'utf8',
    env: {
      ...process.env,
      CODEX_HOME: path.join(root, 'codex'),
      XDG_CONFIG_HOME: configBase,
      YUGO_MEMORY_HOME: memoryRoot,
      YUGO_MEMORY_SOURCE_DIR: path.join(root, 'empty-sessions'),
      YUGO_MEMORY_CODEX_STATE_DB: path.join(root, 'missing-state.sqlite'),
      YUGO_MEMORY_SKIP_INDEX: '1',
    },
  });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(fs.existsSync(path.join(memoryRoot, 'archives', relative)), true);
  assert.match(result.stdout, /"legacyMigrations": 1/);
});
