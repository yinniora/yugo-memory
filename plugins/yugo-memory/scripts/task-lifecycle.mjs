#!/usr/bin/env node

import path from 'node:path';
import { spawnSync } from 'node:child_process';

let input = '';
for await (const chunk of process.stdin) input += chunk;
let hookInput = {};
try {
  hookInput = input.trim() ? JSON.parse(input) : {};
} catch {}

function findSessionId(value, depth = 0) {
  if (!value || typeof value !== 'object' || depth > 2) return '';
  for (const key of ['session_id', 'sessionId', 'thread_id', 'threadId', 'conversation_id', 'conversationId']) {
    if (typeof value[key] === 'string' && value[key].trim()) return value[key].trim();
  }
  for (const key of ['session', 'thread', 'conversation', 'context', 'client', '_meta']) {
    const nested = findSessionId(value[key], depth + 1);
    if (nested) return nested;
  }
  return '';
}

const sessionId = findSessionId(hookInput);
if (!sessionId) process.exit(0);
const controlScript = path.join(path.dirname(process.argv[1]), 'memory_control.py');
const result = spawnSync('python3', [controlScript, 'clear-task', '--session-id', sessionId], {
  encoding: 'utf8',
  env: process.env,
});
if (result.status !== 0) {
  process.stderr.write(result.stderr || 'Yugo Memory task cleanup failed.\n');
  process.exit(result.status || 1);
}
