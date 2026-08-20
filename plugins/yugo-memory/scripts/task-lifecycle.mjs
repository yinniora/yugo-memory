#!/usr/bin/env node

import path from 'node:path';
import { spawnSync } from 'node:child_process';

let input = '';
for await (const chunk of process.stdin) input += chunk;
let hookInput = {};
try {
  hookInput = input.trim() ? JSON.parse(input) : {};
} catch {}

const sessionId = hookInput.session_id || hookInput.sessionId || '';
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
