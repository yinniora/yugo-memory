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
const controlScript = path.join(path.dirname(process.argv[1]), 'memory_control.py');
let additionalContext = [
  'Yugo Memory: after compaction, use prepare_context for hidden history or multi-step continuity; it selects the response profile automatically.',
  sessionId ? `current_session_id=${sessionId}.` : '',
  'Verify exact facts with read_evidence; summaries are navigation only; abstain when evidence is insufficient.',
].filter(Boolean).join(' ');
if (sessionId) {
  const result = spawnSync('python3', [controlScript, 'compact-hint', '--session-id', sessionId], {
    encoding: 'utf8',
    env: process.env,
  });
  if (result.status === 0) {
    try {
      const parsed = JSON.parse(result.stdout);
      if (parsed.additional_context) additionalContext = parsed.additional_context;
    } catch {}
  }
}

process.stdout.write(`${JSON.stringify({
  hookSpecificOutput: {
    hookEventName: 'SessionStart',
    additionalContext,
  },
})}\n`);
