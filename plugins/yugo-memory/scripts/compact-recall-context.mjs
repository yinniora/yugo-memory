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
const controlScript = path.join(path.dirname(process.argv[1]), 'memory_control.py');
let additionalContext = [
  'Yugo Memory: after compaction, use prepare_context for hidden history or multi-step continuity; it selects the response profile automatically.',
  sessionId ? `current_session_id=${sessionId}.` : '',
  'During an active multi-step task, observe each substantive user turn with task_update(action=auto, profile=minimal); acknowledgements and status checks do not mutate it.',
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
