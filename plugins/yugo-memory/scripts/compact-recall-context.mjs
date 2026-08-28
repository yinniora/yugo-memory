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
  if (!value || typeof value !== 'object' || depth > 6) return '';
  if (Array.isArray(value)) {
    for (const item of value) {
      const nested = findSessionId(item, depth + 1);
      if (nested) return nested;
    }
    return '';
  }
  for (const key of ['session_id', 'sessionId', 'thread_id', 'threadId', 'conversation_id', 'conversationId']) {
    if (typeof value[key] === 'string' && value[key].trim()) return value[key].trim();
  }
  for (const key of ['session', 'thread', 'conversation', 'context', 'client', '_meta', 'hook', 'payload', 'data', 'event', 'details']) {
    const nested = findSessionId(value[key], depth + 1);
    if (nested) return nested;
  }
  return '';
}

const sessionId = findSessionId(hookInput)
  || process.env.CODEX_THREAD_ID
  || process.env.QODER_SESSION_ID
  || process.env.YUGO_MEMORY_SESSION_ID
  || '';
const controlScript = path.join(path.dirname(process.argv[1]), 'memory_control.py');
let additionalContext = [
  'Yugo Memory: after compaction, use read-only prepare_context for hidden history or continuity; it selects the response profile automatically.',
  sessionId ? `current_session_id=${sessionId}.` : '',
  'Task checkpoints are optional and only preserve durable user constraints, acceptance criteria, and blockers; do not mirror every turn or ordinary follow-up into them.',
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
