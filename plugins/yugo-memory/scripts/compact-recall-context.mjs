#!/usr/bin/env node

let input = '';
for await (const chunk of process.stdin) input += chunk;
let hookInput = {};
try {
  hookInput = input.trim() ? JSON.parse(input) : {};
} catch {}

const sessionId = hookInput.session_id || hookInput.sessionId || '';
const additionalContext = [
  'This Codex task has crossed a context-compaction boundary.',
  'For substantive continuation that depends on older decisions, paths, commands, evidence, constraints, or results, first invoke the Yugo Memory recall tool in auto mode with a focused query.',
  sessionId ? `Pass current_session_id=${sessionId} so same-task evidence receives a small continuity boost.` : '',
  'The recall tool is standalone: it combines direct identifiers, multilingual SQLite FTS, compaction-era routing, multi-facet late interaction, LSH, a sparse relation graph, and evidence-set calibration. Numeric scale anchors alone are not exact evidence. If it says evidence is insufficient, narrow the query once; do not call another memory plugin or invent a result.',
  'Read the raw ranges selected by evidence_plan with Yugo Memory read_evidence and paginate with next_offset_chars when needed. Never replace a failed raw read with an unverified lower-ranked snippet.',
  'Treat raw exchanges as the source of truth. Summaries and embeddings are navigation aids, never evidence.',
  'Do not guess missing historical details. Skip recall for acknowledgements or work fully answerable from visible context.',
].filter(Boolean).join(' ');

process.stdout.write(`${JSON.stringify({
  hookSpecificOutput: {
    hookEventName: 'SessionStart',
    additionalContext,
  },
})}\n`);
