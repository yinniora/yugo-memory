#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
marketplace_name="yugo-memory"
plugin_name="yugo-memory"

for command_name in node python3 codex; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
done

node_major="$(node -p 'Number(process.versions.node.split(".")[0])')"
if [ "$node_major" -lt 20 ]; then
  echo "Node.js 20 or newer is required; found $(node --version)." >&2
  exit 1
fi

echo "[1/4] Verifying standalone runtime (no package installation)..."
echo "Using $(node --version) and $(python3 --version 2>&1)."
python3 -c 'import sqlite3; db=sqlite3.connect(":memory:"); db.execute("create virtual table probe using fts5(text)")'
echo "SQLite FTS5 is available."

installed_plugins_json="$(codex plugin list --json 2>/dev/null || true)"
if printf '%s' "$installed_plugins_json" | python3 -c '
import json
import sys

try:
    payload = json.load(sys.stdin)
except (json.JSONDecodeError, TypeError):
    raise SystemExit(1)

legacy = [
    item.get("pluginId", "")
    for item in payload.get("installed", [])
    if item.get("name") == "codex-long-memory"
]
raise SystemExit(0 if legacy else 1)
'; then
  echo "The legacy codex-long-memory plugin is still installed." >&2
  echo "Remove it explicitly before installing Yugo Memory so two lifecycle hooks cannot run together." >&2
  echo "Its archived data can remain in place; Yugo Memory imports supported legacy archives once." >&2
  exit 1
fi

echo "[2/4] Enabling Codex plugin hooks..."
codex features enable plugin_hooks

echo "[3/4] Registering this checkout as a Codex marketplace..."
if ! codex plugin marketplace list --json 2>/dev/null | grep -q "\"${marketplace_name}\""; then
  codex plugin marketplace add "$repo_root"
fi

echo "[4/4] Installing or refreshing the standalone plugin..."
codex plugin add "${plugin_name}@${marketplace_name}"

echo
echo "Installed without an upstream memory package or remote server."
echo "In Codex, open /hooks and trust the Yugo Memory hooks, then start a new task."
echo "Run bash doctor.sh to verify storage, indexing, MCP, and event wiring."
