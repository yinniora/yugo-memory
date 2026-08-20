#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script="$repo_root/plugins/yugo-memory/scripts/yugo-memory.mjs"

node "$script" --doctor
echo
python3 "$repo_root/plugins/yugo-memory/scripts/recall_index.py" status
echo
python3 "$repo_root/plugins/yugo-memory/scripts/memory_control.py" status
echo
python3 "$repo_root/scripts/qoder-adapter.py" status --repo "$repo_root"
echo
codex plugin list
