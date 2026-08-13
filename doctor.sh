#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script="$repo_root/plugins/yugo-memory/scripts/yugo-memory.mjs"

node "$script" --doctor
echo
python3 "$repo_root/plugins/yugo-memory/scripts/recall_index.py" status
echo
codex plugin list
