#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for command_name in node python3; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
done

echo "Configuring Qoder to use the existing dependency-free Yugo Memory runtime..."
python3 "$repo_root/scripts/qoder-adapter.py" install --repo "$repo_root"
echo "Restart Qoder, then verify that the yugo-memory MCP and skill are available."
