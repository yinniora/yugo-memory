#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$repo_root/scripts/qoder-adapter.py" uninstall
echo "Qoder integration was removed. Shared Yugo Memory data was preserved."
