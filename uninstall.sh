#!/usr/bin/env bash
set -euo pipefail

codex plugin remove yugo-memory 2>/dev/null || true
codex plugin marketplace remove yugo-memory 2>/dev/null || true

echo "Yugo Memory was removed."
echo "Standalone conversation archives and indexes were intentionally preserved."
echo "Remove ~/.config/yugo-memory separately only after making a backup."
