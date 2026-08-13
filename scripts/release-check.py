#!/usr/bin/env python3
"""Run the mandatory local privacy and quality gate before a GitHub release."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    raw_path = os.environ.get("YUGO_MEMORY_PRIVATE_DENYLIST", "").strip()
    if not raw_path:
        print(
            "Refusing to publish: set YUGO_MEMORY_PRIVATE_DENYLIST to an untracked "
            "newline-delimited file containing private project terms, identifiers, and phrases.",
            file=sys.stderr,
        )
        return 2
    denylist = Path(raw_path).expanduser().resolve()
    if not denylist.is_file():
        print("Refusing to publish: the private denylist file does not exist.", file=sys.stderr)
        return 2
    try:
        denylist.relative_to(ROOT)
    except ValueError:
        pass
    else:
        print("Refusing to publish: keep the private denylist outside the repository.", file=sys.stderr)
        return 2

    run("npm", "run", "validate")
    run("npm", "test")
    run(
        sys.executable,
        str(ROOT / "scripts" / "privacy-scan.py"),
        "--root", str(ROOT),
        "--all-refs",
        "--denylist", str(denylist),
    )
    print("release check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
