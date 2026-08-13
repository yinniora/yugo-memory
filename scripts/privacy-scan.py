#!/usr/bin/env python3
"""Fail closed when a release tree or its Git history contains private material."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".7z", ".db", ".gz", ".jsonl", ".key", ".pem", ".pdf", ".sqlite",
    ".sqlite3", ".tar", ".tgz", ".zip",
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    revision: str | None = None


def _patterns(extra_literals: tuple[str, ...] = ()) -> tuple[tuple[str, re.Pattern[str]], ...]:
    # Assemble sensitive literals so this scanner does not flag its own source.
    users_path = "/" + "Us" + "ers/"
    workspace_path = "/mnt/" + "work" + "space/"
    oss_path = "/data/" + "oss"
    internal_domains = "(?:ali" + "baba-inc|aliyuque\\.ant" + "fin|ata\\.ata" + "tech)\\.com"
    attachment_path = "\\.co" + "dex/attachments/"
    delegation = "<codex_" + "delegation>"
    browser_context = "<in-app-browser-" + "context"
    base = (
        ("personal-home-path", re.compile(re.escape(users_path) + r"[^/\s\"']+/")),
        ("remote-workspace-path", re.compile(re.escape(workspace_path))),
        ("private-oss-path", re.compile(re.escape(oss_path))),
        ("internal-domain", re.compile(internal_domains, re.IGNORECASE)),
        ("codex-attachment", re.compile(attachment_path, re.IGNORECASE)),
        ("raw-delegation", re.compile(re.escape(delegation), re.IGNORECASE)),
        ("raw-browser-context", re.compile(re.escape(browser_context), re.IGNORECASE)),
        ("real-rollout-id", re.compile(r"rollout-\d{4}-\d{2}-\d{2}T[^\s/]*-019[0-9a-f]{29,}")),
        ("real-thread-id", re.compile(r"\b019[0-9a-f]{29,}\b")),
        ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
        ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{8,}\b")),
        (
            "credential-assignment",
            re.compile(
                r"(?i)\b(?:password|passwd|private[_ -]?token|access[_ -]?token|api[_ -]?key|secret[_ -]?key)\b"
                r"\s*(?:=|:)\s*[\"']?[A-Za-z0-9_./+:-]{12,}"
            ),
        ),
        ("email-address", re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")),
    )
    return base + tuple(("private-denylist", re.compile(re.escape(item), re.IGNORECASE)) for item in extra_literals)


def _run_git(root: Path, args: list[str], check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=check)


def candidate_files(root: Path) -> list[Path]:
    run = _run_git(root, ["ls-files", "--cached", "--others", "--exclude-standard", "-z"])
    return [root / item.decode("utf-8", errors="surrogateescape") for item in run.stdout.split(b"\0") if item]


def _scan_bytes(
    path: str,
    raw: bytes,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
    revision: str | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    suffix = Path(path).suffix.lower()
    if suffix in FORBIDDEN_SUFFIXES:
        findings.append(Finding(path, 0, "forbidden-publishable-file", revision))
    if b"\0" in raw[:8192]:
        findings.append(Finding(path, 0, "binary-artifact", revision))
        return findings
    text = raw.decode("utf-8", errors="replace")
    for number, line in enumerate(text.splitlines(), 1):
        for rule, pattern in patterns:
            if pattern.search(line):
                findings.append(Finding(path, number, rule, revision))
    return findings


def scan_files(
    root: Path,
    files: list[Path],
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> list[Finding]:
    findings: list[Finding] = []
    scanner = Path(__file__).resolve()
    for path in files:
        if not path.is_file() or path.resolve() == scanner:
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        findings.extend(_scan_bytes(path.relative_to(root).as_posix(), raw, patterns))
    return findings


def history_blobs(root: Path, refs: list[str]) -> list[tuple[str, str, bytes]]:
    if not refs:
        return []
    run = _run_git(root, ["rev-list", "--objects", *refs], check=False)
    if run.returncode != 0:
        # A brand-new repository has no HEAD yet; its working tree is still scanned.
        return []
    objects: dict[str, str] = {}
    for raw_line in run.stdout.splitlines():
        oid, _, raw_path = raw_line.partition(b" ")
        objects.setdefault(oid.decode("ascii"), raw_path.decode("utf-8", errors="surrogateescape"))
    blobs: list[tuple[str, str, bytes]] = []
    for oid, path in objects.items():
        object_type = _run_git(root, ["cat-file", "-t", oid]).stdout.strip()
        if object_type != b"blob":
            continue
        blobs.append((path or f"<blob:{oid[:12]}>", oid, _run_git(root, ["cat-file", "blob", oid]).stdout))
    return blobs


def scan_history(
    blobs: list[tuple[str, str, bytes]],
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> list[Finding]:
    findings: list[Finding] = []
    for path, oid, raw in blobs:
        if path == "scripts/privacy-scan.py":
            continue
        findings.extend(_scan_bytes(path, raw, patterns, oid))
    return findings


def load_denylist(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#") and len(value) >= 4:
            values.append(value)
    return tuple(dict.fromkeys(values))


def all_refs(root: Path) -> list[str]:
    run = _run_git(root, ["for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes", "refs/tags"])
    return [line.decode("utf-8") for line in run.stdout.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--denylist", type=Path, help="Untracked newline-delimited private phrases; values are never echoed")
    parser.add_argument("--history-ref", action="append", default=[], help="Git revision whose reachable blobs must be scanned")
    parser.add_argument("--all-refs", action="store_true", help="Scan every local branch, remote-tracking branch, and tag")
    args = parser.parse_args()
    root = args.root.resolve()
    patterns = _patterns(load_denylist(args.denylist))
    files = candidate_files(root)
    refs = all_refs(root) if args.all_refs else (args.history_ref or ["HEAD"])
    blobs = history_blobs(root, refs)
    findings = scan_files(root, files, patterns) + scan_history(blobs, patterns)
    findings = list(dict.fromkeys(findings))
    payload = {
        "passed": not findings,
        "files_scanned": len(files),
        "history_blobs_scanned": len(blobs),
        "history_refs": refs,
        "findings": [asdict(item) for item in findings],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif findings:
        for finding in findings:
            revision = f"@{finding.revision[:12]}" if finding.revision else ""
            print(f"{finding.path}{revision}:{finding.line}: {finding.rule}")
        print(f"privacy scan failed with {len(findings)} finding(s)")
    else:
        print(
            f"privacy scan passed ({payload['files_scanned']} working-tree files, "
            f"{payload['history_blobs_scanned']} historical blobs)"
        )
    return 0 if not findings else 2


if __name__ == "__main__":
    raise SystemExit(main())
