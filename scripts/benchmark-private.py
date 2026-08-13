#!/usr/bin/env python3
"""Benchmark private long-conversation recall cases without copying transcripts.

Case file JSONL fields:
  id, query, expected_session_id (optional), expected_phrase (optional),
  expected_line_start/expected_line_end (optional), expected_targets (optional),
  expected_answerable (default true), max_rank (default 3), mode (optional).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "plugins" / "yugo-memory" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from recall_index import default_paths, search_index  # noqa: E402


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def phrase_in_range(result: dict, phrase: str) -> bool:
    path = Path(result["archive_path"])
    if not path.is_file():
        return False
    start = result.get("context_line_start", result["line_start"])
    end = result.get("context_line_end", result["line_end"])
    needle = phrase.encode("utf-8")
    overlap = b""
    line = 1
    with path.open("rb") as handle:
        while line <= int(end):
            chunk = handle.read(64 * 1024)
            if not chunk:
                return False
            pieces = chunk.splitlines(keepends=True)
            for piece in pieces:
                if line >= int(start) and needle in overlap + piece:
                    return True
                overlap = (overlap + piece)[-max(0, len(needle) - 1):]
                if piece.endswith(b"\n"):
                    line += 1
                    overlap = b""
                    if line > int(end):
                        break
    return False


def _is_single_target(result: dict, case: dict) -> bool:
    if case.get("expected_session_id") and result["session_id"] != case["expected_session_id"]:
        return False
    if case.get("expected_line_start") is not None:
        wanted_start = int(case["expected_line_start"])
        wanted_end = int(case.get("expected_line_end", wanted_start))
        if result["line_end"] < wanted_start or result["line_start"] > wanted_end:
            return False
    if case.get("expected_phrase") and not phrase_in_range(result, case["expected_phrase"]):
        return False
    return bool(case.get("expected_session_id") or case.get("expected_line_start") is not None or case.get("expected_phrase"))


def is_relevant(result: dict, case: dict) -> bool:
    targets = case.get("expected_targets")
    if not targets:
        return _is_single_target(result, case)
    common = {key: value for key, value in case.items() if key.startswith("expected_") and key != "expected_targets"}
    return any(_is_single_target(result, {**common, **target}) for target in targets)


def main() -> int:
    _, default_index = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=default_index)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--default-mode", choices=("fast", "auto", "deep"), default="auto")
    args = parser.parse_args()

    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    hybrid_latencies = []
    hybrid_reciprocal_ranks: list[float] = []
    positive_rows: list[dict] = []
    negative_rows: list[dict] = []
    for case in cases:
        started = time.perf_counter()
        hybrid = search_index(args.index, case["query"], limit=10, mode=case.get("mode", args.default_mode))
        hybrid_elapsed = (time.perf_counter() - started) * 1000
        hybrid_latencies.append(hybrid_elapsed)
        expected_answerable = bool(case.get("expected_answerable", True))
        hybrid_rank = next((index for index, result in enumerate(hybrid["results"], 1) if is_relevant(result, case)), None) if expected_answerable else None
        hashes = [result["content_hash"] for result in hybrid["results"]]
        max_rank = int(case.get("max_rank", 3))
        passed = (
            bool(hybrid_rank and hybrid_rank <= max_rank and hybrid.get("safe_to_answer"))
            if expected_answerable else not hybrid.get("safe_to_answer")
        )
        row = {
                "id": case.get("id", f"case-{len(rows) + 1:03d}"),
                "expected_answerable": expected_answerable,
                "hybrid_rank": hybrid_rank,
                "max_accepted_rank": max_rank if expected_answerable else None,
                "hybrid_safe_to_answer": bool(hybrid.get("safe_to_answer")),
                "duplicate_free": len(hashes) == len(set(hashes)),
                "hybrid_latency_ms": round(hybrid_elapsed, 2),
                "top_confidence": hybrid["results"][0]["confidence"] if hybrid["results"] else None,
                "passed": passed,
            }
        rows.append(row)
        if expected_answerable:
            positive_rows.append(row)
            hybrid_reciprocal_ranks.append(1.0 / hybrid_rank if hybrid_rank else 0.0)
        else:
            negative_rows.append(row)

    count = len(rows)
    def metrics(prefix: str, reciprocal_ranks: list[float]) -> dict:
        rank_key = f"{prefix}_rank"
        denominator = len(positive_rows)
        return {
            "recall_at_1": sum(bool(row[rank_key] and row[rank_key] <= 1) for row in positive_rows) / denominator if denominator else 0,
            "recall_at_3": sum(bool(row[rank_key] and row[rank_key] <= 3) for row in positive_rows) / denominator if denominator else 0,
            "recall_at_5": sum(bool(row[rank_key] and row[rank_key] <= 5) for row in positive_rows) / denominator if denominator else 0,
            "mrr": statistics.fmean(reciprocal_ranks) if positive_rows else 0,
        }
    report = {
        "schema": "yugo-memory.private-benchmark.v2",
        "cases": count,
        "positive_cases": len(positive_rows),
        "negative_cases": len(negative_rows),
        "metrics": {
            "standalone_hybrid": metrics("hybrid", hybrid_reciprocal_ranks),
            "duplicate_free_rate": sum(row["duplicate_free"] for row in rows) / count if count else 0,
            "top1_exact_evidence_rate": sum(row["hybrid_rank"] == 1 for row in positive_rows) / len(positive_rows) if positive_rows else 0,
            "positive_answerability_rate": sum(row["hybrid_safe_to_answer"] for row in positive_rows) / len(positive_rows) if positive_rows else 0,
            "negative_false_positive_rate": sum(row["hybrid_safe_to_answer"] for row in negative_rows) / len(negative_rows) if negative_rows else 0,
            "release_gate_pass_rate": sum(row["passed"] for row in rows) / count if count else 0,
            "hybrid_latency_p50_ms": round(percentile(hybrid_latencies, 0.50), 2),
            "hybrid_latency_p95_ms": round(percentile(hybrid_latencies, 0.95), 2),
        },
        "results": rows,
        "privacy": "Queries and expectations are omitted from the report; transcript contents are never copied.",
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if rows and all(row["passed"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
