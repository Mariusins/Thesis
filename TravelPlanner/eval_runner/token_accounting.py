"""Aggregate token usage from results.jsonl files for compute-matching analysis.

Used by RQ1 to report mean tokens per query (input for RQ2 token-matching) and
by RQ2 to verify that each baseline's budget is within tolerance of the PE
anchor (default ~52k tokens/query, ±10%).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional


def _read_jsonl(p: Path) -> List[dict]:
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summarise(records: List[dict]) -> Dict[str, float]:
    totals = []
    prompts = []
    completions = []
    by_role: Dict[str, List[int]] = {}
    n_calls = []
    for r in records:
        ledger = r.get("ledger") or {}
        totals.append(ledger.get("total_tokens", 0))
        prompts.append(ledger.get("prompt_tokens", 0))
        completions.append(ledger.get("completion_tokens", 0))
        n_calls.append(ledger.get("n_calls", 0))
        for role, slot in (ledger.get("by_role") or {}).items():
            by_role.setdefault(role, []).append(slot.get("total_tokens", 0))

    def stats(xs):
        if not xs:
            return {"mean": 0, "stdev": 0, "n": 0}
        return {
            "mean": statistics.mean(xs),
            "stdev": statistics.pstdev(xs),
            "n": len(xs),
        }

    return {
        "total": stats(totals),
        "prompt": stats(prompts),
        "completion": stats(completions),
        "n_calls": stats(n_calls),
        "by_role_mean": {role: statistics.mean(xs) for role, xs in by_role.items()},
    }


def budget_match(records: List[dict], target: int, tolerance: float = 0.10) -> Dict[str, float]:
    """Return per-query budget conformance vs `target` within `±tolerance`."""
    totals = [(r.get("ledger") or {}).get("total_tokens", 0) for r in records]
    if not totals:
        return {"target": target, "tolerance": tolerance, "mean": 0, "ratio": 0,
                "pct_within": 0, "n_over": 0, "n_under": 0, "n": 0}
    mean = statistics.mean(totals)
    lo = target * (1 - tolerance)
    hi = target * (1 + tolerance)
    within = [1 for t in totals if lo <= t <= hi]
    over = [1 for t in totals if t > hi]
    under = [1 for t in totals if t < lo]
    return {
        "target": target,
        "tolerance": tolerance,
        "lo": lo,
        "hi": hi,
        "mean": mean,
        "ratio": mean / target if target else 0,
        "pct_within": len(within) / len(totals),
        "n_over": len(over),
        "n_under": len(under),
        "n": len(totals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="path to results.jsonl")
    ap.add_argument("--target", type=int, default=None,
                    help="if set, also report budget_match vs this target (tokens/query)")
    ap.add_argument("--tolerance", type=float, default=0.10,
                    help="fractional tolerance for budget_match (default 0.10 = ±10%%)")
    args = ap.parse_args()
    records = _read_jsonl(Path(args.results))
    out = {"summary": summarise(records)}
    if args.target is not None:
        out["budget_match"] = budget_match(records, args.target, args.tolerance)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
