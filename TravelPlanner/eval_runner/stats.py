"""Statistics for RQ1 headline numbers.

Inputs: per-config-per-seed results.jsonl files plus the upstream eval output
(eval_score.json). This module focuses on the cross-seed aggregation and paired
bootstrap CI; the upstream evaluation/eval.py is responsible for the per-query
pass-rate calculation itself.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


def _load_per_query_scores(eval_score_path: Path) -> Dict[int, Dict[str, float]]:
    """Returns {idx: {final_pass_rate, hard_micro, ...}}.

    The upstream eval.py writes a per-query record with idx (1-based). We
    normalise to 0-based to match results.jsonl idx semantics.
    """
    raw = json.loads(eval_score_path.read_text(encoding="utf-8"))
    out = {}
    for entry in raw:
        idx = int(entry.get("idx", entry.get("query_id", -1))) - 1
        out[idx] = entry
    return out


def paired_bootstrap_ci(
    scores_a: List[float],
    scores_b: List[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    rng_seed: int = 0,
) -> Tuple[float, float, float]:
    """Returns (mean_diff, ci_low, ci_high) for B - A on per-query paired scores."""
    assert len(scores_a) == len(scores_b)
    rng = random.Random(rng_seed)
    n = len(scores_a)
    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    mean_diff = statistics.mean(diffs)
    boots = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(statistics.mean(sample))
    boots.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = boots[int(alpha * n_resamples)]
    hi = boots[int((1.0 - alpha) * n_resamples)]
    return mean_diff, lo, hi


def extract_metric(rec: Dict, metric: str) -> float:
    """Map a metric name to a per-query float in [0, 1].

    Handles the eval_score.json schema where macro checks are booleans
    (possibly None) and micro checks are stored as passes/total pairs.
    None (e.g. hard_pass on an undelivered plan) counts as 0.0.
    """
    # derived micro ratios
    if metric in ("commonsense_micro", "hard_micro"):
        prefix = metric[: -len("_micro")]
        passes = rec.get(f"{prefix}_micro_passes", 0) or 0
        total = rec.get(f"{prefix}_micro_total", 0) or 0
        return passes / total if total > 0 else 0.0
    val = rec.get(metric, 0)
    if val is None:
        return 0.0
    return float(val)


def aggregate_seeds(per_seed_means: Dict[int, float]) -> Dict[str, float]:
    xs = list(per_seed_means.values())
    return {
        "seed_mean": statistics.mean(xs),
        "seed_stdev": statistics.pstdev(xs),
        "n_seeds": len(xs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-a", required=True, help="config name (baseline), e.g. single_react")
    ap.add_argument("--config-b", required=True, help="config name (treatment), e.g. planner_executor")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--metric", default="final_pass_rate")
    ap.add_argument("--results-root", default="results/rq1")
    ap.add_argument("--results-root-a", default=None,
                    help="override results root for config-a (defaults to --results-root); "
                         "use to compare two models, e.g. 14B vs 72B")
    ap.add_argument("--results-root-b", default=None,
                    help="override results root for config-b (defaults to --results-root)")
    args = ap.parse_args()

    root_a = Path(args.results_root_a or args.results_root)
    root_b = Path(args.results_root_b or args.results_root)

    def per_query_scores(config: str, seed: int, root: Path) -> Dict[int, float]:
        p = root / config / args.split / f"seed{seed}" / "eval_score.json"
        if not p.exists():
            raise FileNotFoundError(f"missing {p} -- run upstream evaluation/eval.py first")
        scores = _load_per_query_scores(p)
        return {idx: extract_metric(rec, args.metric) for idx, rec in scores.items()}

    a_per_seed: Dict[int, float] = {}
    b_per_seed: Dict[int, float] = {}
    diffs_per_seed: List[Tuple[int, float, float, float]] = []

    for seed in args.seeds:
        a_scores = per_query_scores(args.config_a, seed, root_a)
        b_scores = per_query_scores(args.config_b, seed, root_b)
        common = sorted(set(a_scores) & set(b_scores))
        a_vec = [a_scores[i] for i in common]
        b_vec = [b_scores[i] for i in common]
        a_per_seed[seed] = statistics.mean(a_vec)
        b_per_seed[seed] = statistics.mean(b_vec)
        mean_diff, lo, hi = paired_bootstrap_ci(a_vec, b_vec, rng_seed=seed)
        diffs_per_seed.append((seed, mean_diff, lo, hi))

    out = {
        "metric": args.metric,
        "config_a": {"name": args.config_a, "results_root": str(root_a),
                     **aggregate_seeds(a_per_seed), "per_seed": a_per_seed},
        "config_b": {"name": args.config_b, "results_root": str(root_b),
                     **aggregate_seeds(b_per_seed), "per_seed": b_per_seed},
        "paired_diff_per_seed": [
            {"seed": s, "mean_diff_b_minus_a": d, "ci_low": lo, "ci_high": hi}
            for (s, d, lo, hi) in diffs_per_seed
        ],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
