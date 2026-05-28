"""RQ2 orchestrator: compute-matched baselines vs planner-executor.

Runs one RQ2 config on one split with one seed. Resumable.

Configs:
  - self_consistency:  N SingleAgentReact samples, pick first valid (RQ2 SC)
  - budget_capped:     SingleAgentReact with raised max_steps + token cap (RQ2 BC)

(plan_then_execute exists in agents/ but was dropped from RQ2 because the
single-agent free-text plan caused GPT-4o to refuse the final Planner[Query]
action; see agents/plan_then_execute.py if revisiting.)

Usage:
    python -m eval_runner.run_rq2 --config self_consistency --split pilot --seed 0 --model gpt-4o --n-samples 2
    python -m eval_runner.run_rq2 --config budget_capped    --split pilot --seed 0 --model gpt-4o

Output layout:
    results/rq2/{config}/{split}/seed{seed}/results.jsonl
    results/rq2/{config}/{split}/seed{seed}/generated_plans/generated_plan_<idx+1>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tqdm import tqdm

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
sys.path.insert(0, str(_ROOT / "agents"))
sys.path.insert(0, str(_ROOT))

# Reuse RQ1 helpers — they're side-effect-free at import time (main is guarded).
from eval_runner.run_rq1 import _load_split, _load_done, _to_query_record  # noqa: E402


DEFAULT_TOKEN_BUDGET = 52000  # PE pilot mean ~52k tokens/query


def _result_paths(config: str, split: str, seed: int):
    base = _ROOT / "results" / "rq2" / config / split / f"seed{seed}"
    base.mkdir(parents=True, exist_ok=True)
    return base, base / "results.jsonl"


def _build_agent(config: str, model: str, seed: int, max_steps: int,
                 token_budget: int, n_samples: int):
    if config == "self_consistency":
        from self_consistency import SelfConsistencyAgent
        return SelfConsistencyAgent(
            model=model, seed=seed, max_steps=max_steps,
            n_samples=n_samples, token_budget=token_budget,
        )
    if config == "plan_then_execute":
        from plan_then_execute import PlanThenExecuteAgent
        return PlanThenExecuteAgent(
            model=model, seed=seed, max_steps=max_steps,
            token_budget=token_budget,
        )
    if config == "budget_capped":
        from budget_capped_react import BudgetCappedReactAgent
        return BudgetCappedReactAgent(
            model=model, seed=seed, max_steps=max_steps,
            token_budget=token_budget,
        )
    raise ValueError(config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    choices=["self_consistency", "budget_capped"])
    ap.add_argument("--split", required=True, choices=["pilot", "validation", "test"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--backend", default="openai", choices=["openai", "vllm"],
                    help="openai = hosted API; vllm = local server at OPENAI_BASE_URL")
    ap.add_argument("--model", default=None,
                    help="defaults to gpt-4o-mini (openai) or Qwen/Qwen2.5-7B-Instruct (vllm)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="overrides per-config default (SC/PtE=30, BC=50)")
    ap.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET,
                    help="matched compute budget (tokens/query); re-tune after pilot")
    ap.add_argument("--tolerance", type=float, default=0.10,
                    help="fractional tolerance for within_budget flag")
    ap.add_argument("--n-samples", type=int, default=2,
                    help="self_consistency only: number of samples to draw")
    ap.add_argument("--limit", type=int, default=None, help="cap queries (debug)")
    args = ap.parse_args()

    if args.model is None:
        args.model = "Qwen/Qwen2.5-7B-Instruct" if args.backend == "vllm" else "gpt-4o-mini"

    if args.backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(2)
    if args.backend == "vllm" and not os.environ.get("OPENAI_BASE_URL"):
        print("ERROR: OPENAI_BASE_URL not set", file=sys.stderr)
        sys.exit(2)

    if args.max_steps is None:
        args.max_steps = 50 if args.config == "budget_capped" else 30

    ds = _load_split(args.split)
    base_dir, jsonl_path = _result_paths(args.config, args.split, args.seed)
    plan_dir = base_dir / "generated_plans"
    plan_dir.mkdir(exist_ok=True)
    done = _load_done(jsonl_path)
    print(f"Resuming: {len(done)} queries already done in {jsonl_path}")

    indices = list(range(len(ds)))
    if args.limit is not None:
        indices = indices[: args.limit]

    budget_hi = args.token_budget * (1 + args.tolerance)

    pbar = tqdm(indices)
    for idx in pbar:
        if idx in done:
            continue
        row = ds[idx]
        query_record = _to_query_record(row)

        agent = _build_agent(
            args.config, args.model, args.seed, args.max_steps,
            args.token_budget, args.n_samples,
        )
        t0 = time.time()
        record: Dict[str, Any] = {
            "idx": idx,
            "config": args.config,
            "split": args.split,
            "seed": args.seed,
            "model": args.model,
            "level": query_record.get("level"),
            "days": query_record.get("days"),
            "token_budget": args.token_budget,
            "tolerance": args.tolerance,
            "max_steps": args.max_steps,
        }
        if args.config == "self_consistency":
            record["n_samples_requested"] = args.n_samples

        try:
            out = agent.run(query_record)
            wall = time.time() - t0
            total = (out["ledger"] or {}).get("total_tokens", 0)
            record.update({
                "answer": out["answer"],
                "ledger": out["ledger"],
                "wall_seconds": round(wall, 3),
                "within_budget": total <= budget_hi,
                "error": None,
            })
            if args.config == "self_consistency":
                record["samples"] = out.get("samples")
                record["n_samples_run"] = out.get("n_samples_run")
                record["winner_idx"] = out.get("winner_idx")
            elif args.config == "plan_then_execute":
                record["plan_text"] = out.get("plan_text")

            generated_plan_file = plan_dir / f"generated_plan_{idx + 1}.json"
            payload = [{
                f"{args.model}_two-stage_results_logs": out["scratchpad"],
                f"{args.model}_two-stage_results": out["answer"],
                f"{args.model}_two-stage_action_logs": out["action_log"],
            }]
            with generated_plan_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            wall = time.time() - t0
            record.update({
                "answer": None,
                "wall_seconds": round(wall, 3),
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(),
            })

        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        pbar.set_postfix(idx=idx, err=("y" if record.get("error") else "n"))


if __name__ == "__main__":
    main()
