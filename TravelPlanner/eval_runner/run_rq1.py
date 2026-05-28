"""RQ1 orchestrator.

Runs one config (single_react | planner_executor) on one split (pilot | validation)
with one seed. Resumable: skips queries already in results.jsonl.

Usage (from TravelPlanner root, with venv active and OPENAI_API_KEY set):

    python -m eval_runner.run_rq1 --config single_react --split pilot --seed 0
    python -m eval_runner.run_rq1 --config planner_executor --split validation --seed 0

Output layout:
    results/rq1/{config}/{seed}/results.jsonl       # per-query records
    results/rq1/{config}/{seed}/generated_plan_<idx>.json  # upstream-eval-compatible plan files
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

# Windows console default cp1252 chokes on non-Latin chars in tool output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datasets import load_dataset
from tqdm import tqdm

# Make agents/ importable as top-level (matches upstream `from prompts import ...`).
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
sys.path.insert(0, str(_ROOT / "agents"))
sys.path.insert(0, str(_ROOT))


PILOT_SIZE = 20


def _load_split(split: str):
    if split == "pilot" or split == "validation":
        ds = load_dataset("osunlp/TravelPlanner", "validation")["validation"]
    elif split == "test":
        ds = load_dataset("osunlp/TravelPlanner", "test")["test"]
    else:
        raise ValueError(split)
    if split == "pilot":
        ds = ds.select(range(PILOT_SIZE))
    return ds


def _result_paths(config: str, split: str, seed: int):
    base = _ROOT / "results" / "rq1" / config / split / f"seed{seed}"
    base.mkdir(parents=True, exist_ok=True)
    return base, base / "results.jsonl"


def _load_done(jsonl_path: Path) -> set:
    done = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(int(rec["idx"]))
            except Exception:
                pass
    return done


def _build_agent(config: str, model: str, seed: int):
    if config == "single_react":
        from single_agent_react import SingleAgentReact
        return SingleAgentReact(model=model, seed=seed)
    if config == "planner_executor":
        from planner_executor import PlannerExecutorAgent
        return PlannerExecutorAgent(model=model, seed=seed)
    raise ValueError(config)


def _to_query_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "org": row.get("org"),
        "dest": row.get("dest"),
        "days": row.get("days"),
        "date": row.get("date"),
        "people_number": row.get("people_number"),
        "local_constraint": row.get("local_constraint"),
        "budget": row.get("budget"),
        "query": row.get("query"),
        "level": row.get("level"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["single_react", "planner_executor"])
    ap.add_argument("--split", required=True, choices=["pilot", "validation", "test"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--backend", default="openai", choices=["openai", "vllm"],
                    help="openai = hosted API (needs OPENAI_API_KEY); "
                         "vllm = local server at OPENAI_BASE_URL (default model Qwen2.5-7B-Instruct)")
    ap.add_argument("--model", default=None,
                    help="defaults to gpt-4o-mini (openai) or Qwen/Qwen2.5-7B-Instruct (vllm)")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None, help="cap queries (debug)")
    args = ap.parse_args()

    if args.model is None:
        args.model = "Qwen/Qwen2.5-7B-Instruct" if args.backend == "vllm" else "gpt-4o-mini"

    if args.backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(2)
    if args.backend == "vllm" and not os.environ.get("OPENAI_BASE_URL"):
        print("ERROR: OPENAI_BASE_URL not set (point at the vLLM server, e.g. http://127.0.0.1:8000/v1)",
              file=sys.stderr)
        sys.exit(2)

    ds = _load_split(args.split)
    base_dir, jsonl_path = _result_paths(args.config, args.split, args.seed)
    plan_dir = base_dir / "generated_plans"
    plan_dir.mkdir(exist_ok=True)
    done = _load_done(jsonl_path)
    print(f"Resuming: {len(done)} queries already done in {jsonl_path}")

    indices = list(range(len(ds)))
    if args.limit is not None:
        indices = indices[: args.limit]

    pbar = tqdm(indices)
    for idx in pbar:
        if idx in done:
            continue
        row = ds[idx]
        query_record = _to_query_record(row)

        agent = _build_agent(args.config, args.model, args.seed)
        t0 = time.time()
        record: Dict[str, Any] = {
            "idx": idx,
            "config": args.config,
            "split": args.split,
            "seed": args.seed,
            "model": args.model,
            "level": query_record.get("level"),
            "days": query_record.get("days"),
        }
        try:
            out = agent.run(query_record)
            wall = time.time() - t0
            record.update({
                "answer": out["answer"],
                "ledger": out["ledger"],
                "wall_seconds": round(wall, 3),
                "error": None,
            })
            if args.config == "planner_executor":
                record["plan_valid"] = out.get("plan_valid")
                record["plan_errors"] = out.get("plan_errors")
                record["degraded"] = out.get("degraded")

            # Upstream eval expects per-query JSON file at generated_plan_<idx+1>.json.
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
