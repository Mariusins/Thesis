"""Pilot-scope evaluation.

For each (config, seed): take the 20 NL plans we generated, convert to JSON via
GPT-4o-mini (same prompt as upstream postprocess/parsing.py), then run upstream
commonsense + hard constraint checks. Writes per-query scores to eval_score.json
plus a roll-up pilot_summary.json.

Run from TravelPlanner root with venv active and OPENAI_API_KEY set.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from tqdm import tqdm

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "evaluation"))
sys.path.insert(0, str(_ROOT / "agents"))

from commonsense_constraint import evaluation as commonsense_eval
from hard_constraint import evaluation as hard_eval


# Named checks exposed by the upstream evaluators, in evaluation order.
# These drive the per-check failure-mode breakdown (failure_modes.py).
COMMONSENSE_KEYS = [
    "is_reasonable_visiting_city",
    "is_valid_restaurants",
    "is_valid_attractions",
    "is_valid_accommodation",
    "is_valid_transportation",
    "is_valid_information_in_current_city",
    "is_valid_information_in_sandbox",
    "is_not_absent",
]
HARD_KEYS = [
    "valid_cuisine",
    "valid_room_rule",
    "valid_transportation",
    "valid_room_type",
    "valid_cost",
]


PARSING_PREFIX = """Please assist me in extracting valid information from a given natural language text and reconstructing it in JSON format, as demonstrated in the following example. If transportation details indicate a journey from one city to another (e.g., from A to B), the 'current_city' should be updated to the destination city (in this case, B). Use a ';' to separate different attractions, with each attraction formatted as 'Name, City'. If there's information about transportation, ensure that the 'current_city' aligns with the destination mentioned in the transportation details (i.e., the current city should follow the format 'from A to B'). Also, ensure that all flight numbers and costs are followed by a colon (i.e., 'Flight Number:' and 'Cost:'), consistent with the provided example. Each item should include ['day', 'current_city', 'transportation', 'breakfast', 'attraction', 'lunch', 'dinner', 'accommodation']. Replace non-specific information like 'eat at home/on the road' with '-'. Additionally, delete any '$' symbols. Output a fenced ```json``` block.
-----EXAMPLE-----
 [{
        "days": 1,
        "current_city": "from Dallas to Peoria",
        "transportation": "Flight Number: 4044830, from Dallas to Peoria, Departure Time: 13:10, Arrival Time: 15:01",
        "breakfast": "-",
        "attraction": "Peoria Historical Society, Peoria;Peoria Holocaust Memorial, Peoria;",
        "lunch": "-",
        "dinner": "Tandoor Ka Zaika, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    },
    {
        "days": 2,
        "current_city": "Peoria",
        "transportation": "-",
        "breakfast": "Tandoor Ka Zaika, Peoria",
        "attraction": "Peoria Riverfront Park, Peoria;The Peoria PlayHouse, Peoria;Glen Oak Park, Peoria;",
        "lunch": "Cafe Hashtag LoL, Peoria",
        "dinner": "The Curzon Room - Maidens Hotel, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    },
    {
        "days": 3,
        "current_city": "from Peoria to Dallas",
        "transportation": "Flight Number: 4045904, from Peoria to Dallas, Departure Time: 07:09, Arrival Time: 09:20",
        "breakfast": "-",
        "attraction": "-",
        "lunch": "-",
        "dinner": "-",
        "accommodation": "-"
    }]
-----EXAMPLE END-----
"""


def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)


def parse_nl_plan_to_json(client: OpenAI, nl_plan: str, parser_model: str = "gpt-4o-mini") -> Optional[List[Dict[str, Any]]]:
    if not nl_plan or nl_plan == "Max Token Length Exceeded.":
        return None
    prompt = PARSING_PREFIX + "Text:\n" + nl_plan + "\nJSON:\n"
    backoff = 2.0
    for _ in range(5):
        try:
            resp = client.chat.completions.create(
                model=parser_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=2048,
            )
            text = resp.choices[0].message.content or ""
            m = re.search(r"```json\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
            if not m:
                m = re.search(r"(\[.*\])", text, flags=re.DOTALL)
            if not m:
                return None
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                try:
                    return eval(m.group(1))
                except Exception:
                    return None
        except RateLimitError:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
        except (APIConnectionError, APIError):
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
    return None


def _commonsense_pass(info_box: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not info_box:
        return None
    for key, val in info_box.items():
        if val[0] is not None and not val[0]:
            return False
    return True


def _hard_pass(info_box: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not info_box:
        return None
    for key, val in info_box.items():
        if val[0] is not None and val[0] is False:
            return False
    return True


def _detail(info_box: Optional[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    """Per-check pass/fail (True/False/None) plus messages for failed checks.

    `None` means the upstream check returned no verdict (not applicable / no info).
    """
    out: Dict[str, Any] = {"checks": {}, "failed": [], "messages": {}}
    if not info_box:
        out["checks"] = {k: None for k in keys}
        return out
    for k in keys:
        verdict = info_box.get(k, (None, None))
        passed = verdict[0]
        out["checks"][k] = passed
        if passed is False:
            out["failed"].append(k)
            if verdict[1]:
                out["messages"][k] = verdict[1]
    return out


def evaluate_one(query_data: Dict[str, Any], plan_json: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    if plan_json is None:
        return {
            "delivered": False,
            "commonsense_pass": None,
            "hard_pass": None,
            "commonsense_micro_passes": 0,
            "commonsense_micro_total": 8,
            "hard_micro_passes": 0,
            "hard_micro_total": 0,
            "final_pass": False,
            "commonsense_detail": _detail(None, COMMONSENSE_KEYS),
            "hard_detail": _detail(None, HARD_KEYS),
        }
    try:
        cs = commonsense_eval(query_data, plan_json)
    except (IndexError, KeyError, TypeError):
        # Upstream evaluator crashes on malformed plans (e.g. empty city list).
        # Treat as undelivered.
        return {
            "delivered": False,
            "commonsense_pass": None,
            "hard_pass": None,
            "commonsense_micro_passes": 0,
            "commonsense_micro_total": 8,
            "hard_micro_passes": 0,
            "hard_micro_total": 0,
            "final_pass": False,
            "eval_error": "commonsense_eval_crash",
            "commonsense_detail": _detail(None, COMMONSENSE_KEYS),
            "hard_detail": _detail(None, HARD_KEYS),
        }
    cs_pass = _commonsense_pass(cs)
    cs_micro_passes = sum(1 for k, v in cs.items() if v[0] is True)

    hard = None
    hard_micro_passes = 0
    hard_micro_total = 0
    if cs and cs.get("is_not_absent", [None])[0] and cs.get("is_valid_information_in_sandbox", [None])[0]:
        hard = hard_eval(query_data, plan_json)
        for k, v in hard.items():
            if v[0] is not None:
                hard_micro_total += 1
                if v[0]:
                    hard_micro_passes += 1
    hard_pass = _hard_pass(hard)

    final = bool(cs_pass) and bool(hard_pass)
    return {
        "delivered": True,
        "commonsense_pass": cs_pass,
        "hard_pass": hard_pass,
        "commonsense_micro_passes": cs_micro_passes,
        "commonsense_micro_total": 8,
        "hard_micro_passes": hard_micro_passes,
        "hard_micro_total": hard_micro_total,
        "final_pass": final,
        "commonsense_detail": _detail(cs, COMMONSENSE_KEYS),
        "hard_detail": _detail(hard, HARD_KEYS),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["single_react", "planner_executor"])
    ap.add_argument("--split", default="pilot", choices=["pilot", "validation"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="gpt-4o-mini", help="agent model used for results key lookup")
    ap.add_argument("--parser-model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--results-root", default="results/rq1",
                    help="root dir for outputs, e.g. results/rq1_steps50")
    args = ap.parse_args()

    base = _ROOT / args.results_root / args.config / args.split / f"seed{args.seed}"
    plan_dir = base / "generated_plans"
    if not plan_dir.exists():
        raise FileNotFoundError(plan_dir)
    parsed_dir = base / "parsed_plans"
    parsed_dir.mkdir(exist_ok=True)

    # Runner-level outcome per query (idx -> error string or None), so a failed
    # delivery can be attributed to a Stage-0 runner error (e.g. context overflow)
    # rather than lumped into generic non-delivery.
    runner_error_by_idx: Dict[int, Optional[str]] = {}
    runner_jsonl = base / "results.jsonl"
    if runner_jsonl.exists():
        for line in runner_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                runner_error_by_idx[int(rec["idx"])] = rec.get("error")
            except Exception:
                pass

    ds = load_dataset("osunlp/TravelPlanner", "validation")["validation"]
    pilot_size = 20
    n = pilot_size if args.split == "pilot" else len(ds)
    if args.limit:
        n = min(n, args.limit)

    client = _client()
    results_key = f"{args.model}_two-stage_results"

    per_query: List[Dict[str, Any]] = []
    for i in tqdm(range(n)):
        plan_file = plan_dir / f"generated_plan_{i+1}.json"
        if not plan_file.exists():
            # No plan emitted — usually a Stage-0 runner error (e.g. context
            # overflow) or a query not yet run in a partial run.
            runner_error = runner_error_by_idx.get(i)
            per_query.append({
                "idx": i,
                "delivered": False,
                "commonsense_pass": None,
                "hard_pass": None,
                "final_pass": False,
                "failure_reason": "runner_error" if runner_error else "not_run",
                "runner_error": runner_error,
                "commonsense_detail": _detail(None, COMMONSENSE_KEYS),
                "hard_detail": _detail(None, HARD_KEYS),
            })
            continue
        payload = json.loads(plan_file.read_text(encoding="utf-8"))
        nl_plan = payload[-1].get(results_key, "")

        # Skip the parser API call for non-deliveries (empty / sentinel).
        if not nl_plan or nl_plan == "Max Token Length Exceeded.":
            plan_json = None
        else:
            plan_json = parse_nl_plan_to_json(client, nl_plan, parser_model=args.parser_model)
        # Persist the parsed plan so future failure-mode analysis needs no re-parsing.
        (parsed_dir / f"parsed_plan_{i+1}.json").write_text(
            json.dumps(plan_json, indent=2), encoding="utf-8")

        query_data = ds[i]
        if isinstance(query_data["local_constraint"], str):
            query_data = dict(query_data)
            query_data["local_constraint"] = eval(query_data["local_constraint"])
        else:
            query_data = dict(query_data)

        score = evaluate_one(query_data, plan_json)
        score["idx"] = i
        score["level"] = query_data["level"]
        score["days"] = query_data["days"]

        # Single mutually-exclusive failure_reason for the Stage 0/1 taxonomy.
        runner_error = runner_error_by_idx.get(i)
        if score["final_pass"]:
            score["failure_reason"] = "pass"
        elif score["delivered"]:
            score["failure_reason"] = "constraint_fail"
        elif runner_error:
            score["failure_reason"] = "runner_error"
            score["runner_error"] = runner_error
        elif nl_plan == "Max Token Length Exceeded.":
            score["failure_reason"] = "max_token_exceeded"
        elif not nl_plan:
            score["failure_reason"] = "empty_answer"
        elif score.get("eval_error") == "commonsense_eval_crash":
            score["failure_reason"] = "eval_crash"
        elif plan_json is None:
            score["failure_reason"] = "parse_failed"
        else:
            score["failure_reason"] = "undelivered_other"
        per_query.append(score)

    delivered = sum(1 for r in per_query if r.get("delivered"))
    cs_macro = sum(1 for r in per_query if r.get("commonsense_pass") is True)
    hard_macro = sum(1 for r in per_query if r.get("hard_pass") is True)
    final_macro = sum(1 for r in per_query if r.get("final_pass") is True)
    cs_micro_pass = sum(r.get("commonsense_micro_passes", 0) for r in per_query)
    cs_micro_total = sum(r.get("commonsense_micro_total", 0) for r in per_query)
    hard_micro_pass = sum(r.get("hard_micro_passes", 0) for r in per_query)
    hard_micro_total = sum(r.get("hard_micro_total", 0) for r in per_query)

    # Stage 0/1 failure-reason histogram + per-check failure counts.
    failure_reasons: Dict[str, int] = {}
    for r in per_query:
        fr = r.get("failure_reason", "unknown")
        failure_reasons[fr] = failure_reasons.get(fr, 0) + 1
    cs_check_fail = {k: 0 for k in COMMONSENSE_KEYS}
    hard_check_fail = {k: 0 for k in HARD_KEYS}
    for r in per_query:
        for k in r.get("commonsense_detail", {}).get("failed", []):
            cs_check_fail[k] = cs_check_fail.get(k, 0) + 1
        for k in r.get("hard_detail", {}).get("failed", []):
            hard_check_fail[k] = hard_check_fail.get(k, 0) + 1

    summary = {
        "config": args.config,
        "split": args.split,
        "seed": args.seed,
        "n_queries": len(per_query),
        "delivery_rate": delivered / len(per_query) if per_query else 0,
        "commonsense_macro_pass_rate": cs_macro / len(per_query) if per_query else 0,
        "hard_macro_pass_rate": hard_macro / len(per_query) if per_query else 0,
        "final_pass_rate": final_macro / len(per_query) if per_query else 0,
        "commonsense_micro_pass_rate": (cs_micro_pass / cs_micro_total) if cs_micro_total else 0,
        "hard_micro_pass_rate": (hard_micro_pass / hard_micro_total) if hard_micro_total else 0,
        "failure_reasons": failure_reasons,
        "commonsense_check_failures": cs_check_fail,
        "hard_check_failures": hard_check_fail,
    }

    (base / "eval_score.json").write_text(json.dumps(per_query, indent=2), encoding="utf-8")
    (base / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
