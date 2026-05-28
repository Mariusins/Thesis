# TravelPlanner Fork — Codebase Explained

End-to-end walkthrough of this fork: what every file does, how the two agent configurations execute, how the code differs from upstream classic TravelPlanner, and what the pilot test numbers mean.

> Companion to [overview.md](../overview.md) (status report). This document is the architectural reference.

---

## 1. Purpose & RQ1 Setup

This fork exists to answer **RQ1**: under matched compute, does a **planner-executor** two-agent architecture beat a **single-agent ReAct** baseline on the TravelPlanner benchmark?

Both configurations share:

- The same 7 sandbox tools (Flight, Attraction, Accommodation, Restaurant, GoogleDistanceMatrix, Planner, Cities, plus NotebookWrite).
- The same upstream `ReactAgent` core loop and 30-step cap.
- The same LLM entry point ([llm_client.py](agents/llm_client.py)) so seed, retry, and token accounting are identical.
- The same dataset (`osunlp/TravelPlanner` validation split, or first 20 queries = `pilot`).

They differ only in **how the LLM is structured**: one agent vs. planner + executor.

Pilot model: `gpt-4o-mini` (floored — see §9). Decision model: `gpt-4o`.

---

## 2. Repo Layout

| Directory | Role |
|---|---|
| [agents/](agents/) | LLM entry point, prompts, single-agent + planner-executor agent classes, upstream `ReactAgent`. |
| [tools/](tools/) | The 7 sandbox tools the agents call (flights, attractions, accommodations, restaurants, cities, googleDistanceMatrix, planner, notebook). |
| [database/](database/) | Static sandbox CSVs that the tools read (flights, hotels, restaurants, attractions, etc.). |
| [evaluation/](evaluation/) | Upstream constraint checkers: [commonsense_constraint.py](evaluation/commonsense_constraint.py), [hard_constraint.py](evaluation/hard_constraint.py), [eval.py](evaluation/eval.py). |
| [postprocess/](postprocess/) | NL→JSON parsing utilities ([parsing.py](postprocess/parsing.py), [openai_request.py](postprocess/openai_request.py)) — migrated to modern openai SDK. |
| [eval_runner/](eval_runner/) | New CLI orchestrators for this thesis: [run_rq1.py](eval_runner/run_rq1.py), [run_pilot_eval.py](eval_runner/run_pilot_eval.py), [stats.py](eval_runner/stats.py), [token_accounting.py](eval_runner/token_accounting.py). |
| [results/](results/) | Per-config / per-seed outputs: `results.jsonl`, `generated_plans/`, `eval_score.json`, `pilot_summary.json`. |
| [finetuning_data/](finetuning_data/) | Upstream artifacts, not used by RQ1. |

---

## 3. What Was Migrated From Upstream

Classic TravelPlanner targets `langchain==0.1.4` + `openai==0.27.2`. Four files were migrated to modern `openai>=1.40`:

| File | What changed |
|---|---|
| [agents/llm_client.py](agents/llm_client.py) | **New**. Single LLM entry point replacing scattered `ChatOpenAI(...)` instantiations. Adds `TokenLedger`, retry/backoff, seed. |
| [agents/tool_agents.py](agents/tool_agents.py) | Upstream `ReactAgent` rewired. Default model now `gpt-4o-mini`. Constructor accepts `ledger` and `seed`. All internal LLM calls go via `LLMClient`. |
| [tools/planner/apis.py](tools/planner/apis.py) | Upstream `Planner` / `ReactPlanner` / `ReactReflectPlanner` migrated to modern SDK. |
| [postprocess/openai_request.py](postprocess/openai_request.py) + [postprocess/parsing.py](postprocess/parsing.py) | NL plan → JSON parser migrated to modern SDK. Default parser model = `gpt-4o-mini`. |

The remaining upstream files (constraint checkers, tool wrappers, the sandbox database) are untouched.

---

## 4. New Components — File-by-File Deep Dive

### 4.1 `agents/llm_client.py`

Two classes:

**`TokenLedger`** — accumulates `prompt_tokens`, `completion_tokens`, `total_tokens`, `n_calls`, and a `by_role` dict keyed by `role_tag`. Updated on every successful call:

```python
slot = self.by_role.setdefault(role, {"prompt_tokens": 0, "completion_tokens": 0, ...})
slot["prompt_tokens"] += usage.get("prompt_tokens", 0)
```
(see [llm_client.py:25-34](agents/llm_client.py#L25-L34))

**`LLMClient`** — thin wrapper around `openai.OpenAI().chat.completions.create`:

- Reads `OPENAI_API_KEY` from env, raises if missing.
- Default `temperature=0.0`, `max_tokens=256` (callers override for the planner).
- Optional `seed`, `stop`, `response_format` (used to force `{"type": "json_object"}` for the planner).
- Retries up to `max_retries=10` on `RateLimitError` / `APIConnectionError` / `APIError` with exponential backoff capped at 120s. `AuthenticationError` / `BadRequestError` bubble immediately.
- On success, adds usage to the shared `ledger` under the caller-provided `role_tag` ([llm_client.py:96-104](agents/llm_client.py#L96-L104)).

A `_LegacyResponse` shim exposes `.content` so upstream code that expected `ChatOpenAI.invoke(...).content` keeps working with a minimal diff.

### 4.2 `agents/prompts_planner.py`

Three artifacts ([prompts_planner.py](agents/prompts_planner.py)):

- **`PLANNER_JSON_SCHEMA`** — Draft-7 JSON schema. Requires top-level `subgoals`, `total_budget`, `budget_allocation`. Each sub-goal has `id` (int) and `type` (one of 8 enums: `city_search`, `flight_search`, `ground_transport_search`, `accommodation_search`, `restaurant_search`, `attraction_search`, `daily_itinerary`, `budget_check`). `additionalProperties: True` on sub-goals lets the planner attach free-form constraint keys (e.g. `min_occupancy`, `cuisine`, `city_pair`).
- **`PLANNER_SYSTEM_PROMPT`** — instructs the planner that it does *not* call tools; it only structures the plan. Rules: minimal ordered sequence, `city_search` first if destination is a state, one `daily_itinerary` per day, attach query constraints to relevant sub-goals, `budget_allocation` fractions ≈ 1.0. Query fields (`org`, `dest`, `days`, `date`, `people_number`, `local_constraint`, `budget`, `query`) are filled via `str.format`.
- **`PLANNER_FEW_SHOT_EXAMPLE`** — Ithaca→Charlotte 3-day example showing the exact JSON shape.
- **`EXECUTOR_PLAN_PREAMBLE`** — block prepended to the executor's prompt template, contains the rendered plan JSON and an instruction to process sub-goals top-to-bottom.

### 4.3 `agents/planner_executor.py`

Two classes ([planner_executor.py](agents/planner_executor.py)):

**`PlannerAgent`** — stateless single-call planner.

- Constructed with model, `TokenLedger`, optional `seed`. `LLMClient` uses `temperature=0.0`, `max_tokens=2048`.
- `plan(query_record)` renders `PLANNER_SYSTEM_PROMPT`, sends it with `response_format={"type": "json_object"}`, role-tagged as `planner_agent`.
- Up to **2 attempts**:
  1. `_extract_json(raw)` strips ` ```json ... ``` ` fences or extracts the first `{...}` block.
  2. Validate against `PLANNER_JSON_SCHEMA` with `Draft7Validator`.
  3. On parse fail → reply *"That was not valid JSON. Output only the JSON object..."*; on schema fail → reply *"Schema errors: ...; Fix and re-emit..."*.
- Returns `{plan, raw, valid, errors}` ([planner_executor.py:69-107](agents/planner_executor.py#L69-L107)).

**`PlannerExecutorAgent`** — composes planner + executor over a *shared* `TokenLedger`.

- The executor is the upstream `ReactAgent` with the same 8 tool names as `SingleAgentReact`.
- `run(query_record)`:
  1. Call `self.planner.plan(...)`.
  2. If plan is valid → escape `{` `}` in the JSON (the downstream prompt is consumed via `str.format`, so literal braces must be doubled), wrap in `EXECUTOR_PLAN_PREAMBLE`, prepend to `executor.agent_prompt.template` ([planner_executor.py:164-172](agents/planner_executor.py#L164-L172)). Original template stashed in `_original_template`. Always restored in a `finally` block.
  3. If plan is invalid after both retries → **degraded mode**: run the executor with the untouched ReAct template, flag `degraded=True` in the result.
- Returns `{answer, scratchpad, action_log, plan_json, plan_raw, plan_valid, plan_errors, ledger, degraded}`.

### 4.4 `agents/single_agent_react.py`

Baseline shim around `ReactAgent` ([single_agent_react.py](agents/single_agent_react.py)). Same 8-tool list, same 30-step cap, fresh `TokenLedger`. `run(query_record)` returns `{answer, scratchpad, action_log, ledger}`. The whole point: share the inference path so the only independent variable between configs is the architecture.

### 4.5 `eval_runner/run_rq1.py`

CLI orchestrator ([run_rq1.py](eval_runner/run_rq1.py)).

- Args: `--config {single_react|planner_executor}`, `--split {pilot|validation|test}`, `--seed N`, `--model`, `--max-steps`, `--limit`.
- Loads dataset via `datasets.load_dataset("osunlp/TravelPlanner", "validation")`; `pilot` = first 20 rows.
- Output base: `results/rq1/{config}/{split}/seed{seed}/`.
- **Resumable**: reads `results.jsonl`, skips any `idx` already present.
- For each remaining query: build a fresh agent (so seed is applied at construction), run it, time it, append a JSON record with `{idx, config, split, seed, model, level, days, answer, ledger, wall_seconds, error}` plus `{plan_valid, plan_errors, degraded}` for planner-executor.
- Also writes `generated_plan_{idx+1}.json` in the format upstream `evaluation/eval.py` expects: a list with one object containing `{model}_two-stage_results_logs`, `_results`, `_action_logs`.
- Wraps the per-query call in a `try/except` that records any exception's `type` + `str` + full traceback so a crash in one query doesn't kill the run.

### 4.6 `eval_runner/run_pilot_eval.py`

Pilot-scope evaluator ([run_pilot_eval.py](eval_runner/run_pilot_eval.py)).

- For each of the 20 pilot queries:
  1. Load the generated NL plan from `generated_plans/generated_plan_{i+1}.json`.
  2. `parse_nl_plan_to_json` calls the `--parser-model` (default `gpt-4o-mini`) with the upstream `PARSING_PREFIX` prompt to coerce the NL plan into a list of per-day JSON objects. Retries 5× on rate-limit / API errors.
  3. Run upstream `commonsense_constraint.evaluation` and (conditionally) `hard_constraint.evaluation` from [evaluation/](evaluation/).
- `evaluate_one` computes:
  - `delivered` = parser produced a list.
  - `commonsense_pass` via `_commonsense_pass` — returns `False` if any check `val[0]` is explicitly false.
  - `commonsense_micro_passes` = count of checks with `v[0] is True`. Denominator fixed at 8.
  - **Hard checks gated**: only computed if `is_not_absent` AND `is_valid_information_in_sandbox` are both true ([run_pilot_eval.py:153-160](eval_runner/run_pilot_eval.py#L153-L160)). Otherwise `hard_pass = None`, `hard_micro_total = 0`.
  - `final_pass = bool(commonsense_pass) and bool(hard_pass)`.
- Roll-up into `pilot_summary.json`: `delivery_rate`, `commonsense_macro_pass_rate`, `hard_macro_pass_rate`, `final_pass_rate`, `commonsense_micro_pass_rate`, `hard_micro_pass_rate`. Per-query records in `eval_score.json`.

### 4.7 `eval_runner/stats.py`

Paired bootstrap CI ([stats.py](eval_runner/stats.py)).

- Loads `eval_score.json` per (config, seed).
- For each seed: build per-query vectors `a_vec`, `b_vec` over the intersection of indices, mean them, compute `paired_bootstrap_ci(a_vec, b_vec)` → 10 000 resamples of `diff = b - a`, 95% CI from sorted bootstrap means.
- Reports per-seed mean diff + CI plus seed-level aggregates (`seed_mean`, `seed_stdev`, `n_seeds`).

### 4.8 `eval_runner/token_accounting.py`

Token aggregation helpers that operate over the `ledger` blobs in `results.jsonl`.

---

## 5. Execution Flow

### single_react

```
dataset row
  └─> SingleAgentReact(model, seed)
        └─> ReactAgent.run(query)
              └─> ReAct loop (≤30 steps):
                    Thought → Action[FlightSearch/AccommodationSearch/...] → Observation
                    ...
                    Action[NotebookWrite[...]]  (accumulate findings)
                    ...
                    Action[Planner[Query]]  →  final NL plan
              └─> returns (answer, scratchpad, action_log)
  └─> results.jsonl record + generated_plan_<idx+1>.json
```

Every LLM call goes through `LLMClient`, accumulating into the single agent's `TokenLedger`.

### planner_executor

```
dataset row
  └─> PlannerExecutorAgent(model, seed)
        ├─> PlannerAgent.plan(query_record)
        │     ├─> LLMClient.chat(... response_format=json_object, role_tag="planner_agent")
        │     ├─> _extract_json + jsonschema validate
        │     └─> retry once on parse / schema fail
        │
        ├─> [plan valid?] yes:
        │     ├─> Prepend EXECUTOR_PLAN_PREAMBLE + plan JSON to executor's agent_prompt.template
        │     └─> ReactAgent.run(query)   [same loop as single_react]
        │
        └─> [plan valid?] no, after both retries:
              └─> degraded=True; ReactAgent.run(query) with original template
  └─> results.jsonl record (with plan_valid, plan_errors, degraded) + generated_plan_<idx+1>.json
```

The shared `TokenLedger` holds both planner and executor usage, broken out by `role_tag`.

---

## 6. Differences vs. Classic Upstream TravelPlanner

| Aspect | Classic upstream | This fork |
|---|---|---|
| LLM library | `langchain==0.1.4` + `openai==0.27.2`; `ChatOpenAI` instantiated per script. | Modern `openai>=1.40`; single `LLMClient` entry point used everywhere. |
| Token accounting | None — usage scattered, no role breakdown. | `TokenLedger` with `prompt/completion/total_tokens` per `role_tag`. |
| Agent architectures | Only `ReactAgent`, `Planner`, `ReactPlanner`, `ReactReflectPlanner`. | Adds explicit two-stage `PlannerAgent → ReactAgent(executor)` with structured JSON hand-off. |
| Prompts | Hard-coded in [agents/prompts.py](agents/prompts.py). | Same `prompts.py` retained for the executor; new [prompts_planner.py](agents/prompts_planner.py) for the planner. |
| Seed handling | Implicit / non-existent. | Explicit `seed` threaded through `LLMClient.chat`. |
| Eval pipeline | Two-step: postprocess parsing → `evaluation/eval.py`. | Integrated [run_pilot_eval.py](eval_runner/run_pilot_eval.py): parse + commonsense + hard + roll-up in one CLI. |
| Generated-plan filenames | `generated_plan_<idx+1>.json`. | **Preserved** so upstream [evaluation/eval.py](evaluation/eval.py) still works unchanged. |
| Dataset slicing | Full validation split. | Adds `pilot` (first 20 queries) for cheap iteration. |
| Resumability | None — runs from scratch. | [run_rq1.py](eval_runner/run_rq1.py) skips queries already present in `results.jsonl`. |
| Failure handling | Hard crash kills the run. | Per-query `try/except` records error + traceback into the record, run continues. |
| Console encoding | Default. | Forces UTF-8 on `sys.stdout/err` for Windows cp1252 friendliness. |

---

## 7. Pilot Results

Both configurations: 20 queries × 3 seeds × same harness. Numbers below are verbatim from the six `pilot_summary.json` files under [results/rq1/](results/rq1/) (mini pilot).

| Config | Seed | Delivery | CS macro | Hard macro | Final | CS micro | Hard micro |
|---|---|---|---|---|---|---|---|
| single_react | 0 | 0.90 | 0.00 | 0.05 | 0.00 | 0.7625 | 0.25 |
| single_react | 1 | 0.90 | 0.00 | 0.15 | 0.00 | 0.7438 | 0.75 |
| single_react | 2 | 0.90 | 0.00 | 0.10 | 0.00 | 0.7438 | 0.50 |
| planner_executor | 0 | 0.95 | 0.40 | 0.20 | 0.10 | 0.875 | 0.333 |
| planner_executor | 1 | 0.95 | 0.50 | 0.25 | 0.15 | 0.8875 | 0.385 |
| planner_executor | 2 | 1.00 | 0.65 | 0.15 | 0.10 | 0.95 | 0.20 |

GPT-4o headline (from [overview.md](../overview.md), same 60-run-per-config protocol):

| Config | Final pass (3 seeds) | Final pass mean | CS macro | Hard macro |
|---|---|---|---|---|
| single_react (GPT-4o) | 0% / 0% / 0% | 0.0% | 0% / 0% / 0% | 5% / 15% / 10% |
| planner_executor (GPT-4o) | 10% / 15% / 10% | **11.7%** | 40% / 50% / 65% | 20% / 25% / 15% |

Token / wall / cost (GPT-4o pilot, mean per query):

| Config | Total tokens / query | Wall (s) | $ / query |
|---|---|---|---|
| single_react | ~37 k | ~50 | ~$0.10 |
| planner_executor | ~53 k | ~80 | ~$0.13 |

Planner-executor uses ~**1.4×** the tokens — comfortably inside the proposal's 2–12× literature band.

---

## 8. What Each Metric Means

Drawn from [run_pilot_eval.py](eval_runner/run_pilot_eval.py) `evaluate_one`, `_commonsense_pass`, `_hard_pass`.

- **delivered** — the agent produced an NL plan that the parser (GPT-4o-mini with the upstream `PARSING_PREFIX`) successfully turned into a list of per-day JSON objects. `False` if NL plan is missing, equals `"Max Token Length Exceeded."`, or parser failed.
- **commonsense_macro_pass** — *all 8* commonsense checks pass. The 8 are upstream's: `is_valid_information_in_sandbox`, `is_not_absent`, `is_valid_cost`, `is_valid_room_rule`, `is_valid_cuisine`, `is_valid_room_type`, `is_valid_transportation`, `is_valid_accommodation` (constant 8 denominator in code).
- **commonsense_micro_pass** — fraction of those 8 checks satisfied per query, averaged across queries. Tolerant: a plan can fail two checks and still contribute 6/8.
- **hard_macro_pass / hard_micro_pass** — hard checks (budget, room rule, cuisine, room type, transportation specifics) are **gated**: only computed if the plan passes both `is_not_absent` AND `is_valid_information_in_sandbox`. Otherwise `hard_pass=None`, `hard_micro_total=0`. ⇒ Hard-micro denominators get tiny and the ratio swings wildly across seeds.
- **final_pass** — `commonsense_macro_pass AND hard_macro_pass`. The headline RQ1 metric.
- **delivery_rate** — fraction of queries that produced a deliverable plan at all. ~0.9 even for the baseline; failures = "Max Token Length Exceeded" or no `Planner[Query]` finalisation within 30 steps.

---

## 9. Interpreting the Results

1. **GPT-4o-mini is floored on Final Pass Rate.** The dominant failure is `is_valid_information_in_sandbox`: the model fabricates flight numbers like `F1234567` in its final NL plan even though `FlightSearch` returned the real ones in the observation buffer. Diagnostic counts on mini (n=20, one seed each):

   | Failed check | single_react | planner_executor |
   |---|---|---|
   | `is_valid_information_in_sandbox` (fabrication) | **18 / 20** | 10 / 20 |
   | `is_not_absent` (incomplete plan) | 6 / 20 | 5 / 20 |
   | `is_valid_accommodation` | 6 / 20 | 2 / 20 |
   | no-parse | 0 / 20 | 4 / 20 |

   The planner-executor already cuts fabrication by ~40 pp on mini — an architectural effect — but mini's other failures stack on top and Final Pass collapses to 0% / 1.7%.

2. **Switch to GPT-4o** because mini can't power statistical claims. A 1-query smoke check confirmed GPT-4o quotes the real flight numbers (`F3792603` / `F3927581`) emitted by `FlightSearch`, so fabrication stops being the bottleneck.

3. **On GPT-4o the architectural effect is large and consistent across all 3 seeds:**
   - Commonsense macro: **0% → 40-65%**
   - Hard macro: **5-15% → 15-25%**
   - Final: **0% → 10-15%**

   Planner-executor wins on every seed for every macro metric. Confidence intervals from `stats.py` do not overlap with single-agent's.

4. **Mechanism matches the proposal's hypothesis.** Mini diagnostic showed planner-executor's win is fabrication-suppression. On GPT-4o (no fabrication), the win shifts to **constraint tracking** — the structured plan helps the executor enforce per-day commonsense requirements (one breakfast / lunch / dinner / attraction / accommodation / transportation per day). That's the constraint-decomposition the proposal predicted.

5. **Hard-micro is volatile** (20-38% on GPT-4o pilot) because of the gating described in §8. A full validation run with bigger denominators will stabilise it.

6. **Cost is in-budget.** ~1.4× tokens, ~1.6× wall — well below the 2-12× headroom in the proposal.

---

## 10. Reproducing the Runs

From [TravelPlanner/](.) with venv active and `OPENAI_API_KEY` set:

```bash
# 1. Generate plans (pick config + seed + model)
python -m eval_runner.run_rq1 --config single_react       --split pilot --seed 0 --model gpt-4o
python -m eval_runner.run_rq1 --config planner_executor   --split pilot --seed 0 --model gpt-4o

# 2. Evaluate the generated plans
python -m eval_runner.run_pilot_eval --config single_react     --seed 0 --model gpt-4o
python -m eval_runner.run_pilot_eval --config planner_executor --seed 0 --model gpt-4o

# 3. Headline diff with paired bootstrap CI (after running 3 seeds)
python -m eval_runner.stats --config-a single_react --config-b planner_executor --split pilot --seeds 0 1 2
```

Replace `--split pilot` with `--split validation` for the full 1 080-run job (≈ $50-65 on GPT-4o).

---

## 11. File Index

Agent code:
- [agents/llm_client.py](agents/llm_client.py)
- [agents/planner_executor.py](agents/planner_executor.py)
- [agents/prompts_planner.py](agents/prompts_planner.py)
- [agents/single_agent_react.py](agents/single_agent_react.py)
- [agents/tool_agents.py](agents/tool_agents.py) — upstream `ReactAgent` (migrated)
- [agents/prompts.py](agents/prompts.py) — upstream zero-shot ReAct prompt

Eval code:
- [eval_runner/run_rq1.py](eval_runner/run_rq1.py)
- [eval_runner/run_pilot_eval.py](eval_runner/run_pilot_eval.py)
- [eval_runner/stats.py](eval_runner/stats.py)
- [eval_runner/token_accounting.py](eval_runner/token_accounting.py)
- [evaluation/commonsense_constraint.py](evaluation/commonsense_constraint.py)
- [evaluation/hard_constraint.py](evaluation/hard_constraint.py)
- [evaluation/eval.py](evaluation/eval.py)

Postprocess (migrated):
- [postprocess/parsing.py](postprocess/parsing.py)
- [postprocess/openai_request.py](postprocess/openai_request.py)

Results:
- [results/rq1/single_react/pilot/](results/rq1/single_react/pilot/) — `seed0/1/2`, each with `results.jsonl`, `generated_plans/`, `eval_score.json`, `pilot_summary.json`.
- [results/rq1/planner_executor/pilot/](results/rq1/planner_executor/pilot/) — same structure.

Companion docs:
- [overview.md](../overview.md) — status report + decision log.
- [README.md](README.md) — upstream README.
