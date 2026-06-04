# RQ1 — single_react across Qwen model sizes

All runs: `single_react` baseline, validation split, 180 queries × 3 seeds = **540 runs**,
30-step ReAct cap, 16 384-token context. Models served locally via vLLM.
Micro rates use the pooled method (Σpasses / Σchecks). Means across seeds 0/1/2.

## Pass rates

| Metric | Qwen2.5-7B | Qwen2.5-14B | Qwen2.5-72B-AWQ |
|---|---:|---:|---:|
| **Delivery rate** | 77.0% | 85.6% | **86.5%** |
| **Commonsense macro** | 0.2% | 0.2% | **0.6%** |
| Commonsense micro | 48.5% | 62.3% | **62.5%** |
| **Hard macro** | 0.4% | 0.4% | **6.9%** |
| Hard micro † | 25.4% | 25.0% | **73.6%** |
| **Final pass** | 0.2% | 0.2% | **0.6%** |

† **Hard-micro is not comparable across these rows.** It is computed only over plans
that reach the hard-constraint gate. Mean checks evaluated per seed: **7B ≈ 4, 14B ≈ 2,
72B ≈ 49**. The 7B/14B percentages are 1–2 checks out of a handful — statistical noise.
Only the 72B figure rests on a usable sample. Use **hard macro** for cross-model claims.

## Delivery breakdown & cost (540 runs each)

| Model | Delivered | Halted (step cap) | Error (ctx overflow) | Tokens / query | LLM calls / query |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B | 416 | 117 | 7 | 82 439 | 32.8 |
| Qwen2.5-14B | 462 | 31 | 47 | 77 928 | 31.9 |
| Qwen2.5-72B-AWQ | 467 | 43 | 30 | 87 590 | 34.3 |

## What the comparison shows

1. **Delivery climbs with model size** (77% → 85.6% → 86.5%), but the *mechanism shifts*:
   the 7B fails mostly by **looping until the step cap** (117 halts) — it can't converge.
   The 14B/72B almost stop halting, and their main remaining loss flips to **context
   overflow** (47 / 30 errors) — they pack the scratchpad with tool output and blow the
   16 384 window. That failure is recoverable with the 32 768-token fix.

2. **7B → 14B buys *reasoning quality*, not pass rate.** Commonsense micro jumps +13.8 pp
   (48.5% → 62.3%) — the 14B satisfies far more individual checks — yet macro and final
   pass stay flat at ~0.2%. More checks pass, but never *all* at once.

3. **The real jump is 14B → 72B on hard constraints.** Hard macro goes 0.4% → 6.9%
   (+6.5 pp) — the only robust per-plan gain in the whole table. Commonsense micro
   plateaus (62.3% → 62.5%), so the 72B's edge is specifically in **budget / room-rule /
   cuisine / transport** satisfaction, not generic commonsense.

4. **Final pass is floored for every single-agent model** (≤0.6%). For reference,
   GPT-4-Turbo scores 0.6% on the published two-stage leaderboard. This is the expected
   baseline ceiling — and the headroom the **planner-executor** architecture (RQ1) is
   designed to unlock. Scaling the base model alone does not move final pass.

5. **Cost is flat across sizes** (~78–88 k tokens/query, ~32–34 calls/query). Bigger
   models are not burning more steps; they use the same compute budget and convert it
   into better intermediate quality. Wall-clock differs (72B is slower per token), but
   the token/step budget is matched — which keeps the RQ1 compute-matching assumption intact.

## Caveats

- **72B is AWQ INT4** quantized; 7B/14B are bf16. Small quality discount on the 72B vs its
  full-precision form — the gains above are therefore conservative.
- All runs used the **16 384** context window; the 32 768 fix will recover the overflow
  errors (especially for 14B/72B) and likely nudge delivery a few points higher.
- `results/rq1_steps50` (7B, 50-step) and the 72B `planner_executor` arm are **not yet
  evaluated** — add them when available.
