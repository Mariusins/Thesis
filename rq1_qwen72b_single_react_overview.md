# RQ1 — Qwen2.5-72B-Instruct-AWQ · single_react · validation

**Model:** Qwen2.5-72B-Instruct (AWQ INT4), served via vLLM on 2× A40 (tensor-parallel).
**Config:** `single_react` (baseline, one agent, 7 tools, ≤30 ReAct steps).
**Split:** validation — 180 queries × 3 seeds = **540 runs**.
**Context window:** 16 384 tokens (this run predates the 32 768 fix; see caveats).

---

## Headline pass rates

All rates are per-plan macro unless marked *micro*. Mean is across seeds 0/1/2.

| Metric | seed0 | seed1 | seed2 | **Mean** |
|---|---:|---:|---:|---:|
| **Delivery rate** | 86.1% | 86.7% | 86.7% | **86.5%** |
| **Commonsense macro** (all 8 checks pass) | 0.6% | 0.6% | 0.6% | **0.6%** |
| **Hard macro** (all hard checks pass) | 6.1% | 7.2% | 7.2% | **6.9%** |
| **Final pass** (every check passes) | 0.6% | 0.6% | 0.6% | **0.6%** |
| Commonsense *micro* (per-check) | 62.4% | 62.6% | 62.5% | **62.5%** |
| Hard *micro* (per-check) | 72.9% | 74.0% | 74.0% | **73.6%** |

---

## Delivery breakdown (per seed, of 180)

| Outcome | seed0 | seed1 | seed2 |
|---|---:|---:|---:|
| Delivered (non-empty plan) | 155 | 156 | 156 |
| Halted / empty (hit 30-step cap, no plan) | 15 | 14 | 14 |
| Python error (**context overflow**, 16 384 cap) | 10 | 10 | 10 |

The 10 errors/seed are `BadRequestError: maximum context length 16384` — the ReAct scratchpad outgrew the window. The 32 768-token fix (`MAX_MODEL_LEN`) lands these back as deliveries in the next run.

---

## How to read these numbers

- **Delivery 86.5%** — the agent reliably produces a scorable plan. The ~13% gap is split between step-cap halts (~8%) and context overflows (~6%).
- **Commonsense micro 62.5% vs macro 0.6%** — individual commonsense checks pass most of the time, but a plan must pass **all 8** to count for macro, and that almost never happens. One weak link per plan sinks the macro rate. This is the classic TravelPlanner pattern.
- **Hard micro 73.6% is on a tiny denominator.** Hard checks are only evaluated on plans that already cleared the upstream gates — only **22–24 of 180 plans/seed** reached them (48–50 individual checks). High percentage, small sample; do not over-read it.
- **Final pass 0.6%** — essentially one query per seed passes everything. For reference, GPT-4-Turbo scores 0.6% on the published two-stage leaderboard, so single-agent 72B sits right at that floor.

## By difficulty (seed0)

| Level | n | Delivery | CS macro | Final |
|---|---:|---:|---:|---:|
| easy | 52 | 92% | 0% | 0% |
| medium | 59 | 86% | 0% | 0% |
| hard | 59 | 95% | 2% | 2% |
| (errored) | 10 | 0% | 0% | 0% |

---

## 72B vs 14B (single_react, paired)

Both models ran the identical validation split, enabling a paired comparison. All
figures use the same pooled-micro method (Σpasses / Σchecks), Δ = 72B − 14B:

| Metric | 14B | 72B | Δ (72B − 14B) |
|---|---:|---:|---:|
| Delivery | 85.6% | 86.5% | +0.9 pp |
| Commonsense macro | 0.2% | 0.6% | +0.4 pp |
| Commonsense micro | 62.3% | 62.5% | +0.2 pp |
| Hard macro | 0.4% | 6.9% | **+6.5 pp** |
| Hard micro | 25.0% | 73.6% | +48.6 pp † |
| Final pass | 0.2% | 0.6% | +0.4 pp |

† Hard-micro is computed over only the ~24 plans/run that reach the hard-constraint
gate (~50 checks); the large Δ rides on a tiny, volatile denominator — report it
with that caveat, not as a headline.

The 72B's clearest robust gain is in **hard-constraint satisfaction** (hard macro
+6.5 pp — a real per-plan effect, not denominator noise). Commonsense and final
pass stay essentially floored for the single-agent baseline on both models — which
is exactly the headroom the planner-executor architecture is meant to unlock (RQ1).

---

## Caveats

1. **Context window was 16 384**, not the fixed 32 768 — 30 plans/run were lost to overflow that the next run will recover.
2. **AWQ INT4 quantization** — small quality loss vs full-precision 72B; acceptable for research but worth a footnote.
3. **Hard-micro small denominator** (~24 plans) — volatile across seeds; report micro for commonsense, macro for the headline.
4. This is the **baseline arm only**. The RQ1 claim needs the `planner_executor` 72B run for the paired architecture comparison.
