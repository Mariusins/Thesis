# CLAUDE.md — Project Context

Read this at the start of every session. It captures the full context needed to work on this project without re-deriving it from scratch.

---

## What this project is

Master's thesis by Marius Inselseth (University of Twente). The research question is:

> **RQ1:** Under matched compute, does a planner-executor two-agent architecture beat a single-agent ReAct baseline on the TravelPlanner benchmark?

> **RQ2:** How do compute-matched baselines (self-consistency, budget-capped single-agent) compare to planner-executor?

The codebase is a fork of `OSU-NLP-Group/TravelPlanner`, migrated off legacy `langchain==0.1.4` + `openai==0.27.2` to modern `openai>=1.40`, with a new two-agent architecture and evaluation harness added on top.

---

## Repo layout

```
/home/s2971127/Thesis/Thesis/
├── CLAUDE.md                        ← this file
├── overview.md                      ← thesis status report + pilot results
├── requirements.txt                 ← pinned env (authoritative)
├── slurm/
│   ├── run_rq1.sbatch               ← SLURM job for RQ1 evaluation
│   ├── run_rq2.sbatch               ← SLURM job for RQ2 evaluation
│   └── Example.sbatch               ← different project (Wan2.2 video), ignore for TravelPlanner work
├── models/Qwen2.5-7B-Instruct/      ← model config files only (weights in HF cache)
└── TravelPlanner/
    ├── agents/
    │   ├── llm_client.py            ← single LLM entry point + TokenLedger
    │   ├── planner_executor.py      ← PlannerAgent + PlannerExecutorAgent (new)
    │   ├── single_agent_react.py    ← baseline shim around ReactAgent (new)
    │   ├── self_consistency.py      ← RQ2: N samples, pick first valid
    │   ├── budget_capped_react.py   ← RQ2: raised step cap + token budget
    │   ├── tool_agents.py           ← upstream ReactAgent (migrated)
    │   ├── prompts.py               ← upstream zero-shot ReAct prompt
    │   └── prompts_planner.py       ← planner system prompt + JSON schema (new)
    ├── eval_runner/
    │   ├── run_rq1.py               ← RQ1 CLI orchestrator (resumable)
    │   ├── run_rq2.py               ← RQ2 CLI orchestrator (resumable)
    │   ├── run_pilot_eval.py        ← pilot evaluator (parse NL plan → JSON → score)
    │   ├── stats.py                 ← paired bootstrap CI
    │   └── token_accounting.py      ← token aggregation helpers
    ├── evaluation/                  ← upstream constraint checkers (untouched)
    ├── tools/                       ← 7 sandbox tools (untouched)
    ├── database/                    ← static CSVs (flights, hotels, etc.)
    ├── results/
    │   ├── rq1/{config}/{split}/seed{N}/   ← results.jsonl, eval_score.json, pilot_summary.json
    │   └── rq2/{config}/{split}/seed{N}/
    ├── CODEBASE_EXPLAINED.md        ← full architectural reference
    └── requirements.txt             ← TravelPlanner-specific pins (keep in sync)
```

---

## HPC environment

**Cluster:** University of Twente HPC (hpc-head2), SLURM scheduler.

**Conda env:** `Marius` — always activate before running anything.

**Key paths:**
- Home: `/home/s2971127/`
- Project: `/home/s2971127/Thesis/Thesis/`
- Logs: `/home/s2971127/logs/` — SLURM stdout as `output-<jobid>.out`, stderr as `ErrorLogs-<jobid>.err`, vLLM as `vllm-<jobid>.log`
- Model HF cache: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/`
- Scratch (per job): `/local/<SLURM_JOB_ID>/` — local NVMe, wiped on job exit by the EXIT trap

**Nodes used:**
- `ctit084` — rq1 (currently, check sbatch for latest)
- `ctit092` — rq2
- `ctit090` — occasionally used
- GPU: ~44.42 GiB VRAM (A40-class) — but real available at startup is ~36.5 GiB due to stale contexts; set `--gpu-memory-utilization 0.82`

**pip note:** The global pip config (`~/.config/pip/pip.conf`) forces the UT Twente private GitLab registry. That registry does not have most PyPI packages. Always install with:
```bash
pip install --index-url https://pypi.org/simple '<package>'
```

---

## Model

**Qwen/Qwen2.5-7B-Instruct** — served locally via vLLM on the SLURM job node.

- Weights: 14.19 GiB (4 safetensor shards), bfloat16
- Max context: 16 384 tokens
- vLLM version: 0.21.0
- PyTorch: 2.11.0+cu130 (CUDA 13.0)

The model is staged from `~/.cache/huggingface/hub/` to `/local/<jobid>/` at job start to avoid NFS I/O during inference. vLLM is launched as a local OpenAI-compatible API server on `127.0.0.1:8000`.

The eval scripts access the model via `OPENAI_BASE_URL=http://127.0.0.1:8000/v1` and `--backend vllm` flag. The `LLMClient` in `agents/llm_client.py` reads `OPENAI_BASE_URL` automatically when using the `openai` SDK.

---

## Known issues and fixes already applied

### 1. FlashInfer JIT fails (no nvcc on PATH)
`flashinfer` tries to JIT-compile CUDA kernels on startup using `nvcc`, which isn't on the default PATH. Fixed by setting:
```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```
Already in both sbatch files. Uses the PyTorch-native sampler instead — no difference in output quality. (nvcc does exist at `/software/nvidia/hpc-sdk/Linux_x86_64/26.3/cuda/13.1/bin/nvcc` if needed for something else.)

### 2. GPU memory error on startup
The node often has ~7.8 GiB of GPU VRAM already occupied (stale CUDA contexts from previous failed jobs). `--gpu-memory-utilization 0.92` (= 40.87 GiB) exceeds the ~36.5 GiB actually free.
Fixed by lowering to `--gpu-memory-utilization 0.82` (= 36.42 GiB). Already in both sbatch files.
If the node is clean (fresh boot), you can raise it back to 0.90 to gain more KV cache.

### 3. `datasets` / `huggingface_hub` version mismatch
Installed `datasets==2.14.4` imported `HfFolder` from `huggingface_hub==1.17.0` which no longer exports it.
Fix: `pip install --index-url https://pypi.org/simple 'datasets==2.21.0'`
`requirements.txt` already pins `datasets==2.21.0`. Run the pip install if the conda env is freshly built or after package upgrades.

### 4. torch.compile cache on first run
vLLM's torch.compile takes ~45-100 seconds on the **first** job run; cached to `~/.cache/vllm/torch_compile_cache/` on `/home` (NFS). Subsequent jobs reuse the cache and complete compilation in ~30 seconds. The 10-minute vLLM health-check timeout (120 × 5s) accounts for this.

---

## How to run

### Submit a SLURM job
```bash
# RQ1 — single agent or planner-executor
sbatch slurm/run_rq1.sbatch single_react      pilot      0
sbatch slurm/run_rq1.sbatch planner_executor  validation 0

# RQ2 — self-consistency or budget-capped
sbatch slurm/run_rq2.sbatch self_consistency  pilot      0
sbatch slurm/run_rq2.sbatch budget_capped     pilot      0

# Override model
TP_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ sbatch slurm/run_rq1.sbatch ...

# RQ2: override token budget via env var (positional arg $4 takes precedence if given)
TOKEN_BUDGET=60000 sbatch slurm/run_rq2.sbatch self_consistency validation 0
```

### Run interactively (no SLURM, model already running elsewhere)
```bash
cd /home/s2971127/Thesis/Thesis/TravelPlanner
conda activate Marius
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
python -m eval_runner.run_rq1 --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
    --config single_react --split pilot --seed 0 --max-steps 30
```

### Evaluate pilot results
```bash
python -m eval_runner.run_pilot_eval --config single_react     --seed 0
python -m eval_runner.run_pilot_eval --config planner_executor --seed 0
python -m eval_runner.stats --config-a single_react --config-b planner_executor \
    --split pilot --seeds 0 1 2
```

### Monitor jobs
```bash
squeue -u s2971127
tail -f /home/s2971127/logs/output-<jobid>.out
tail -f /home/s2971127/logs/vllm-<jobid>.log
```

### Cancel jobs
```bash
scancel <jobid> [<jobid2> ...]
```

---

## Architecture summary

### RQ1 configs

**`single_react`** — baseline. `SingleAgentReact` wraps the upstream `ReactAgent`. One agent, all 7 tools, ≤30 ReAct steps. LLM calls go through `LLMClient`, usage tracked in `TokenLedger`.

**`planner_executor`** — two-stage. `PlannerAgent` makes a single structured JSON call (no tools, `response_format=json_object`, validated against `PLANNER_JSON_SCHEMA` with 8 sub-goal types). If the plan validates, its JSON is prepended to the executor's prompt template. The executor is the same `ReactAgent` as the baseline. Shared `TokenLedger` across both stages. On plan-parse failure → degrades to plain ReAct.

### RQ2 configs

**`self_consistency`** — runs N `SingleAgentReact` samples, picks the first that delivered a valid plan.

**`budget_capped`** — single `SingleAgentReact` run with raised step cap and a token budget ceiling matched to the planner-executor's mean token cost (~52k/query).

### Token accounting

`TokenLedger` accumulates `prompt_tokens`, `completion_tokens`, `total_tokens`, `n_calls` globally and per `role_tag` (`planner_agent`, `react_agent`, etc.). Serialised into every `results.jsonl` record.

### Resumability

Both `run_rq1.py` and `run_rq2.py` read existing `results.jsonl` on startup and skip any query `idx` already present. Safe to resubmit a job after a partial run.

---

## Key results so far (pilot, n=20 queries × 3 seeds)

| Config | Model | Final pass mean | CS macro |
|---|---|---|---|
| single_react | GPT-4o-mini | 0.0% | 0% |
| planner_executor | GPT-4o-mini | 1.7% | ~2% |
| single_react | GPT-4o | 0.0% | 0% |
| **planner_executor** | **GPT-4o** | **11.7%** | **40-65%** |

Mini is floored due to flight-number fabrication (18/20 plans fail `is_valid_information_in_sandbox`). GPT-4o doesn't fabricate. Current work: switching to Qwen2.5-7B-Instruct locally to run the full validation split without per-query API cost.

Token cost: planner-executor uses ~1.4× tokens vs single-react — within the 2-12× literature bound cited in the proposal.

---

## Current status (as of 2026-05-29)

- Pilot runs on GPT-4o complete, results in `results/rq1/*/pilot/`
- Migrating from GPT-4o API to local Qwen2.5-7B-Instruct via vLLM for the full validation run
- vLLM startup issues resolved (FlashInfer, GPU memory, datasets version)
- First successful vLLM job ran (job 504296, single_react validation seed 0), started experiment but crashed on `datasets` import — fixed with `pip install datasets==2.21.0`
- Next step: confirm the datasets fix holds, then run full validation (single_react + planner_executor × validation × seeds 0/1/2)
