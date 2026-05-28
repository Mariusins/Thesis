# Thesis
Marius Inselseth Bachelor Thesis

## Running on the EEMCS-HPC cluster

The TravelPlanner harness is now backend-agnostic: a `--backend {openai,vllm}` flag
on every eval runner picks between the hosted API and a local vLLM server. Code
changes were minimal — `agents/llm_client.py` honours `OPENAI_BASE_URL`, so any
OpenAI-compatible endpoint works.

### One-time setup

1. SSH to head node:
   ```bash
   ssh s2971127@hpc-head1.ewi.utwente.nl
   ```
2. Clone the repo (private GitHub repo recommended):
   ```bash
   git clone <repo-url> ~/Thesis
   cd ~/Thesis/TravelPlanner
   ```
3. Bootstrap venv + database + prefetch model weights:
   ```bash
   bash scripts/env_setup.sh
   ```
   This installs deps, downloads the 327 MB TravelPlanner database from
   Google Drive (via `scripts/fetch_database.sh`, idempotent), and prefetches
   the model weights. Default model: `Qwen/Qwen2.5-7B-Instruct`. Override with
   `TP_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ bash scripts/env_setup.sh` if scaling up.

   If you only need the database (e.g. local dev after a fresh clone):
   ```bash
   bash scripts/fetch_database.sh
   ```

### Submitting jobs

Each sbatch script requests 1 GPU + 8 cores + 64 GB RAM on the `main` partition.
It launches vLLM in the background, waits for `/health`, runs the eval against
`OPENAI_BASE_URL=http://127.0.0.1:8000/v1`, then terminates the server.

```bash
# RQ1 pilot
sbatch scripts/run_rq1.sbatch single_react    pilot 0
sbatch scripts/run_rq1.sbatch planner_executor pilot 0

# RQ2 pilot (token budget is the GPT-4o-mini anchor; re-tune after RQ1 pilot)
sbatch scripts/run_rq2.sbatch self_consistency pilot 0 52000 2
sbatch scripts/run_rq2.sbatch budget_capped    pilot 0 52000
```

Monitor with `squeue -u $USER` and `tail -f logs/rq1-<jobid>.out`.

### Re-tuning the token budget

GPT-4o-mini's 52 k/query anchor will not match Qwen2.5-7B's token economy. After
the RQ1 pilot completes, compute the new anchor:

```bash
python -m eval_runner.token_accounting \
  --results results/rq1/planner_executor/pilot/seed0/results.jsonl
```

Use the reported `summary.total.mean` as the `token_budget` for the RQ2 sbatch
calls on the validation split.

### Scaling up to 72B

If 7B underperforms on TravelPlanner, switch to Qwen2.5-72B-Instruct-AWQ. It
fits one RTX6000 Pro 96 GB node (hpc-node15) or one H200 NVL 141 GB
(hpc-node19). Add a node constraint in the sbatch header, e.g.:

```bash
#SBATCH --nodelist=hpc-node15
```

and submit with `TP_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ`.

### Pulling results back

```bash
rsync -av s2971127@hpc-head1.ewi.utwente.nl:~/Thesis/TravelPlanner/results/ \
  ./TravelPlanner/results/
```
