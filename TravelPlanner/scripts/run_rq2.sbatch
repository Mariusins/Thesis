#!/usr/bin/env bash
#SBATCH --job-name=tp-rq2
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rq2-%j.out
#SBATCH --error=logs/rq2-%j.err
#
# Single vLLM server serves all RQ2 configs in sequence.
#
# Submit:
#   sbatch scripts/run_rq2.sbatch self_consistency pilot 0
#   sbatch scripts/run_rq2.sbatch budget_capped    pilot 0
#
# Token budget defaults to 52000 (GPT-4o-mini anchor). After re-running the
# planner-executor pilot on Qwen, pass the new anchor:
#   sbatch scripts/run_rq2.sbatch self_consistency validation 0 <new_budget>

set -euo pipefail

CONFIG="${1:?usage: sbatch run_rq2.sbatch <config> <split> <seed> [token_budget] [n_samples]}"
SPLIT="${2:?}"
SEED="${3:?}"
TOKEN_BUDGET="${4:-52000}"
N_SAMPLES="${5:-2}"

MODEL="${TP_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${VLLM_PORT:-8000}"

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

module purge
module load cuda/12.4 || module load cuda || true

source "$HOME/Thesis/.venv/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

echo "[$(date -Is)] starting vLLM: $MODEL on port $PORT"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --host 127.0.0.1 \
  --dtype auto \
  --gpu-memory-utilization 0.92 \
  --max-model-len 16384 \
  --disable-log-requests \
  > "logs/vllm-${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "[$(date -Is)] stopping vLLM pid=$VLLM_PID"
  kill -TERM "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null; then
    echo "[$(date -Is)] vLLM ready after ${i}*5s"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM died during startup; see logs/vllm-${SLURM_JOB_ID}.log" >&2
    exit 1
  fi
  sleep 5
done

echo "[$(date -Is)] launching RQ2: config=$CONFIG split=$SPLIT seed=$SEED budget=$TOKEN_BUDGET"
python -m eval_runner.run_rq2 \
  --backend vllm \
  --model "$MODEL" \
  --config "$CONFIG" \
  --split "$SPLIT" \
  --seed "$SEED" \
  --token-budget "$TOKEN_BUDGET" \
  --n-samples "$N_SAMPLES"

echo "[$(date -Is)] RQ2 done"
