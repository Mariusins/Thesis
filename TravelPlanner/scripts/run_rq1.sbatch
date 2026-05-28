#!/usr/bin/env bash
#SBATCH --job-name=tp-rq1
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rq1-%j.out
#SBATCH --error=logs/rq1-%j.err
#
# Fused job: start vLLM in background on the allocated GPU, wait for /health,
# run run_rq1 for both configs, then terminate the server.
#
# Submit examples:
#   sbatch scripts/run_rq1.sbatch single_react pilot 0
#   sbatch scripts/run_rq1.sbatch planner_executor validation 0
#
# Override model:
#   TP_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ sbatch ...   (needs RTX6000Pro 96GB node;
#                                                       request via --constraint or --gres)

set -euo pipefail

CONFIG="${1:?usage: sbatch run_rq1.sbatch <config> <split> <seed>}"
SPLIT="${2:?}"
SEED="${3:?}"

MODEL="${TP_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${VLLM_PORT:-8000}"

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

module purge
module load cuda/12.4 || module load cuda || true

source "$HOME/Thesis/.venv/bin/activate"

# Cache HF weights on the node's fast local scratch when available.
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

# Wait for server ready (max 10 min).
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

echo "[$(date -Is)] launching RQ1: config=$CONFIG split=$SPLIT seed=$SEED model=$MODEL"
python -m eval_runner.run_rq1 \
  --backend vllm \
  --model "$MODEL" \
  --config "$CONFIG" \
  --split "$SPLIT" \
  --seed "$SEED"

echo "[$(date -Is)] RQ1 done"
