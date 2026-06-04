#!/bin/bash
#SBATCH --job-name=tp-rq1-ms
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/s2971127/logs/output-%j.out
#SBATCH --error=/home/s2971127/logs/ErrorLogs-%j.err
#SBATCH --chdir=/home/s2971127/Thesis/Thesis/TravelPlanner
#
# Run multiple seeds in one job — vLLM starts once, seeds run sequentially.
#
# Submit:
#   sbatch slurm/run_rq1_multi.sbatch planner_executor  validation 0 1 2
#   sbatch slurm/run_rq1_multi.sbatch single_react      validation 0 1 2
#   sbatch slurm/run_rq1_multi.sbatch planner_executor  validation 1 2   # resume missing seeds
#
# Override model:
#   TP_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ sbatch slurm/run_rq1_multi.sbatch ...

set -euo pipefail

CONFIG="${1:?usage: sbatch slurm/run_rq1_multi.sbatch <config> <split> <seed> [seed2] [seed3] ...}"
SPLIT="${2:?}"
shift 2
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && { echo "ERROR: at least one seed required" >&2; exit 1; }

MODEL="${TP_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${VLLM_PORT:-8000}"

# ============================================================================
# Local scratch (per HPC admin: do NOT do heavy I/O on /home)
# ============================================================================
ScratchDir="/local/${SLURM_JOB_ID}"

cleanup_node() {
    set +e
    if [ -n "${VLLM_PID:-}" ]; then
        echo "[cleanup] stopping vLLM pid=$VLLM_PID"
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    if [ -d "$ScratchDir" ]; then
        echo "[cleanup] removing $ScratchDir"
        rm -rf "$ScratchDir"
    fi
}
trap cleanup_node EXIT SIGINT SIGTERM

if [ -d "$ScratchDir" ]; then
    echo "[scratch] '$ScratchDir' already exists (stale?). Reusing."
else
    echo "[scratch] creating '$ScratchDir'"
    mkdir -p "$ScratchDir"
fi

# ============================================================================
# Conda env
# ============================================================================
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Marius

# ============================================================================
# Resolve HF model snapshot on /home (read once, then stage)
# ============================================================================
MODEL_CACHE_NAME="models--$(echo "$MODEL" | sed 's|/|--|g')"
MODEL_CACHE_HOME="$HOME/.cache/huggingface/hub/$MODEL_CACHE_NAME"
MODEL_SRC=$(ls -dt "$MODEL_CACHE_HOME"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
[ -z "$MODEL_SRC" ] && {
    echo "ERROR: $MODEL not cached at $MODEL_CACHE_HOME" >&2
    echo "       Run: huggingface-cli download $MODEL" >&2
    exit 1
}

# ============================================================================
# Stage model to local scratch
# ============================================================================
mkdir -p "$ScratchDir/hf_home/hub/$MODEL_CACHE_NAME/snapshots/staged"

echo "[stage] copying $MODEL ($MODEL_SRC) -> $ScratchDir/hf_home"
cp -aL "$MODEL_SRC"/. "$ScratchDir/hf_home/hub/$MODEL_CACHE_NAME/snapshots/staged"/

echo "[stage] staged size:"
du -sh "$ScratchDir/hf_home" 2>/dev/null || true

# ============================================================================
# Re-bind HF / Torch caches to scratch
# ============================================================================
export HF_HOME="$ScratchDir/hf_home"
export HF_HUB_CACHE="$ScratchDir/hf_home/hub"
export TRANSFORMERS_CACHE="$ScratchDir/hf_home"
export HF_HUB_OFFLINE=1
export TORCH_HOME="$ScratchDir/torch"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

echo "[paths] HF_HOME=$HF_HOME  TORCH_HOME=$TORCH_HOME"

# ============================================================================
# Start vLLM server (once, shared across all seeds)
# ============================================================================
export VLLM_USE_FLASHINFER_SAMPLER=0

TP_SIZE="${TP_SIZE:-1}"
echo "[$(date -Is)] starting vLLM: $MODEL on port $PORT  (tensor_parallel_size=$TP_SIZE)"
python -m vllm.entrypoints.openai.api_server \
    --model "$ScratchDir/hf_home/hub/$MODEL_CACHE_NAME/snapshots/staged" \
    --served-model-name "$MODEL" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --dtype auto \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.82}" \
    --max-model-len "${MAX_MODEL_LEN:-16384}" \
    --tensor-parallel-size "$TP_SIZE" \
    --no-enable-log-requests \
    > "/home/s2971127/logs/vllm-${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

# Wait for server ready (max 20 min = 240 * 5s)
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
echo "[$(date -Is)] waiting for vLLM /health ..."
vllm_ready=0
for i in $(seq 1 240); do
    if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health')" 2>/dev/null; then
        echo "[$(date -Is)] vLLM ready after ${i}*5s"
        vllm_ready=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM died during startup; see /home/s2971127/logs/vllm-${SLURM_JOB_ID}.log" >&2
        tail -30 "/home/s2971127/logs/vllm-${SLURM_JOB_ID}.log" >&2
        exit 1
    fi
    sleep 5
done
if [ "$vllm_ready" -eq 0 ]; then
    echo "ERROR: vLLM did not become ready within 1200s; see /home/s2971127/logs/vllm-${SLURM_JOB_ID}.log" >&2
    exit 1
fi

# Restore NFS-side HF cache so the experiment script can load the dataset.
# Keep HF_HUB_OFFLINE=1: compute nodes have no internet.
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE

# ============================================================================
# Tunables
# ============================================================================
export TIKTOKEN_CACHE_DIR="$HOME/.cache/tiktoken"
export MAX_STEPS="${MAX_STEPS:-30}"
export RESULTS_ROOT="${RESULTS_ROOT:-results/rq1}"
export LIMIT="${LIMIT:-}"

echo "[run] CONFIG=$CONFIG  SPLIT=$SPLIT  MODEL=$MODEL"
echo "[run] Seeds to run: ${SEEDS[*]}"
echo "[run] RESULTS_ROOT=$RESULTS_ROOT  MAX_STEPS=$MAX_STEPS"

# ============================================================================
# Run all seeds sequentially against the single vLLM instance
# ============================================================================
failed_seeds=()

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "========================================================"
    echo "[$(date -Is)] Starting seed $SEED  (OUTPUT: $RESULTS_ROOT/$CONFIG/$SPLIT/seed$SEED)"
    echo "========================================================"

    if python -m eval_runner.run_rq1 \
        --backend vllm \
        --model "$MODEL" \
        --config "$CONFIG" \
        --split "$SPLIT" \
        --seed "$SEED" \
        --max-steps "$MAX_STEPS" \
        --results-root "$RESULTS_ROOT" \
        ${LIMIT:+--limit "$LIMIT"}; then
        echo "[$(date -Is)] Seed $SEED done OK"
    else
        echo "[$(date -Is)] ERROR: seed $SEED exited with non-zero status" >&2
        failed_seeds+=("$SEED")
    fi
done

echo ""
echo "========================================================"
echo "[$(date -Is)] All seeds finished."
if [ ${#failed_seeds[@]} -gt 0 ]; then
    echo "FAILED seeds: ${failed_seeds[*]}" >&2
    exit 1
else
    echo "All seeds completed successfully."
fi
# Cleanup of $ScratchDir and vLLM are handled by the EXIT trap above.
