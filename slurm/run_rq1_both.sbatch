#!/bin/bash
#SBATCH --job-name=tp-rq1-both
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ampere:2
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/s2971127/logs/output-%j.out
#SBATCH --error=/home/s2971127/logs/ErrorLogs-%j.err
#SBATCH --chdir=/home/s2971127/Thesis/Thesis/TravelPlanner
#
# Run BOTH configs (single_react + planner_executor) for all given seeds in ONE
# job. vLLM starts once and is shared across every config/seed combination.
#
# Submit (the full temp=0.7, 50-step, 72B run):
#   TP_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ \
#   GPU_MEM_UTIL=0.90 TP_SIZE=2 MAX_MODEL_LEN=32768 \
#   TP_TEMPERATURE=0.7 MAX_STEPS=50 \
#   RESULTS_ROOT=results/rq1_qwen72b_t07_s50 \
#     sbatch slurm/run_rq1_both.sbatch validation 0 1 2

set -euo pipefail

SPLIT="${1:?usage: sbatch slurm/run_rq1_both.sbatch <split> <seed> [seed2] ...}"
shift 1
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && { echo "ERROR: at least one seed required" >&2; exit 1; }

CONFIGS=(single_react planner_executor)

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
    echo "       Run: hf download $MODEL" >&2
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
# Start vLLM server (once, shared across all configs and seeds)
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
export TP_TEMPERATURE="${TP_TEMPERATURE:-0.0}"   # >0 makes seeds diverge (real variance)
export LIMIT="${LIMIT:-}"

echo "[run] CONFIGS=${CONFIGS[*]}  SPLIT=$SPLIT  MODEL=$MODEL"
echo "[run] Seeds: ${SEEDS[*]}"
echo "[run] RESULTS_ROOT=$RESULTS_ROOT  MAX_STEPS=$MAX_STEPS  TP_TEMPERATURE=$TP_TEMPERATURE"

# ============================================================================
# Run every config × seed against the single vLLM instance
# ============================================================================
failed=()

for CONFIG in "${CONFIGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "========================================================"
        echo "[$(date -Is)] CONFIG=$CONFIG SEED=$SEED  (OUTPUT: $RESULTS_ROOT/$CONFIG/$SPLIT/seed$SEED)"
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
            echo "[$(date -Is)] $CONFIG seed $SEED done OK"
        else
            echo "[$(date -Is)] ERROR: $CONFIG seed $SEED exited non-zero" >&2
            failed+=("$CONFIG/seed$SEED")
        fi
    done
done

echo ""
echo "========================================================"
echo "[$(date -Is)] All config/seed combinations finished."
if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
else
    echo "All combinations completed successfully."
fi
# Cleanup handled by EXIT trap.
