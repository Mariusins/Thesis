#!/bin/bash
#SBATCH --job-name=tp-rq1
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/s2971127/logs/output-%j.out
#SBATCH --error=/home/s2971127/logs/ErrorLogs-%j.err
#SBATCH --chdir=/home/s2971127/Thesis/Thesis/TravelPlanner
#
# Submit:
#   sbatch slurm/run_rq1.sbatch single_react      pilot      0
#   sbatch slurm/run_rq1.sbatch planner_executor  validation 0
#
# Override model (e.g. larger quant needs more VRAM — adjust --mem/--gres accordingly):
#   TP_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ sbatch slurm/run_rq1.sbatch ...

set -euo pipefail

CONFIG="${1:?usage: sbatch slurm/run_rq1.sbatch <config> <split> <seed>}"
SPLIT="${2:?}"
SEED="${3:?}"

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
# Convert "Qwen/Qwen2.5-7B-Instruct" -> "models--Qwen--Qwen2.5-7B-Instruct"
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
# ----------------------------------------------------------------------------
# cp -aL follows symlinks (HF hub stores blobs via symlink) so the staged
# copy is real files — vLLM reads from local NVMe, not NFS.
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
echo "[paths] MODEL staged at: $ScratchDir/hf_home/hub/$MODEL_CACHE_NAME/snapshots/staged"

# ============================================================================
# Start vLLM server
# ============================================================================
# FlashInfer's sampler needs nvcc for JIT compilation, which is not available
# on this cluster. Fall back to the PyTorch-native sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "[$(date -Is)] starting vLLM: $MODEL on port $PORT"
python -m vllm.entrypoints.openai.api_server \
    --model "$ScratchDir/hf_home/hub/$MODEL_CACHE_NAME/snapshots/staged" \
    --served-model-name "$MODEL" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --dtype auto \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.82}" \
    --max-model-len "${MAX_MODEL_LEN:-16384}" \
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

# vLLM has loaded the model into GPU memory — restore the NFS-side HF cache so
# the experiment script can load the TravelPlanner dataset from the local cache.
# Keep HF_HUB_OFFLINE=1: compute nodes have no internet; dataset must be pre-cached
# on the head node with: python -c "from datasets import load_dataset; load_dataset('osunlp/TravelPlanner','validation')"
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
export LIMIT="${LIMIT:-}"   # empty = all queries; set e.g. LIMIT=5 for a smoke test

echo "[run] CONFIG=$CONFIG  SPLIT=$SPLIT  SEED=$SEED  MODEL=$MODEL"
echo "[run] OUTPUT_DIR=$RESULTS_ROOT/$CONFIG/$SPLIT/seed$SEED"

# ============================================================================
# Run
# ----------------------------------------------------------------------------
# Results (small .jsonl files) are written directly to results/ on /home —
# that's the kind of lightweight, sequential output the HPC admin allows.
# Heavy I/O (model weights, HF cache) is all on $ScratchDir.
# ============================================================================
python -m eval_runner.run_rq1 \
    --backend vllm \
    --model "$MODEL" \
    --config "$CONFIG" \
    --split "$SPLIT" \
    --seed "$SEED" \
    --max-steps "$MAX_STEPS" \
    --results-root "$RESULTS_ROOT" \
    ${LIMIT:+--limit "$LIMIT"}

echo "[$(date -Is)] RQ1 done"
# Cleanup of $ScratchDir and vLLM are handled by the EXIT trap above.
