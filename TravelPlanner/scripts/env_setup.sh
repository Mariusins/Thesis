#!/usr/bin/env bash
# One-time HPC environment bootstrap. Run from the TravelPlanner/ directory
# after cloning the repo. Idempotent: rerunnable.
#
#   ssh s2971127@hpc-head1.ewi.utwente.nl
#   git clone <your-private-repo-url> ~/Thesis
#   cd ~/Thesis/TravelPlanner
#   bash scripts/env_setup.sh
set -euo pipefail

# 1. CUDA module (vLLM needs >=12.1). Adjust version if `module avail cuda` shows different.
module purge
module load cuda/12.4 || module load cuda || echo "WARN: no cuda module; vLLM may still work via bundled wheels"

# 2. Python venv at ~/Thesis/.venv (home is NFS, shared across nodes).
PY=python3
$PY --version

if [ ! -d "$HOME/Thesis/.venv" ]; then
  $PY -m venv "$HOME/Thesis/.venv"
fi
source "$HOME/Thesis/.venv/bin/activate"

python -m pip install --upgrade pip wheel

# 3. Dependencies. vLLM pulls torch automatically.
pip install \
  "vllm>=0.6.0" \
  "openai>=1.40" \
  "datasets>=2.20" \
  "tqdm" \
  "pandas" \
  "scipy" \
  "gdown"

# 3b. Fetch TravelPlanner database (327 MB, Google Drive). Idempotent.
bash "$(dirname "$0")/fetch_database.sh"

# 4. Pre-download model weights to NFS-shared HF cache so all nodes hit local disk.
#    Override HF_HOME if you want to keep it on /local (faster but per-node).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
model = os.environ.get("TP_MODEL", "Qwen/Qwen2.5-7B-Instruct")
print(f"prefetching {model} to {os.environ['HF_HOME']}")
snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
print("done")
PY

echo "env_setup.sh OK. Submit jobs with: sbatch scripts/run_rq1.sbatch ..."
