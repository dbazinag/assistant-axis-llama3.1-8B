#!/usr/bin/env bash
set -euo pipefail

# Ensure USER exists (RunAI sometimes doesn't set it)
: "${USER:=$(whoami)}"

# Detect scratch mount inside the container
if [ -d /dlabscratch1/"$USER" ]; then
  BASE=/dlabscratch1/"$USER"
elif [ -d /mnt/dlabscratch1/"$USER" ]; then
  BASE=/mnt/dlabscratch1/"$USER"
elif [ -d /mnt/dlab/scratch/dlabscratch1/"$USER" ]; then
  BASE=/mnt/dlab/scratch/dlabscratch1/"$USER"
else
  echo "ERROR: could not find scratch mount for $USER"
  exit 1
fi

echo "Using scratch BASE=$BASE"

# Network-only caches + tmp
export HF_HOME=$BASE/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=$BASE/.cache
export TORCH_HOME=$BASE/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=$BASE/.cache/torchinductor
export VLLM_CACHE_DIR=$BASE/.cache/vllm
export TMPDIR=$BASE/.tmp
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_DIR" "$TMPDIR"

# (Optional) HuggingFace auth if token passed into job
if command -v huggingface-cli >/dev/null 2>&1; then
  if [ -n "${HF_TOKEN:-}" ]; then
    echo "$HF_TOKEN" | huggingface-cli login --token --stdin
  fi
fi

# Repo + inputs/outputs
REPO_DIR="$BASE/assistant-axis-llama3.1-8B"
RESP_DIR="$BASE/assistant_axis_outputs/llama-3.1-8b/responses"
ACT_DIR="$BASE/assistant_axis_outputs/llama-3.1-8b/activations"
mkdir -p "$ACT_DIR"

cd "$REPO_DIR"

# Sanity checks (explicit, no assumptions)
test -d "$RESP_DIR"
test -f "$RESP_DIR/default.jsonl"

# Ensure deps (same env you used in step 1)
uv sync

# Run step 2 (use the same model as step 1)
uv run pipeline/2_activations.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --responses_dir "$RESP_DIR" \
  --output_dir "$ACT_DIR" \
  --batch_size 16 \
  --tensor_parallel_size 2
