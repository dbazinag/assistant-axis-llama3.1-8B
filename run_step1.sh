#!/usr/bin/env bash
# Ensure USER is set in RunAI containers

: "${USER:=$(whoami)}"

set -euo pipefail

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

# Network-only caches
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

cd "$BASE/assistant-axis-llama3.1-8B"

# Sanity check inputs
test -f data/extraction_questions.jsonl
test -d data/roles/instructions

# Ensure deps
uv sync

# Full Step 1 generation (paper defaults, but correct explicit paths)
mkdir -p "$BASE/assistant_axis_outputs/llama-3.1-8b/responses"

uv run pipeline/1_generate.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir "$BASE/assistant_axis_outputs/llama-3.1-8b/responses" \
  --roles_dir data/roles/instructions \
  --questions_file data/extraction_questions.jsonl \
  --tensor_parallel_size 2
