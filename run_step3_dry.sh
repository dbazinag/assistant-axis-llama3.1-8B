#!/usr/bin/env bash
set -euo pipefail
: "${USER:=$(whoami)}"

# Detect scratch mount inside container
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

export HF_HOME=$BASE/.cache/huggingface
export XDG_CACHE_HOME=$BASE/.cache
export TMPDIR=$BASE/.tmp
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

REPO_DIR="$BASE/assistant-axis-llama3.1-8B"
RESP_DIR="$BASE/assistant_axis_outputs/llama-3.1-8b/responses"
SCORES_DIR="$BASE/assistant_axis_outputs/llama-3.1-8b/scores"

cd "$REPO_DIR"
test -d "$RESP_DIR"
mkdir -p "$SCORES_DIR"

uv sync

# Dry run: counts prompts and prints one sample judge prompt
uv run pipeline/3_judge.py \
  --responses_dir "$RESP_DIR" \
  --roles_dir data/roles/instructions \
  --output_dir "$SCORES_DIR" \
  --judge_model gpt-4.1-mini \
  --max_tokens 10 \
  --batch_size 50 \
  --requests_per_second 100 \
  --dry_run
