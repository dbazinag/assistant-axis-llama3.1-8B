#!/usr/bin/env bash
set -euo pipefail
: "${USER:=$(whoami)}"

# ---- detect scratch mount ----
if [ -d /dlabscratch1/"$USER" ]; then BASE=/dlabscratch1/"$USER";
elif [ -d /mnt/dlabscratch1/"$USER" ]; then BASE=/mnt/dlabscratch1/"$USER";
elif [ -d /mnt/dlab/scratch/dlabscratch1/"$USER" ]; then BASE=/mnt/dlab/scratch/dlabscratch1/"$USER";
else echo "ERROR: could not find scratch mount for $USER"; exit 1; fi

REPO_DIR="$BASE/assistant-axis-llama3.1-8B"
RESP_DIR="$BASE/assistant_axis_outputs/llama-3.1-8b/responses_q50"
SCORES_DIR="$BASE/assistant_axis_outputs/llama-3.1-8b/scores_q50"

cd "$REPO_DIR"
test -d "$RESP_DIR"
mkdir -p "$SCORES_DIR"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set"
  exit 1
fi

uv sync

uv run pipeline/3_judge.py \
  --responses_dir "$RESP_DIR" \
  --roles_dir data/roles/instructions \
  --output_dir "$SCORES_DIR" \
  --judge_model gpt-4o-mini \
  --max_tokens 10 \
  --batch_size 50 \
  --requests_per_second 100
