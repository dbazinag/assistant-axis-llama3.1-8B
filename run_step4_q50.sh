#!/usr/bin/env bash
set -euo pipefail
: "${USER:=$(whoami)}"

if [ -d /dlabscratch1/"$USER" ]; then BASE=/dlabscratch1/"$USER";
elif [ -d /mnt/dlabscratch1/"$USER" ]; then BASE=/mnt/dlabscratch1/"$USER";
elif [ -d /mnt/dlab/scratch/dlabscratch1/"$USER" ]; then BASE=/mnt/dlab/scratch/dlabscratch1/"$USER";
else echo "ERROR: could not find scratch mount for $USER"; exit 1; fi

REPO_DIR="$BASE/assistant-axis-llama3.1-8B"
ACT_DIR="$BASE/assistant_axis_outputs/llama-3.1-8b/activations"
SCO_DIR="$BASE/assistant_axis_outputs/llama-3.1-8b/scores_q50"
VEC_DIR="$BASE/assistant_axis_outputs/llama-3.1-8b/vectors_q50"

cd "$REPO_DIR"

test -d "$ACT_DIR"
test -d "$SCO_DIR"

mkdir -p "$VEC_DIR"

uv sync

uv run pipeline/4_vectors.py \
  --activations_dir "$ACT_DIR" \
  --scores_dir "$SCO_DIR" \
  --output_dir "$VEC_DIR" \
  --min_count 30
