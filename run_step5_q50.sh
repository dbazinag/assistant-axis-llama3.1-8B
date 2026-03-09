#!/usr/bin/env bash
set -euo pipefail
: "${USER:=$(whoami)}"

# ---- detect scratch mount ----
if [ -d /dlabscratch1/"$USER" ]; then BASE=/dlabscratch1/"$USER";
elif [ -d /mnt/dlabscratch1/"$USER" ]; then BASE=/mnt/dlabscratch1/"$USER";
elif [ -d /mnt/dlab/scratch/dlabscratch1/"$USER" ]; then BASE=/mnt/dlab/scratch/dlabscratch1/"$USER";
else echo "ERROR: could not find scratch mount for $USER"; exit 1; fi

REPO_DIR="$BASE/assistant-axis-llama3.1-8B"
VEC_DIR="$BASE/assistant_axis_outputs/llama-3.1-8b/vectors_q50"
OUT_AXIS="$BASE/assistant_axis_outputs/llama-3.1-8b/axis_q50.pt"

cd "$REPO_DIR"
test -d "$VEC_DIR"

uv sync

uv run pipeline/5_axis.py \
  --vectors_dir "$VEC_DIR" \
  --output "$OUT_AXIS"

echo "DONE. Axis saved to: $OUT_AXIS"
