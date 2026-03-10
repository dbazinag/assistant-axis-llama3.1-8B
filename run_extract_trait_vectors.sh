#!/usr/bin/env bash
# run_extract_trait_vectors.sh
# RunAI wrapper for extract_trait_vectors.py
# Copy to: /mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/
# Submit:  rs extract-trait-vectors --gpu 1.0 --memory 40G --node-type G10 -- \
#            /dlabscratch1/bazina/assistant-axis-llama3.1-8B/run_extract_trait_vectors.sh

set -euo pipefail
: "${USER:=$(whoami)}"

if   [ -d /dlabscratch1/"$USER" ];                  then BASE=/dlabscratch1/"$USER"
elif [ -d /mnt/dlabscratch1/"$USER" ];              then BASE=/mnt/dlabscratch1/"$USER"
elif [ -d /mnt/dlab/scratch/dlabscratch1/"$USER" ]; then BASE=/mnt/dlab/scratch/dlabscratch1/"$USER"
else echo "ERROR: scratch not found"; exit 1; fi

REPO="assistant-axis-llama3.1-8B"
OUTPUTS="assistant_axis_outputs/llama-3.1-8b"
OPENAI_API_KEY="YOUR_KEY_HERE"

cd "$BASE/$REPO"

export HF_HOME=$BASE/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=$BASE/.cache
export TORCH_HOME=$BASE/.cache/torch
export TMPDIR=$BASE/.tmp
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR"

uv sync

uv run tools/extract_trait_vectors.py \
    --trait_dir      "$BASE/$REPO/data/extraction_questions/traits" \
    --model_id       meta-llama/Llama-3.1-8B-Instruct \
    --out_dir        "$BASE/$OUTPUTS/vectors_q50" \
    --layer          16 \
    --openai_api_key "$OPENAI_API_KEY" \
    --judge_model    gpt-4o-mini
