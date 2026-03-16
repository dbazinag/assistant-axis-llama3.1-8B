#!/usr/bin/env bash
set -euo pipefail

cd /mnt/dlab/scratch/dlabscratch1/$USER/assistant-axis-llama3.1-8B

mkdir -p full_trait_output/traits40_generation

uv run trait_pipeline/1_generate_traits40.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --traits_dir data/traits/instructions \
  --output_root full_trait_output/traits40_generation