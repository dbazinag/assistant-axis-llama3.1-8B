#!/usr/bin/env bash
set -euo pipefail

cd /dlabscratch1/bazina/assistant-axis-llama3.1-8B
/dlabscratch1/bazina/.local/bin/uv run python full_trait_tools/run_baselines_jbshield_fjd.py \
  --test \
  --n_seeds 3 \
  --recompute_fjd \
  --output_dir full_trait_output/baselines_jbshield_fjd_test
