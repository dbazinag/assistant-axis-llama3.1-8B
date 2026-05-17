#!/usr/bin/env bash
set -euo pipefail

cd /dlabscratch1/bazina/assistant-axis-llama3.1-8B
/dlabscratch1/bazina/.local/bin/uv run python full_trait_tools/run_baselines_fjd_paper.py \
  --test \
  --test_rows 200 \
  --n_seeds 3 \
  --recompute \
  --output_dir full_trait_output/baselines_fjd_paper_test
