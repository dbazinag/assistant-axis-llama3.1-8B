#!/usr/bin/env bash
set -euo pipefail

cd /dlabscratch1/bazina/assistant-axis-llama3.1-8B
/dlabscratch1/bazina/.local/bin/uv run python full_trait_tools/run_baselines_harmbench_ppl_smoothllm.py \
  --test \
  --test_rows 30 \
  --n_seeds 3 \
  --smoothllm_num_copies 3 \
  --smoothllm_batch_size 3 \
  --smoothllm_max_new_tokens 80 \
  --recompute_harmbench \
  --recompute_ppl \
  --recompute_smoothllm \
  --output_dir full_trait_output/baselines_harmbench_ppl_smoothllm_test
