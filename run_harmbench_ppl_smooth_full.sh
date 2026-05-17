#!/usr/bin/env bash
set -euo pipefail

cd /dlabscratch1/bazina/assistant-axis-llama3.1-8B
/dlabscratch1/bazina/.local/bin/uv run python full_trait_tools/run_baselines_harmbench_ppl_smoothllm.py \
  --n_seeds 50 \
  --smoothllm_num_copies 10 \
  --smoothllm_batch_size 10 \
  --smoothllm_max_new_tokens 100 \
  --output_dir full_trait_output/baselines_harmbench_ppl_smoothllm
