#!/usr/bin/env bash
set -euo pipefail

cd /dlabscratch1/bazina/assistant-axis-llama3.1-8B
uv run python full_trait_tools/run_baselines_gradsafe.py
