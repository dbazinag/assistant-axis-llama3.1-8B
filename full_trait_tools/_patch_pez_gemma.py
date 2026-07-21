#!/usr/bin/env python3
"""Append Gemma-4-31B experiment blocks to HarmBench PEZ_config.yaml (idempotent).

PEZ is a pure in-process gradient attack (no target server, no attacker model) —
it just needs a target_model the harness can load with HF + gradients. We load
Gemma-4 with attn_implementation=eager (sidesteps the head_dim=512 SDPA backend
limit; PEZ sequences are short so eager is cheap and autograd-clean) and let
get_template fall back to Gemma's apply_chat_template. Backs up the config once.
"""
import shutil
from pathlib import Path

CONFIG = Path("/dlabscratch1/bazina/HarmBench/configs/method_configs/PEZ_config.yaml")
SNAP = ("/mnt/dlabscratch1/bazina/.cache/huggingface/hub/"
        "models--google--gemma-4-31B-it/snapshots/fb9ae262347c3945692f09a612f8bb189def854f")

BLOCK = f'''
gemma4_31b:
  target_model:
    model_name_or_path: {SNAP}
    dtype: bfloat16
    trust_remote_code: True
    attn_implementation: eager
    num_gpus: 1

gemma4_31b_smoke:
  num_steps: 20
  num_test_cases_per_behavior: 1
  test_cases_batch_size: 1
  target_model:
    model_name_or_path: {SNAP}
    dtype: bfloat16
    trust_remote_code: True
    attn_implementation: eager
    num_gpus: 1
'''


def main():
    src = CONFIG.read_text()
    if "gemma4_31b:" in src:
        print("PEZ_config.yaml: gemma4_31b already present")
        return
    b = CONFIG.with_suffix(CONFIG.suffix + ".bak_gemma")
    if not b.exists():
        shutil.copy2(CONFIG, b)
        print(f"backup -> {b}")
    CONFIG.write_text(src.rstrip() + "\n" + BLOCK)
    print("PEZ_config.yaml: appended gemma4_31b + gemma4_31b_smoke")


if __name__ == "__main__":
    main()
