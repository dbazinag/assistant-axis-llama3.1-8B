#!/usr/bin/env python3
"""Wire HarmBench PAIR to use the Gemma server as TARGET (idempotent).

PAIR already routes any target whose model_name contains "gpt-" through its GPT
(OpenAI client) class. We make that class honour a GEMMA_TARGET_URL env var
(base_url) so the target hits gemma_openai_server.py, and drop the stale
`request_timeout` kwarg (invalid in openai>=1.0, breaks the create call). The
attack + judge models stay the default Mixtral-8x7B (vLLM), matching the OLMo run.
All changes are env-guarded / backups taken, so the vLLM/GPT paths are unchanged
when GEMMA_TARGET_URL is unset.
"""
import shutil
from pathlib import Path

HB = Path("/dlabscratch1/bazina/HarmBench")
LM = HB / "baselines/pair/language_models.py"
CONFIG = HB / "configs/method_configs/PAIR_config.yaml"
SNAP_NAME = "gpt-gemma-4-31b-it"   # must contain "gpt-" so load_indiv_model -> GPT
TARGETS = "/mnt/dlabscratch1/bazina/HarmBench/data/optimizer_targets/harmbench_targets_text.json"

OLD_INIT = (
    "    def __init__(self, model_name, token):\n"
    "        self.model_name = model_name\n"
    "        self.client = OpenAI(api_key=token, timeout=self.API_TIMEOUT)\n"
)
NEW_INIT = (
    "    def __init__(self, model_name, token):\n"
    "        self.model_name = model_name\n"
    "        _gemma_url = os.environ.get(\"GEMMA_TARGET_URL\")\n"
    "        if _gemma_url:\n"
    "            self.client = OpenAI(api_key=(token or \"sk-local\"), base_url=_gemma_url, timeout=600)\n"
    "        else:\n"
    "            self.client = OpenAI(api_key=token, timeout=self.API_TIMEOUT)\n"
)

OLD_CREATE = (
    "                    top_p=top_p,\n"
    "                    request_timeout=self.API_TIMEOUT,\n"
    "                )\n"
)
NEW_CREATE = (
    "                    top_p=top_p,\n"
    "                )\n"
)

CONFIG_BLOCK = f'''
gemma4_31b:
  targets_path: {TARGETS}
  target_model:
    model_name_or_path: {SNAP_NAME}
    token: sk-local

gemma4_31b_smoke:
  n_streams: 5
  steps: 2
  targets_path: {TARGETS}
  target_model:
    model_name_or_path: {SNAP_NAME}
    token: sk-local
'''


def backup(p: Path):
    b = p.with_suffix(p.suffix + ".bak_gemma")
    if not b.exists():
        shutil.copy2(p, b)
        print(f"backup -> {b}")


def patch_lm():
    src = LM.read_text()
    if "GEMMA_TARGET_URL" in src:
        print("language_models.py: already patched")
        return
    assert OLD_INIT in src, "GPT.__init__ block not found"
    assert OLD_CREATE in src, "create() request_timeout block not found"
    backup(LM)
    src = src.replace(OLD_INIT, NEW_INIT, 1).replace(OLD_CREATE, NEW_CREATE, 1)
    LM.write_text(src)
    print("language_models.py: GPT target -> server base_url + dropped request_timeout")


def patch_config():
    src = CONFIG.read_text()
    if "gemma4_31b:" in src:
        print("PAIR_config.yaml: gemma4_31b already present")
        return
    backup(CONFIG)
    CONFIG.write_text(src.rstrip() + "\n" + CONFIG_BLOCK)
    print("PAIR_config.yaml: appended gemma4_31b + gemma4_31b_smoke")


def main():
    patch_lm()
    patch_config()
    print("\n--- verify ---")
    print("GEMMA_TARGET_URL guard in lm:", "GEMMA_TARGET_URL" in LM.read_text())
    print("request_timeout removed:", "request_timeout" not in LM.read_text())
    print("gemma4_31b in config:", "gemma4_31b:" in CONFIG.read_text())


if __name__ == "__main__":
    main()
