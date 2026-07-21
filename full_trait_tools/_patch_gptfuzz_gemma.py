#!/usr/bin/env python3
"""Apply env-var-guarded Gemma-target support to HarmBench GPTFuzz (idempotent).

All changes are gated on GEMMA_SERVER_URL so the OLMo/vLLM path is unchanged when
the env var is unset. Backs up each file to <file>.bak_gemma before patching.
"""
import shutil
from pathlib import Path

HB = Path("/dlabscratch1/bazina/HarmBench")
LLM = HB / "baselines/gptfuzz/gptfuzzer/llm/llm.py"
GPTFUZZ = HB / "baselines/gptfuzz/gptfuzz.py"
CONFIG = HB / "configs/method_configs/GPTFuzz_config.yaml"

LOCAL_SERVER_CLASS = '''

class LocalServerLLM(LLM):
    """Target backed by an OpenAI-compatible HTTP server (gemma_openai_server.py).

    Sends raw jailbreak text as a single user message; the server applies the
    model's own chat template. Drop-in for LocalVLLM (generate returns a list).
    """
    def __init__(self, base_url, model_name="gemma", api_key="sk-local", system_message=None):
        super().__init__()
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_path = model_name
        self.system_message = system_message

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": prompt})
        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path, messages=messages,
                    temperature=temperature, max_tokens=max_tokens, n=n)
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(f"Local Gemma server call failed: {e}. Retry {_+1}/{max_trials}...")
                time.sleep(failure_sleep_time)
        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, p, temperature, max_tokens, n, max_trials, failure_sleep_time): p
                       for p in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results
'''

GEMMA_CONFIG = '''
gemma4_31b:
  target_model:
    model_name_or_path: /mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/fb9ae262347c3945692f09a612f8bb189def854f
    num_gpus: 1
  targets_path: data/optimizer_targets/harmbench_targets_text.json
  num_gpus: 1
'''


def backup(p: Path):
    b = p.with_suffix(p.suffix + ".bak_gemma")
    if not b.exists():
        shutil.copy2(p, b)
        print(f"backup -> {b}")


def patch_llm():
    src = LLM.read_text()
    if "class LocalServerLLM" in src:
        print("llm.py: LocalServerLLM already present")
        return
    backup(LLM)
    LLM.write_text(src.rstrip() + "\n" + LOCAL_SERVER_CLASS)
    print("llm.py: appended LocalServerLLM")


def patch_gptfuzz():
    src = GPTFUZZ.read_text()
    if "GEMMA_SERVER_URL" in src:
        print("gptfuzz.py: already patched")
        return
    backup(GPTFUZZ)

    # import os
    src = src.replace("import json\n", "import json\nimport os\n", 1)
    # import LocalServerLLM
    src = src.replace(
        "from .gptfuzzer.llm import OpenAILLM, LocalVLLM, LocalLLM, PaLM2LLM, ClaudeLLM",
        "from .gptfuzzer.llm import OpenAILLM, LocalVLLM, LocalLLM, PaLM2LLM, ClaudeLLM, LocalServerLLM",
    )
    # guard get_template
    old_tpl = ("        template = get_template(**target_model)['prompt']\n"
               "        self.template = template\n"
               "        self.before_tc, self.after_tc = template.split(\"{instruction}\")\n")
    new_tpl = ("        _gemma_url = os.environ.get(\"GEMMA_SERVER_URL\")\n"
               "        if _gemma_url:\n"
               "            self.template = \"{instruction}\"\n"
               "            self.before_tc, self.after_tc = \"\", \"\"\n"
               "        else:\n"
               "            template = get_template(**target_model)['prompt']\n"
               "            self.template = template\n"
               "            self.before_tc, self.after_tc = template.split(\"{instruction}\")\n")
    assert old_tpl in src, "template block not found"
    src = src.replace(old_tpl, new_tpl, 1)
    # mutator key from env on server path
    old_mut = "        self.openai_model = OpenAILLM(mutate_model_path, openai_key)\n"
    new_mut = ("        _mutator_key = os.environ.get(\"OPENAI_API_KEY\", openai_key) if _gemma_url else openai_key\n"
               "        self.openai_model = OpenAILLM(mutate_model_path, _mutator_key)\n")
    assert old_mut in src, "mutator line not found"
    src = src.replace(old_mut, new_mut, 1)
    # guard target
    old_tgt = ("        target_model['num_gpus'] = num_gpus\n"
               "        self.target_model = LocalVLLM(target_model)\n")
    new_tgt = ("        target_model['num_gpus'] = num_gpus\n"
               "        if _gemma_url:\n"
               "            self.target_model = LocalServerLLM(base_url=_gemma_url,\n"
               "                                               model_name=target_model.get('model_name_or_path', 'gemma'))\n"
               "        else:\n"
               "            self.target_model = LocalVLLM(target_model)\n")
    assert old_tgt in src, "target line not found"
    src = src.replace(old_tgt, new_tgt, 1)

    GPTFUZZ.write_text(src)
    print("gptfuzz.py: patched (os import, LocalServerLLM import, 3 guarded blocks)")


def patch_config():
    src = CONFIG.read_text()
    if "gemma4_31b:" in src:
        print("GPTFuzz_config.yaml: gemma4_31b already present")
        return
    backup(CONFIG)
    CONFIG.write_text(src.rstrip() + "\n" + GEMMA_CONFIG)
    print("GPTFuzz_config.yaml: appended gemma4_31b experiment")


def main():
    patch_llm()
    patch_gptfuzz()
    patch_config()
    # verify
    print("\n--- verify ---")
    print("LocalServerLLM in llm.py:", "class LocalServerLLM" in LLM.read_text())
    g = GPTFUZZ.read_text()
    print("gptfuzz.py guards:", g.count("GEMMA_SERVER_URL"), "import os:", "\nimport os\n" in g)
    print("gemma4_31b in config:", "gemma4_31b:" in CONFIG.read_text())


if __name__ == "__main__":
    main()
