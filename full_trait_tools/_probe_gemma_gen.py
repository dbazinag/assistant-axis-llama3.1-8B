"""Diagnostic: time a single Gemma-4-31B greedy generation on GPTFuzz pair 0's
prompt to find out why the collector hangs. Prints input/output token counts,
wall time, tokens/sec, and whether max_new_tokens is honored. Mirrors the
collector's exact load + generate recipe."""
import json, time, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from chunked_sdpa import chunked_sdpa_scope

MODEL = sys.argv[1]
TC = "/dlabscratch1/bazina/HarmBench/results/HarmBench_GPTFuzz/gemma4_31b/test_cases/test_cases_individual_behaviors/5g_covid19_link_argument/test_cases.json"

d = json.load(open(TC))
prompt = next(iter(d.values()))[0]
if isinstance(prompt, list):
    prompt = prompt[-1]
print(f"[probe] prompt chars={len(prompt)}", flush=True)

print(f"[probe] loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    device_map="cuda:0", trust_remote_code=True)
model.eval()
print("[probe] loaded. gen_config:", {
    "max_new_tokens": getattr(model.generation_config, "max_new_tokens", None),
    "max_length": getattr(model.generation_config, "max_length", None),
    "do_sample": getattr(model.generation_config, "do_sample", None),
    "eos_token_id": getattr(model.generation_config, "eos_token_id", None),
    "use_cache": getattr(model.generation_config, "use_cache", None),
}, flush=True)
print("[probe] tokenizer eos:", tok.eos_token, tok.eos_token_id, "pad:", tok.pad_token, flush=True)

conv = [{"role": "user", "content": prompt}]
text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
input_ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
print(f"[probe] input tokens={input_ids.shape[1]}", flush=True)

for mnt in (64, 512):
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(), chunked_sdpa_scope():
        out = model.generate(input_ids=input_ids, max_new_tokens=mnt,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); dt = time.time() - t0
    n_new = out.shape[1] - input_ids.shape[1]
    print(f"[probe] max_new_tokens={mnt}: produced {n_new} tokens in {dt:.1f}s "
          f"({n_new/dt:.2f} tok/s)  hit_cap={n_new>=mnt}", flush=True)
    print(f"[probe]   response head: {tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)[:300]!r}", flush=True)
