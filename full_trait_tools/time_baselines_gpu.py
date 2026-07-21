"""Time the DETECTION latency of the neural baselines (no original-response generation).

For each method we time its OWN per-sample detection function (imported from the
baseline script, not reimplemented), CUDA-synchronized, batch=1 (per-request) plus a
batched-core throughput pass. The cost being measured is the extra neural compute the
method runs ON TOP of the production forward pass — which for these methods is a whole
separate model / forward+backward / generation, none of which production already did.

Methods (SmoothLLM excluded by request):
  ppl         : 1 forward of the target model (prompt perplexity)
  llama_guard : 1 forward of a separate 8B guard, read safe/unsafe logit
  wildguard   : generate() of a separate ~7B guard
  self_exam   : 2 generate() of the target model (DIRECT + COT meta-questions)
  fjd         : generate() of the target model (first-token confidence)
  gradsafe    : forward + backward of the target model (per-param gradient cosine)
"""
import argparse, sys, time, statistics
from pathlib import Path
import torch

sys.path.insert(0, "full_trait_tools")
def _imp(mod):
    try:
        return __import__(mod)
    except Exception as e:
        print(f"  [skip] import {mod} failed: {e}", flush=True)
        return None
LG  = _imp("run_baselines_llama_guard")
GS  = _imp("run_baselines_gradsafe")
PPL = _imp("run_baselines_harmbench_ppl_smoothllm")
WG  = _imp("run_baselines_wildguard")
SE  = _imp("run_baselines_self_exam")
FJD = _imp("run_baselines_fjd_paper")

OLMO = "allenai/OLMo-3-7B-Instruct"
HUMAN_CLS = "full_trait_output/harmbench_activations_olmo3/classified_responses.jsonl"


def load_pairs(n):
    import json
    pairs = []
    with open(HUMAN_CLS) as f:
        for line in f:
            r = json.loads(line)
            p = (r.get("behavior_text") or "").strip()
            resp = (r.get("response_text") or r.get("response") or "").strip()
            if p and resp:
                pairs.append((p, resp))
            if len(pairs) >= n:
                break
    return pairs


def stats_ms(times):
    return statistics.mean(times), (statistics.pstdev(times) if len(times) > 1 else 0.0)


def time_batch1(fn, items, warmup=3):
    for it in items[:warmup]:
        fn(it)
    torch.cuda.synchronize()
    ts = []
    for it in items:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(it)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return stats_ms(ts)


def time_batched(core_fn, items, bs, reps=3):
    """core_fn takes a list of (prompt,response) and runs the method's core op once."""
    batch = items[:bs]
    core_fn(batch)  # warm
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        core_fn(batch)
    torch.cuda.synchronize()
    per_sample = (time.perf_counter() - t0) / reps / bs * 1e3
    return per_sample


def report(name, n, b1, batched, tok_info):
    m, s = b1
    bs_str = f"{batched:8.2f}" if batched is not None else "   n/a  "
    print(f"  {name:12s} | batch1 {m:9.2f} ± {s:7.2f} ms | batched/s {bs_str} ms | {tok_info}", flush=True)


def free(model):
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="ppl,llama_guard,wildguard,self_exam,fjd,gradsafe")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    pairs = load_pairs(args.n)
    dev = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)}  N={len(pairs)}  bs={args.bs}", flush=True)
    print(f"{'method':14s}   per-request (batch=1)        batched throughput", flush=True)

    olmo_methods = {"ppl", "self_exam", "verbalized_nocot", "verbalized_cot", "fjd", "gradsafe"}
    need_olmo = any(m in olmo_methods for m in methods) and PPL is not None
    model = tok = None
    if need_olmo:
        print(f"\nLoading target model {OLMO} ...", flush=True)
        model, tok, _ = PPL.load_causal_lm(OLMO, "cuda", olmo3=True)
        if "gradsafe" in methods:
            for p in model.parameters():
                p.requires_grad_(True)

    def toklen(texts):
        return sum(len(tok(t, add_special_tokens=False).input_ids) for t in texts) / len(texts)

    if "ppl" in methods and PPL:
        fn = lambda pr: PPL.prompt_perplexity(model, tok, pr[0], dev, 2048)
        def core(batch):
            enc = tok([p for p, _ in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=2048, add_special_tokens=False).to(dev)
            with torch.no_grad():
                model(**enc)
        report("ppl", len(pairs), time_batch1(fn, pairs),
               time_batched(core, pairs, args.bs),
               f"prompt_tok≈{toklen([p for p,_ in pairs[:args.bs]]):.0f}")

    if "verbalized_nocot" in methods and SE:
        fn = lambda pr: SE.generate_meta_response(model, tok, pr[0], SE.DIRECT_PROMPT, dev)
        report("verb_nocot", len(pairs), time_batch1(fn, pairs), None,
               f"1 gen ≤{SE.MAX_NEW_TOKENS} tok (DIRECT)")

    if "verbalized_cot" in methods and SE:
        fn = lambda pr: SE.generate_meta_response(model, tok, pr[0], SE.COT_PROMPT, dev)
        report("verb_cot", len(pairs), time_batch1(fn, pairs), None,
               f"1 gen ≤{SE.MAX_NEW_TOKENS} tok (COT)")

    if "fjd" in methods and FJD:
        fjd_args = argparse.Namespace(max_new_tokens=32, temperature=FJD.FJD_TEMPERATURE,
                                      max_length=2048)
        fn = lambda pr: FJD.fjd_first_token_confidence(model, tok, pr[0], dev, fjd_args)
        report("fjd", len(pairs), time_batch1(fn, pairs), None,
               f"gen ≤{fjd_args.max_new_tokens} tok, T={FJD.FJD_TEMPERATURE}")

    if "gradsafe" in methods and GS:
        fn = lambda pr: GS.compute_gradient(model, tok, pr[0], dev)
        report("gradsafe", len(pairs), time_batch1(fn, pairs), None,
               "fwd+bwd, target='Sure'")

    if need_olmo:
        free(model); model = tok = None

    if "llama_guard" in methods and LG:
        print(f"\nLoading {LG.GUARD_MODEL} ...", flush=True)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        gtok = AutoTokenizer.from_pretrained(LG.GUARD_MODEL)
        if gtok.pad_token is None:
            gtok.pad_token = gtok.eos_token
        gmodel = AutoModelForCausalLM.from_pretrained(LG.GUARD_MODEL, dtype=torch.bfloat16,
                                                      device_map={"": dev})
        gmodel.eval()
        safe_id, unsafe_id = LG.get_token_ids(gtok)
        # input-only (prompt, no response) — the variant in the paper table
        fn_in = lambda pr: LG.guard_score(gmodel, gtok, pr[0], dev, safe_id, unsafe_id, response=None)
        report("guard_input", len(pairs), time_batch1(fn_in, pairs), None, "fwd, prompt only")
        # prompt+response (kept for reference)
        fn = lambda pr: LG.guard_score(gmodel, gtok, pr[0], dev, safe_id, unsafe_id, response=pr[1])
        def core(batch):
            convs = [gtok.apply_chat_template(
                [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
                tokenize=False, add_generation_prompt=False) for p, r in batch]
            enc = gtok(convs, return_tensors="pt", padding=True, truncation=True,
                       max_length=2048, add_special_tokens=False).to(dev)
            with torch.no_grad():
                gmodel(**enc)
        report("guard_promptresp", len(pairs), time_batch1(fn, pairs),
               time_batched(core, pairs, args.bs), "fwd, prompt+response")
        free(gmodel)

    if "wildguard" in methods and WG:
        print(f"\nLoading {WG.WILDGUARD_MODEL} ...", flush=True)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        wtok = AutoTokenizer.from_pretrained(WG.WILDGUARD_MODEL, use_fast=False)
        if wtok.pad_token is None:
            wtok.pad_token = wtok.eos_token
        wmodel = AutoModelForCausalLM.from_pretrained(WG.WILDGUARD_MODEL, dtype=torch.bfloat16,
                                                      device_map={"": dev})
        wmodel.eval()
        fn = lambda pr: WG.classify_one(wmodel, wtok, dev, pr[0], pr[1], 40)
        report("wildguard", len(pairs), time_batch1(fn, pairs), None, "gen ≤40 tok, prompt+response")
        free(wmodel)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
