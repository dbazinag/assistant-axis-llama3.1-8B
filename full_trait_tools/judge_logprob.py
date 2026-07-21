#!/usr/bin/env python3
"""
judge_logprob.py

Re-score persona trait expression with the FAITHFUL persona_vectors method:
temperature 0, max_tokens 1, logprobs/top_logprobs, and a logprob-weighted
EXPECTED VALUE over the 0-100 number tokens (not a single sampled integer).

  score = Σ(int_token * prob) / Σ(prob over valid 0..100 tokens)   # None if Σprob < 0.25

Both judges (gpt-4.1-mini via OpenAI, deepseek-v4-flash via OpenRouter) are
re-scored on the same items with the SAME per-trait eval_prompt. This produces
continuous scores (e.g. 47.3) that should be far more stable at the 50 boundary
than the single-sample integers used so far.

Modes:
  --probe N   : grade N items on each judge, print scores + whether logprobs came back
  (default)   : grade the 2000-item tune split with both judges, compare to single-sample baseline
"""
import argparse, asyncio, json, math, os, random, sys
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from compare_judge_deepseek import assistant_text
from assistant_axis.judge import RateLimiter

load_dotenv()
ROOT = Path("full_trait_output")
JC = ROOT / "judge_comparison"
TI = Path("data/traits/instructions")
OUT = JC / "logprob"


def aggregate_0_100(token_probs: dict):
    s = total = 0.0
    for tok, p in token_probs.items():
        try:
            i = int(tok.strip())
        except ValueError:
            continue
        if 0 <= i <= 100:
            s += i * p
            total += p
    return None if total < 0.25 else s / total


async def judge_one(client, prompt, model, rl, *, use_seed, extra=None, retries=6):
    for a in range(retries):
        await rl.acquire()
        try:
            kw = dict(model=model, messages=[{"role": "user", "content": prompt}],
                      max_tokens=1, temperature=0, logprobs=True, top_logprobs=20)
            if use_seed:
                kw["seed"] = 0
            if extra:
                kw["extra_body"] = extra
            r = await client.chat.completions.create(**kw)
            lp = r.choices[0].logprobs
            if not lp or not lp.content:
                return ("NO_LOGPROBS", None)
            tp = {}
            for t in lp.content[0].top_logprobs:
                tp[t.token] = tp.get(t.token, 0.0) + math.exp(t.logprob)
            return (lp.content[0].token, aggregate_0_100(tp))
        except Exception as e:
            if a == retries - 1:
                return (f"ERR:{type(e).__name__}:{str(e)[:80]}", None)
            await asyncio.sleep(2 ** a)


async def grade(client, items, model, cache_path, *, use_seed, extra, batch, rps):
    done = {}
    if cache_path.exists():
        for r in (json.loads(l) for l in open(cache_path)):
            done[r["trait"] + "|" + r["label"]] = r
    todo = [it for it in items if it["trait"] + "|" + it["label"] not in done]
    if not todo:
        return done
    resp_cache, ep_cache, prompts = {}, {}, []
    for it in todo:
        t = it["trait"]
        if t not in ep_cache:
            ep_cache[t] = json.load(open(TI / f"{t}.json"))["eval_prompt"]
            resp_cache[t] = {json.loads(x)["label"]: json.loads(x)
                             for x in open(ROOT / "traits40_generation" / "responses" / f"{t}.jsonl")}
        r = resp_cache[t][it["label"]]
        prompts.append(ep_cache[t].format(question=r["question"], answer=assistant_text(r)))
    rl = RateLimiter(rps)
    with open(cache_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch):
            chunk, pchunk = todo[i:i + batch], prompts[i:i + batch]
            res = await asyncio.gather(*[judge_one(client, p, model, rl, use_seed=use_seed, extra=extra)
                                         for p in pchunk])
            for it, (tok, score) in zip(chunk, res):
                rec = {"trait": it["trait"], "label": it["label"], "score": score, "first_tok": tok}
                done[it["trait"] + "|" + it["label"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{model} {cache_path.stem}] {min(i + batch, len(todo))}/{len(todo)}", flush=True)
    return done


def split(seed, n):
    pairs = [json.loads(l) for l in open(JC / "persona" / "paired_scores.jsonl")]
    random.Random(seed).shuffle(pairs)
    return pairs[:n], pairs


def clients():
    gpt = AsyncOpenAI()
    ds = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    return gpt, ds


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", type=int, default=0)
    p.add_argument("--tune_n", type=int, default=2000)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tune, allpairs = split(0, args.tune_n)
    gpt, ds = clients()
    DS_EXTRA = {"reasoning": {"enabled": False}}

    if args.probe:
        items = tune[:args.probe]
        rl = RateLimiter(20)
        print("=== GPT probe ===")
        for it in items[:args.probe]:
            t = it["trait"]
            ep = json.load(open(TI / f"{t}.json"))["eval_prompt"]
            r = {json.loads(x)["label"]: json.loads(x) for x in open(ROOT/"traits40_generation"/"responses"/f"{t}.jsonl")}[it["label"]]
            prompt = ep.format(question=r["question"], answer=assistant_text(r))
            tok, sc = await judge_one(gpt, prompt, "gpt-4.1-mini", rl, use_seed=True)
            tokd, scd = await judge_one(ds, prompt, "deepseek/deepseek-v4-flash", rl, use_seed=False, extra=DS_EXTRA)
            print(f"  {t}/{it['label']}: GPT tok={tok!r} score={sc}  ||  DS tok={tokd!r} score={scd}  (orig gpt={it['gpt']} ds={it['deepseek']})")
        return

    gpt_done = await grade(gpt, tune, "gpt-4.1-mini", OUT / "gpt.jsonl", use_seed=True, extra=None, batch=50, rps=50)
    ds_done = await grade(ds, tune, "deepseek/deepseek-v4-flash", OUT / "ds.jsonl", use_seed=False, extra=DS_EXTRA, batch=100, rps=100)

    # treat refusal/None as 0 (matches existing pipeline's REFUSAL->0)
    def sc(d, k):
        v = d.get(k, {}).get("score")
        return 0.0 if v is None else v
    def b(x):
        return 1 if x >= 50 else 0
    rows = [(it, sc(gpt_done, it["trait"]+"|"+it["label"]), sc(ds_done, it["trait"]+"|"+it["label"]),
             it["gpt"], it["deepseek"]) for it in tune
            if it["trait"]+"|"+it["label"] in gpt_done and it["trait"]+"|"+it["label"] in ds_done]
    G = np.array([b(g) for _, g, d, _, _ in rows]); D = np.array([b(d) for _, g, d, _, _ in rows])
    BG = np.array([b(g0) for _, _, _, g0, d0 in rows]); BD = np.array([b(d0) for _, _, _, g0, d0 in rows])
    print("\n========== FAITHFUL temp0 + logprob expected-value ==========")
    print(f"SINGLE-SAMPLE baseline (temp1 gpt / temp0 ds): agree={100*np.mean(BG==BD):.1f}%  kappa={cohen_kappa_score(BG,BD):.3f}  (n={len(rows)})")
    print(f"LOGPROB EV (temp0 both):                       agree={100*np.mean(G==D):.1f}%  kappa={cohen_kappa_score(G,D):.3f}  (n={len(rows)})")
    for pol in ["positive", "negative"]:
        idx = [i for i, (it, *_ ) in enumerate(rows) if it["label"].startswith(pol)]
        if idx:
            gg = G[idx]; dd = D[idx]
            print(f"    {pol:9s} logprob agree={100*np.mean(gg==dd):.1f}%  (n={len(idx)})")
    # continuous MAE between judges (now both are expected values)
    cg = np.array([g for _, g, d, _, _ in rows]); cd = np.array([d for _, g, d, _, _ in rows])
    print(f"continuous MAE between judges (logprob EV): {np.mean(np.abs(cg-cd)):.2f}  (0-100 scale)")
    json.dump({"n": len(rows), "logprob_agree": float(np.mean(G==D)),
               "baseline_agree": float(np.mean(BG==BD)),
               "logprob_kappa": float(cohen_kappa_score(G, D)),
               "continuous_mae": float(np.mean(np.abs(cg-cd)))}, open(OUT/"metrics.json", "w"), indent=2)


if __name__ == "__main__":
    asyncio.run(main())
