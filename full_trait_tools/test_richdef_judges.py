#!/usr/bin/env python3
"""
Option A: give BOTH judges a richer NEUTRAL trait definition and see if agreement rises.

The richer definition = the trait's `instruction` pos-descriptors (what a response that
expresses the trait does), inserted before the Q/A. It does NOT reveal this item's
induction polarity. Both judges (GPT-4.1-mini, DeepSeek-V4-Flash) are re-scored with it.

To isolate the definition effect from GPT's temperature-1 resampling noise, we also
re-grade GPT *blind* (original prompt) as a control. DeepSeek is temp 0 (deterministic),
so its blind scores in paired_scores.jsonl are reused.

Grades the 2,000-item tune split (seed 0). Resumable.
"""
import asyncio, json, os, random, sys
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from compare_judge_deepseek import parse_persona_score, assistant_text
from assistant_axis.judge import RateLimiter

load_dotenv()
ROOT = Path("full_trait_output")
JC = ROOT / "judge_comparison"
TI = Path("data/traits/instructions")
OUT = JC / "richdef"
ORIG_BLOCK_MARKER = "\n\nPrompt:"


def enriched_ep(trait: str) -> str:
    tf = json.load(open(TI / f"{trait}.json"))
    descr = "\n".join(f"- {d['pos']}" for d in tf["instruction"])
    block = ("\n\nFor reference, a response that strongly expresses this trait typically "
             f"does the following:\n{descr}")
    return tf["eval_prompt"].replace(ORIG_BLOCK_MARKER, block + ORIG_BLOCK_MARKER, 1)


def orig_ep(trait: str) -> str:
    return json.load(open(TI / f"{trait}.json"))["eval_prompt"]


async def call_one(client, prompt, model, rl, *, max_param, max_val, temperature, extra=None, retries=6):
    for a in range(retries):
        await rl.acquire()
        try:
            kw = {"model": model, "messages": [{"role": "user", "content": prompt}],
                  max_param: max_val, "temperature": temperature}
            if extra:
                kw["extra_body"] = extra
            r = await client.chat.completions.create(**kw)
            return r.choices[0].message.content if (r.choices and r.choices[0].message.content) else None
        except Exception:
            if a == retries - 1:
                return None
            await asyncio.sleep(2 ** a)


async def grade(client, items, model, ep_fn, cache_path, *, max_param, max_val, temperature, extra, batch, rps):
    done = {}
    if cache_path.exists():
        for r in (json.loads(l) for l in open(cache_path)):
            done[r["trait"] + "|" + r["label"]] = r
    todo = [it for it in items if it["trait"] + "|" + it["label"] not in done]
    if not todo:
        return done
    ep_cache, resp_cache, prompts = {}, {}, []
    for it in todo:
        t = it["trait"]
        if t not in ep_cache:
            ep_cache[t] = ep_fn(t)
            resp_cache[t] = {json.loads(x)["label"]: json.loads(x)
                             for x in open(ROOT / "traits40_generation" / "responses" / f"{t}.jsonl")}
        r = resp_cache[t][it["label"]]
        prompts.append(ep_cache[t].format(question=r["question"], answer=assistant_text(r)))
    rl = RateLimiter(rps)
    with open(cache_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch):
            chunk, pchunk = todo[i:i + batch], prompts[i:i + batch]
            raws = await asyncio.gather(*[call_one(client, p, model, rl, max_param=max_param, max_val=max_val,
                                                   temperature=temperature, extra=extra) for p in pchunk])
            for it, raw in zip(chunk, raws):
                rec = {"trait": it["trait"], "label": it["label"], "score": parse_persona_score(raw), "raw": raw}
                done[it["trait"] + "|" + it["label"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{model} {cache_path.stem}] {min(i + batch, len(todo))}/{len(todo)}", flush=True)
    return done


def b(x):
    return 1 if x >= 50 else 0


def report(name, g_scores, d_scores, items):
    rows = [(it, g_scores.get(it["trait"] + "|" + it["label"]), d_scores.get(it["trait"] + "|" + it["label"]))
            for it in items]
    rows = [(it, g, d) for it, g, d in rows if g is not None and d is not None]
    G = np.array([b(g) for _, g, d in rows]); D = np.array([b(d) for _, g, d in rows])
    agree = 100 * np.mean(G == D)
    kappa = cohen_kappa_score(G, D)
    print(f"{name:32s} n={len(rows)}  agree={agree:.1f}%  kappa={kappa:.3f}")
    for pol in ["positive", "negative"]:
        sub = [(it, g, d) for it, g, d in rows if it["label"].startswith(pol)]
        if sub:
            gg = np.array([b(g) for _, g, d in sub]); dd = np.array([b(d) for _, g, d in sub])
            print(f"    {pol:9s} agree={100*np.mean(gg==dd):.1f}%  (n={len(sub)})")
    return rows


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(l) for l in open(JC / "persona" / "paired_scores.jsonl")]
    random.Random(0).shuffle(pairs)
    tune = pairs[:2000]
    # blind reference scores (from the main run): GPT temp1 blind, DeepSeek temp0 blind
    blind_gpt = {p["trait"] + "|" + p["label"]: p["gpt"] for p in tune}
    blind_ds = {p["trait"] + "|" + p["label"]: p["deepseek"] for p in tune}

    gpt_client = AsyncOpenAI()  # OPENAI_API_KEY
    ds_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    # 1) GPT blind control (temp1 resampling control)
    gpt_blind = await grade(gpt_client, tune, "gpt-4.1-mini", orig_ep, OUT / "gpt_blind.jsonl",
                            max_param="max_completion_tokens", max_val=10, temperature=1, extra=None, batch=50, rps=50)
    gpt_blind = {k: v["score"] for k, v in gpt_blind.items()}
    # 2) GPT enriched
    gpt_rich = await grade(gpt_client, tune, "gpt-4.1-mini", enriched_ep, OUT / "gpt_rich.jsonl",
                           max_param="max_completion_tokens", max_val=10, temperature=1, extra=None, batch=50, rps=50)
    gpt_rich = {k: v["score"] for k, v in gpt_rich.items()}
    # 3) DeepSeek enriched
    ds_rich = await grade(ds_client, tune, "deepseek/deepseek-v4-flash", enriched_ep, OUT / "ds_rich.jsonl",
                          max_param="max_tokens", max_val=256, temperature=0,
                          extra={"reasoning": {"enabled": False}}, batch=100, rps=100)
    ds_rich = {k: v["score"] for k, v in ds_rich.items()}

    print("\n========== OPTION A: richer trait definition ==========")
    report("BASELINE blind (from main run)", blind_gpt, blind_ds, tune)
    report("CONTROL  GPT-blind-fresh vs DS-blind", gpt_blind, blind_ds, tune)
    report("ENRICHED both (richer def)", gpt_rich, ds_rich, tune)
    # how far each judge moved
    def shift(new, old):
        v = [abs(new[k] - old[k]) for k in new if k in old and new[k] is not None and old[k] is not None]
        return np.mean(v) if v else 0
    print(f"\nmean |score change| vs blind:  GPT={shift(gpt_rich, blind_gpt):.1f}   DeepSeek={shift(ds_rich, blind_ds):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
