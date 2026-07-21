#!/usr/bin/env python3
"""Re-score GPT-4.1-mini at temperature 0 (single sample, original prompt) and compare to
DeepSeek's existing temp-0 scores. Both deterministic single samples — no logprobs needed.
Compares to the temp-1-GPT baseline. Grades the 2000-item tune split. Resumable."""
import asyncio, json, os, random, sys
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent))
from compare_judge_deepseek import parse_persona_score, assistant_text
from assistant_axis.judge import RateLimiter

load_dotenv()
ROOT = Path("full_trait_output"); JC = ROOT / "judge_comparison"; TI = Path("data/traits/instructions")
CACHE = JC / "logprob" / "gpt_temp0_single.jsonl"


async def one(client, prompt, rl, retries=6):
    for a in range(retries):
        await rl.acquire()
        try:
            r = await client.chat.completions.create(model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=10, temperature=0, seed=0)
            return r.choices[0].message.content if r.choices else None
        except Exception:
            if a == retries - 1:
                return None
            await asyncio.sleep(2 ** a)


async def main():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(l) for l in open(JC / "persona" / "paired_scores.jsonl")]
    random.Random(0).shuffle(pairs)
    tune = pairs[:2000]
    done = {}
    if CACHE.exists():
        for r in (json.loads(l) for l in open(CACHE)):
            done[r["trait"] + "|" + r["label"]] = r
    todo = [it for it in tune if it["trait"] + "|" + it["label"] not in done]
    client = AsyncOpenAI(); rl = RateLimiter(50)
    resp_cache, ep_cache, prompts = {}, {}, []
    for it in todo:
        t = it["trait"]
        if t not in ep_cache:
            ep_cache[t] = json.load(open(TI / f"{t}.json"))["eval_prompt"]
            resp_cache[t] = {json.loads(x)["label"]: json.loads(x)
                             for x in open(ROOT / "traits40_generation" / "responses" / f"{t}.jsonl")}
        r = resp_cache[t][it["label"]]
        prompts.append(ep_cache[t].format(question=r["question"], answer=assistant_text(r)))
    with open(CACHE, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), 50):
            chunk, pchunk = todo[i:i+50], prompts[i:i+50]
            raws = await asyncio.gather(*[one(client, p, rl) for p in pchunk])
            for it, raw in zip(chunk, raws):
                rec = {"trait": it["trait"], "label": it["label"], "score": parse_persona_score(raw)}
                done[it["trait"]+"|"+it["label"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush(); print(f"  {min(i+50,len(todo))}/{len(todo)}", flush=True)

    def b(x): return 1 if x >= 50 else 0
    rows = [(it, done[it["trait"]+"|"+it["label"]]["score"]) for it in tune
            if done.get(it["trait"]+"|"+it["label"], {}).get("score") is not None]
    G0 = np.array([b(s) for _, s in rows]); D = np.array([b(it["deepseek"]) for it, _ in rows])
    G1 = np.array([b(it["gpt"]) for it, _ in rows])
    print("\n========== GPT temp0 single-sample vs DeepSeek temp0 ==========")
    print(f"BASELINE  GPT(temp1) vs DS(temp0): agree={100*np.mean(G1==D):.1f}%  kappa={cohen_kappa_score(G1,D):.3f}  (n={len(rows)})")
    print(f"NEW       GPT(temp0) vs DS(temp0): agree={100*np.mean(G0==D):.1f}%  kappa={cohen_kappa_score(G0,D):.3f}  (n={len(rows)})")
    for pol in ["positive", "negative"]:
        idx = [i for i, (it, _) in enumerate(rows) if it["label"].startswith(pol)]
        print(f"    {pol:9s} temp0 agree={100*np.mean(G0[idx]==D[idx]):.1f}%  (n={len(idx)})")
    cg0 = np.array([s for _, s in rows]); cd = np.array([it["deepseek"] for it, _ in rows])
    cg1 = np.array([it["gpt"] for it, _ in rows])
    print(f"continuous MAE  GPT(temp1)vsDS={np.mean(np.abs(cg1-cd)):.2f}   GPT(temp0)vsDS={np.mean(np.abs(cg0-cd)):.2f}")
    # how much did GPT move temp1 -> temp0
    print(f"mean|GPT temp0 - GPT temp1| = {np.mean(np.abs(cg0-cg1)):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
