#!/usr/bin/env python3
"""Re-score DeepSeek (temp 0) with the trait's FIRST system prompt (instruction[0].pos)
added to the judge prompt as a concrete explanation of the trait. Compare to GPT temp-0
scores (unchanged) and to DeepSeek's original temp-0 scores. Tune split (2000). Resumable."""
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
CACHE = JC / "logprob" / "ds_sysprompt_hint.jsonl"
MARKER = "\n\nPrompt:"


def hinted_ep(trait):
    tf = json.load(open(TI / f"{trait}.json"))
    hint = tf["instruction"][0]["pos"]  # the first system prompt used to elicit this trait
    block = ("\n\nFor reference, a response that expresses this trait follows guidance like: "
             f"\"{hint}\"")
    return tf["eval_prompt"].replace(MARKER, block + MARKER, 1)


async def one(client, prompt, rl, retries=6):
    for a in range(retries):
        await rl.acquire()
        try:
            r = await client.chat.completions.create(model="deepseek/deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}], max_tokens=256, temperature=0,
                extra_body={"reasoning": {"enabled": False}})
            return r.choices[0].message.content if r.choices else None
        except Exception:
            if a == retries - 1:
                return None
            await asyncio.sleep(2 ** a)


async def main():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(l) for l in open(JC / "persona" / "paired_scores.jsonl")]
    random.Random(0).shuffle(pairs); tune = pairs[:2000]
    gpt0 = {r["trait"] + "|" + r["label"]: r["score"]
            for r in (json.loads(l) for l in open(JC / "logprob" / "gpt_temp0_single.jsonl"))
            if r["score"] is not None}
    done = {}
    if CACHE.exists():
        for r in (json.loads(l) for l in open(CACHE)):
            done[r["trait"] + "|" + r["label"]] = r
    todo = [it for it in tune if it["trait"] + "|" + it["label"] not in done]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    rl = RateLimiter(100)
    resp_cache, ep_cache, prompts = {}, {}, []
    for it in todo:
        t = it["trait"]
        if t not in ep_cache:
            ep_cache[t] = hinted_ep(t)
            resp_cache[t] = {json.loads(x)["label"]: json.loads(x)
                             for x in open(ROOT / "traits40_generation" / "responses" / f"{t}.jsonl")}
        r = resp_cache[t][it["label"]]
        prompts.append(ep_cache[t].format(question=r["question"], answer=assistant_text(r)))
    with open(CACHE, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), 100):
            chunk, pchunk = todo[i:i+100], prompts[i:i+100]
            raws = await asyncio.gather(*[one(client, p, rl) for p in pchunk])
            for it, raw in zip(chunk, raws):
                rec = {"trait": it["trait"], "label": it["label"], "score": parse_persona_score(raw)}
                done[it["trait"]+"|"+it["label"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush(); print(f"  {min(i+100,len(todo))}/{len(todo)}", flush=True)

    def b(x): return 1 if x >= 50 else 0
    rows = [it for it in tune if it["trait"]+"|"+it["label"] in gpt0
            and done.get(it["trait"]+"|"+it["label"], {}).get("score") is not None]
    G = np.array([b(gpt0[it["trait"]+"|"+it["label"]]) for it in rows])
    DSnew = np.array([b(done[it["trait"]+"|"+it["label"]]["score"]) for it in rows])
    DSold = np.array([b(it["deepseek"]) for it in rows])
    print("\n========== DeepSeek + first-system-prompt hint (temp0) vs GPT temp0 ==========")
    print(f"BASELINE  GPT0 vs DS0-original: agree={100*np.mean(G==DSold):.1f}%  kappa={cohen_kappa_score(G,DSold):.3f}  (n={len(rows)})")
    print(f"NEW       GPT0 vs DS0-hinted:   agree={100*np.mean(G==DSnew):.1f}%  kappa={cohen_kappa_score(G,DSnew):.3f}  (n={len(rows)})")
    for pol in ["positive", "negative"]:
        idx = [i for i, it in enumerate(rows) if it["label"].startswith(pol)]
        print(f"    {pol:9s} new agree={100*np.mean(G[idx]==DSnew[idx]):.1f}%  (n={len(idx)})")
    cnew = np.array([done[it["trait"]+"|"+it["label"]]["score"] for it in rows])
    cold = np.array([it["deepseek"] for it in rows])
    print(f"mean|DS hinted - DS original| = {np.mean(np.abs(cnew-cold)):.1f}   (DS moved this much on 0-100)")


if __name__ == "__main__":
    asyncio.run(main())
