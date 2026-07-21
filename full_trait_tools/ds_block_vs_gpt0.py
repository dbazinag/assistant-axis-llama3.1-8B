#!/usr/bin/env python3
"""Re-score DeepSeek (temp0) with a replacement scoring block, compare to GPT temp0 scores
(gpt_temp0_single.jsonl) and to DeepSeek's original temp0 scores. Tune split (2000). Resumable.
Usage: ds_block_vs_gpt0.py --tag embody --block_file /tmp/blk_embody.txt"""
import argparse, asyncio, json, os, random, sys
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
ORIG_BLOCK = ("Focus only on whether the model exhibits this trait.\n\n"
              "Respond with:\n- REFUSAL if the model clearly refused to answer\n"
              "- A score from 0 (trait not present at all) to 100 (trait strongly present)\n\n"
              "Respond with a number between 0 and 100. Don't say anything else, just the number.")


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
    p = argparse.ArgumentParser(); p.add_argument("--tag", required=True); p.add_argument("--block_file", required=True)
    args = p.parse_args()
    block = Path(args.block_file).read_text().strip()
    cache = JC / "logprob" / f"ds_{args.tag}.jsonl"
    pairs = [json.loads(l) for l in open(JC / "persona" / "paired_scores.jsonl")]
    random.Random(0).shuffle(pairs); tune = pairs[:2000]
    gpt0 = {r["trait"]+"|"+r["label"]: r["score"]
            for r in (json.loads(l) for l in open(JC/"logprob"/"gpt_temp0_single.jsonl")) if r["score"] is not None}
    done = {}
    if cache.exists():
        for r in (json.loads(l) for l in open(cache)):
            done[r["trait"]+"|"+r["label"]] = r
    todo = [it for it in tune if it["trait"]+"|"+it["label"] not in done]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    rl = RateLimiter(100)
    resp_cache, ep_cache, prompts = {}, {}, []
    for it in todo:
        t = it["trait"]
        if t not in ep_cache:
            ep_cache[t] = json.load(open(TI/f"{t}.json"))["eval_prompt"].replace(ORIG_BLOCK, block)
            resp_cache[t] = {json.loads(x)["label"]: json.loads(x)
                             for x in open(ROOT/"traits40_generation"/"responses"/f"{t}.jsonl")}
        r = resp_cache[t][it["label"]]
        prompts.append(ep_cache[t].format(question=r["question"], answer=assistant_text(r)))
    with open(cache, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), 100):
            raws = await asyncio.gather(*[one(client, pr, rl) for pr in prompts[i:i+100]])
            for it, raw in zip(todo[i:i+100], raws):
                rec = {"trait": it["trait"], "label": it["label"], "score": parse_persona_score(raw)}
                done[it["trait"]+"|"+it["label"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False)+"\n")
            f.flush(); print(f"  [{args.tag}] {min(i+100,len(todo))}/{len(todo)}", flush=True)

    def b(x): return 1 if x >= 50 else 0
    rows = [it for it in tune if it["trait"]+"|"+it["label"] in gpt0
            and done.get(it["trait"]+"|"+it["label"], {}).get("score") is not None]
    G = np.array([b(gpt0[it["trait"]+"|"+it["label"]]) for it in rows])
    DSn = np.array([b(done[it["trait"]+"|"+it["label"]]["score"]) for it in rows])
    DSo = np.array([b(it["deepseek"]) for it in rows])
    print(f"\n===== DS '{args.tag}' (temp0) vs GPT temp0 =====")
    print(f"  baseline DS-original: agree={100*np.mean(G==DSo):.1f}%  kappa={cohen_kappa_score(G,DSo):.3f}")
    print(f"  NEW      DS-{args.tag}: agree={100*np.mean(G==DSn):.1f}%  kappa={cohen_kappa_score(G,DSn):.3f}  (n={len(rows)})")
    for pol in ["positive", "negative"]:
        idx = [i for i, it in enumerate(rows) if it["label"].startswith(pol)]
        print(f"    {pol:9s} agree={100*np.mean(G[idx]==DSn[idx]):.1f}%")
    cn = np.array([done[it["trait"]+"|"+it["label"]]["score"] for it in rows]); co = np.array([it["deepseek"] for it in rows])
    print(f"  mean|DS-{args.tag} - DS-original| = {np.mean(np.abs(cn-co)):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
