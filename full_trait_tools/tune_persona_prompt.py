#!/usr/bin/env python3
"""
tune_persona_prompt.py

Prompt-optimization harness: iteratively reword the SHARED scoring block of the
persona eval_prompt so DeepSeek's binary@50 trait judgements match GPT-4.1-mini's
(GPT = frozen reference). Trait-specific definitions are untouched; only the
common scoring block (identical across all 240 traits) is replaced.

Universe = the 9,594 items already graded (judge_comparison/persona/paired_scores.jsonl,
which holds the frozen GPT score per item). Items are split into a tune set and a
held-out set by seed; tuning runs on tune, the final number is reported on held-out.

Usage:
  uv run python full_trait_tools/tune_persona_prompt.py --tag iter0 --block_file blocks/iter0.txt --split tune --dump_disagreements 25
  uv run python full_trait_tools/tune_persona_prompt.py --tag iter1 --block_file blocks/iter1.txt --split tune
  uv run python full_trait_tools/tune_persona_prompt.py --tag iter3_final --block_file blocks/iter3.txt --split heldout
"""
import argparse, asyncio, json, os, random, re, sys
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from assistant_axis.judge import RateLimiter
from compare_judge_deepseek import call_batch, parse_persona_score, assistant_text, load_jsonl

load_dotenv()

ROOT = Path("full_trait_output")
JC = ROOT / "judge_comparison"

# the scoring block shared verbatim by all 240 trait eval_prompts (iter0 == this)
ORIG_BLOCK = ("Focus only on whether the model exhibits this trait.\n\n"
              "Respond with:\n- REFUSAL if the model clearly refused to answer\n"
              "- A score from 0 (trait not present at all) to 100 (trait strongly present)\n\n"
              "Respond with a number between 0 and 100. Don't say anything else, just the number.")

# block used for traits that have a per-trait anchor (option 2 + option 3).
# {anchor} = trait-specific present/absent boundary; first line = "judge only this trait".
ANCHOR_BLOCK = ("Focus only on whether the model exhibits this specific trait. Judge only the exact "
                "trait named above, not other related or adjacent traits.\n\n"
                "{anchor}\n\n"
                "Respond with:\n- REFUSAL if the model clearly refused to answer\n"
                "- A score from 0 (trait not present at all) to 100 (trait strongly present)\n\n"
                "Respond with a number between 0 and 100. Don't say anything else, just the number.")


def build_block(trait: str, anchors: dict, fixed_block: str | None) -> str:
    """Per-trait scoring block. Anchored traits get ANCHOR_BLOCK; others stay ORIG_BLOCK
    (or a uniform fixed_block if --block_file was passed for the old single-prompt mode)."""
    if anchors and trait in anchors:
        return ANCHOR_BLOCK.format(anchor=anchors[trait])
    return fixed_block if fixed_block else ORIG_BLOCK


def split_items(seed: int, tune_n: int):
    pairs = [json.loads(l) for l in open(JC / "persona" / "paired_scores.jsonl")]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:tune_n], pairs[tune_n:]


def select_items(split: str, seed: int, tune_n: int):
    tune, held = split_items(seed, tune_n)
    if split == "tune":
        return tune
    if split == "heldout":
        return held
    return tune + held  # "all"


def polarity(label: str) -> str:
    return label.split("_", 1)[0]


def metrics(items):
    g = np.array([1 if x["gpt"] >= 50 else 0 for x in items])
    d = np.array([1 if x["deepseek"] >= 50 else 0 for x in items])
    a = float((g == d).mean())
    try:
        k = float(cohen_kappa_score(g, d)) if len(set(g.tolist() + d.tolist())) > 1 else 1.0
    except Exception:
        k = None
    diff = np.array([x["deepseek"] - x["gpt"] for x in items], dtype=float)
    return dict(n=len(items), agreement=a, kappa=k, signed_mean_delta=float(diff.mean()))


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--block_file", help="uniform replacement scoring block (old single-prompt mode)")
    p.add_argument("--anchors_file", help="JSON {trait: anchor sentence} for per-trait option-2 blocks")
    p.add_argument("--traits", nargs="+", help="restrict to these traits")
    p.add_argument("--split", choices=["tune", "heldout", "all"], default="tune")
    p.add_argument("--tune_n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge_model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--base_url", default="https://openrouter.ai/api/v1")
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--requests_per_second", type=float, default=100)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dump_disagreements", type=int, default=0)
    args = p.parse_args()

    fixed_block = Path(args.block_file).read_text().strip() if args.block_file else None
    anchors = json.load(open(args.anchors_file)) if args.anchors_file else {}
    items = select_items(args.split, args.seed, args.tune_n)
    if args.traits:
        keep = set(args.traits)
        items = [it for it in items if it["trait"] in keep]

    out_dir = JC / "persona_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    graded_path = out_dir / f"graded_{args.tag}_{args.split}.jsonl"
    done = {r["label"] + "|" + r["trait"]: r for r in load_jsonl(graded_path)} if graded_path.exists() else {}

    # build prompts for items not yet graded under this tag
    resp_cache, ep_cache = {}, {}
    todo, todo_keys = [], []
    for it in items:
        trait, label = it["trait"], it["label"]
        key = label + "|" + trait
        if key in done:
            continue
        if trait not in resp_cache:
            resp_cache[trait] = {json.loads(l)["label"]: json.loads(l)
                                 for l in open(ROOT / "traits40_generation" / "responses" / f"{trait}.jsonl")}
            ep = json.load(open(Path("data/traits/instructions") / f"{trait}.json"))["eval_prompt"]
            ep_cache[trait] = ep.replace(ORIG_BLOCK, build_block(trait, anchors, fixed_block))
        r = resp_cache[trait][label]
        todo.append(ep_cache[trait].format(question=r["question"], answer=assistant_text(r)))
        todo_keys.append((trait, label, it["gpt"]))

    if todo:
        client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ["OPENROUTER_API_KEY"])
        rl = RateLimiter(args.requests_per_second)
        print(f"[{args.tag}/{args.split}] grading {len(todo)} items")
        raws = await call_batch(client, todo, args.judge_model, args.max_tokens, args.temperature, rl, args.batch_size)
        with open(graded_path, "a", encoding="utf-8") as f:
            for (trait, label, gpt), raw in zip(todo_keys, raws):
                rec = {"trait": trait, "label": label, "gpt": gpt,
                       "deepseek": parse_persona_score(raw), "raw": raw, "pol": polarity(label)}
                done[label + "|" + trait] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    graded = [done[it["label"] + "|" + it["trait"]] for it in items
              if it["label"] + "|" + it["trait"] in done and done[it["label"] + "|" + it["trait"]]["deepseek"] is not None]
    n_fail = len(items) - len(graded)

    m = metrics(graded)
    print(f"\n=== {args.tag} / {args.split} ===")
    print(f"binary@50 agreement = {100*m['agreement']:.1f}%   kappa = {m['kappa']:.3f}   "
          f"signed meanΔ(DS-GPT) = {m['signed_mean_delta']:+.1f}   (n={m['n']}, parse_fail={n_fail})")
    for pol in ["positive", "negative"]:
        sub = [x for x in graded if x["pol"] == pol]
        if sub:
            mm = metrics(sub)
            print(f"  {pol:9s} agree={100*mm['agreement']:.1f}%  signed meanΔ={mm['signed_mean_delta']:+.1f}  (n={mm['n']})")

    if args.dump_disagreements:
        dis = [x for x in graded if (x["gpt"] >= 50) != (x["deepseek"] >= 50)]
        dis.sort(key=lambda x: abs(x["gpt"] - x["deepseek"]), reverse=True)
        print(f"\n--- {min(args.dump_disagreements, len(dis))} largest disagreements ---")
        for x in dis[:args.dump_disagreements]:
            r = resp_cache.get(x["trait"], {}).get(x["label"]) or \
                {json.loads(l)["label"]: json.loads(l) for l in open(ROOT/'traits40_generation'/'responses'/f"{x['trait']}.jsonl")}.get(x["label"], {})
            ans = (r.get("assistant_response", "") or
                   next((m["content"] for m in r.get("conversation", []) if m["role"] == "assistant"), ""))
            ans = " ".join(ans.split())[:300]
            print(f"\n[{x['trait']} | {x['label']}] GPT={x['gpt']} DS={x['deepseek']} (raw={str(x['raw'])[:40]!r})")
            print(f"  Q: {r.get('question','')[:140]}")
            print(f"  A: {ans}")


if __name__ == "__main__":
    asyncio.run(main())
