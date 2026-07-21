#!/usr/bin/env python3
"""
compare_judge_deepseek.py

Re-grades two judging surfaces with a DeepSeek judge (via OpenRouter) and
compares against the existing GPT-4.1-mini labels, to show our results are not
an artifact of one judge.

Surfaces:
  persona    : trait-expression scores (0-100) from full_trait_pipeline step 3.
               Reuses each trait's eval_prompt and the SAME stored model
               responses, so DeepSeek grades the identical (question, answer).
               Metrics: MAE, MSE, RMSE, exact/within-tolerance % agreement,
               Pearson, Spearman (per-trait + overall).
  jailbroken : HarmBench Yes/No "did the model comply" labels. Reuses the exact
               JUDGE_PROMPT from judge_harmbench_responses.py.
               Metrics: % agreement, Cohen's kappa, confusion. (MAE == MSE ==
               1 - agreement for binary labels.)

The judge model defaults to deepseek/deepseek-v4-flash with reasoning DISABLED
(it is a hybrid model; CoT would overflow the short token budget and break
parsing). Resumable: already-graded items are skipped on re-run.

Usage:
  uv run python full_trait_tools/compare_judge_deepseek.py --task probe
  uv run python full_trait_tools/compare_judge_deepseek.py --task both --persona_per_trait 20
"""

import argparse
import asyncio
import json
import glob
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from assistant_axis.judge import RateLimiter
from judge_harmbench_responses import JUDGE_PROMPT  # exact same prompt as GPT run

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# ── DeepSeek/OpenRouter call ────────────────────────────────────────────────

async def call_one(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    rate_limiter: RateLimiter,
    retries: int = 6,
) -> Optional[str]:
    for attempt in range(retries):
        await rate_limiter.acquire()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"reasoning": {"enabled": False}},  # no CoT
            )
            if resp.choices and resp.choices[0].message.content:
                return resp.choices[0].message.content
            return None
        except Exception as e:
            if attempt == retries - 1:
                logger.warning(f"call failed after {retries} tries: {e}")
                return None
            await asyncio.sleep(2 ** attempt)
    return None


async def call_batch(
    client: AsyncOpenAI,
    prompts: List[str],
    model: str,
    max_tokens: int,
    temperature: float,
    rate_limiter: RateLimiter,
    batch_size: int,
) -> List[Optional[str]]:
    results: List[Optional[str]] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        tasks = [call_one(client, p, model, max_tokens, temperature, rate_limiter) for p in batch]
        got = await asyncio.gather(*tasks, return_exceptions=True)
        for r in got:
            results.append(None if isinstance(r, Exception) else r)
        logger.info(f"  graded {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return results


# ── Parsing ─────────────────────────────────────────────────────────────────

def parse_persona_score(text: Optional[str]) -> Optional[int]:
    """Parse a 0-100 trait score. REFUSAL -> 0.

    DeepSeek sometimes ignores 'just the number' and explains first, ending with
    the score (e.g. '...so I'd rate it 70'). We therefore take the LAST integer
    in [0,100] rather than the first, and treat any REFUSAL mention as 0.
    """
    if not text:
        return None
    t = text.strip()
    if "REFUSAL" in t.upper():
        return 0
    vals = [int(n) for n in re.findall(r"\d{1,3}", t) if 0 <= int(n) <= 100]
    return vals[-1] if vals else None


def parse_jb_label(text: Optional[str]) -> Optional[bool]:
    """Parse Yes/No -> jailbroken bool. Robust to a leading clause before the verdict."""
    if not text:
        return None
    m = re.search(r"\b(yes|no)\b", text.strip().lower())
    if not m:
        return None
    return m.group(1) == "yes"


# ── Data loading ────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def assistant_text(row: dict) -> str:
    a = row.get("assistant_response", "")
    if a:
        return a
    return next((m["content"] for m in row.get("conversation", []) if m["role"] == "assistant"), "")


# ── Persona surface ─────────────────────────────────────────────────────────

async def run_persona(args, client, rate_limiter) -> None:
    scores_dir = Path(args.persona_scores_dir)
    responses_dir = Path(args.responses_dir)
    traits_dir = Path(args.traits_dir)
    out_dir = Path(args.output_root) / "persona"
    ds_dir = out_dir / "deepseek_scores"
    ds_dir.mkdir(parents=True, exist_ok=True)

    score_files = sorted(scores_dir.glob("*.json"))
    if args.traits:
        score_files = [f for f in score_files if f.stem in args.traits]
    rng = random.Random(args.seed)

    # ── pass 1: pick sampled labels per trait, load existing, collect global todo ──
    chosen_by_trait: Dict[str, List[str]] = {}
    gpt_by_trait: Dict[str, Dict[str, int]] = {}
    ds_by_trait: Dict[str, Dict[str, dict]] = {}
    todo: List[Tuple[str, str, str]] = []  # (trait, label, prompt)

    for sf in score_files:
        trait = sf.stem
        gpt_scores: Dict[str, int] = json.load(open(sf))
        resp_file = responses_dir / f"{trait}.jsonl"
        trait_file = traits_dir / f"{trait}.json"
        if not resp_file.exists() or not trait_file.exists():
            logger.warning(f"{trait}: missing responses/trait file, skipping")
            continue
        eval_prompt = json.load(open(trait_file)).get("eval_prompt", "")
        if not eval_prompt:
            continue
        resp_by_label = {r["label"]: r for r in load_jsonl(resp_file)}

        candidates = sorted(lab for lab in gpt_scores if lab in resp_by_label)
        ds_file = ds_dir / f"{trait}.json"
        ds_existing: Dict[str, dict] = json.load(open(ds_file)) if ds_file.exists() else {}

        N = args.persona_per_trait
        if N and N < len(candidates):
            # keep already-graded labels, top up with fresh random ones (so a larger
            # N stays a superset of a smaller earlier run and reuses cached scores)
            cached = [lab for lab in candidates if lab in ds_existing]
            if len(cached) >= N:
                chosen = sorted(cached[:N])
            else:
                remaining = [lab for lab in candidates if lab not in ds_existing]
                chosen = sorted(cached + rng.sample(remaining, N - len(cached)))
        else:
            chosen = candidates

        chosen_by_trait[trait] = chosen
        gpt_by_trait[trait] = gpt_scores
        ds_by_trait[trait] = ds_existing
        for lab in chosen:
            if lab not in ds_existing:
                r = resp_by_label[lab]
                todo.append((trait, lab, eval_prompt.format(question=r["question"], answer=assistant_text(r))))

    # ── pass 2: grade all todo prompts pooled into wide batches ──
    if todo:
        logger.info(f"PERSONA: grading {len(todo)} items across {len(chosen_by_trait)} traits")
        raws = await call_batch(client, [p for _, _, p in todo], args.judge_model,
                                args.persona_max_tokens, args.temperature, rate_limiter, args.batch_size)
        touched = set()
        for (trait, lab, _), raw in zip(todo, raws):
            ds_by_trait[trait][lab] = {"raw": raw, "score": parse_persona_score(raw)}
            touched.add(trait)
        for trait in touched:
            json.dump(ds_by_trait[trait], open(ds_dir / f"{trait}.json", "w"), indent=2, ensure_ascii=False)

    # ── assemble paired records ──
    paired: List[dict] = []
    n_parse_fail = 0
    for trait, chosen in chosen_by_trait.items():
        for lab in chosen:
            ds = ds_by_trait[trait].get(lab, {})
            if ds.get("score") is None:
                n_parse_fail += 1
                continue
            paired.append({"trait": trait, "label": lab, "gpt": gpt_by_trait[trait][lab],
                           "deepseek": ds["score"], "raw": ds.get("raw")})

    # save paired + metrics
    with open(out_dir / "paired_scores.jsonl", "w", encoding="utf-8") as f:
        for p in paired:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    metrics = persona_metrics(paired, n_parse_fail)
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=2)
    logger.info(f"PERSONA: n={metrics['overall']['n']} MAE={metrics['overall']['mae']:.3f} "
                f"MSE={metrics['overall']['mse']:.3f} exact={metrics['overall']['agree_exact']:.3f} "
                f"within10={metrics['overall']['agree_within_10']:.3f} "
                f"spearman={metrics['overall']['spearman']:.3f}")


def _pair_metrics(g: np.ndarray, d: np.ndarray) -> dict:
    diff = g - d
    out = {
        "n": int(len(g)),
        "mae": float(np.mean(np.abs(diff))),
        "mse": float(np.mean(diff ** 2)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "agree_exact": float(np.mean(diff == 0)),
        "agree_within_5": float(np.mean(np.abs(diff) <= 5)),
        "agree_within_10": float(np.mean(np.abs(diff) <= 10)),
        "gpt_mean": float(np.mean(g)),
        "deepseek_mean": float(np.mean(d)),
    }
    if len(g) >= 3 and np.std(g) > 0 and np.std(d) > 0:
        out["pearson"] = float(pearsonr(g, d)[0])
        out["spearman"] = float(spearmanr(g, d)[0])
    else:
        out["pearson"] = None
        out["spearman"] = None
    return out


def persona_metrics(paired: List[dict], n_parse_fail: int) -> dict:
    g = np.array([p["gpt"] for p in paired], dtype=float)
    d = np.array([p["deepseek"] for p in paired], dtype=float)
    overall = _pair_metrics(g, d) if len(g) else {"n": 0}
    overall["parse_failures"] = n_parse_fail
    per_trait = {}
    by = {}
    for p in paired:
        by.setdefault(p["trait"], []).append(p)
    for trait, items in sorted(by.items()):
        gg = np.array([x["gpt"] for x in items], dtype=float)
        dd = np.array([x["deepseek"] for x in items], dtype=float)
        per_trait[trait] = _pair_metrics(gg, dd)
    return {"overall": overall, "per_trait": per_trait}


# ── Jailbroken surface ──────────────────────────────────────────────────────

async def run_jailbroken(args, client, rate_limiter) -> None:
    src = Path(args.harmbench_path)
    out_dir = Path(args.output_root) / "jailbroken"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "harmbench_deepseek.jsonl"

    rows = load_jsonl(src)
    if args.limit:
        rows = rows[:args.limit]

    done = {}
    if out_file.exists():
        for r in load_jsonl(out_file):
            # keep only successfully-parsed rows; None (transient API failures) get retried
            if r.get("deepseek_label") is not None:
                done[r["pair_id"]] = r

    todo = [r for r in rows if r["pair_id"] not in done]
    if todo:
        prompts = [JUDGE_PROMPT.format(behavior=r["behavior_text"], response=r["response_text"]) for r in todo]
        logger.info(f"JAILBROKEN: grading {len(prompts)} (of {len(rows)} total)")
        raws = await call_batch(client, prompts, args.judge_model, args.jb_max_tokens,
                                args.temperature, rate_limiter, args.batch_size)
        for r, raw in zip(todo, raws):
            label = parse_jb_label(raw)
            rec = {"pair_id": r["pair_id"], "behavior_id": r["behavior_id"],
                   "attack_type": r.get("attack_type", "unknown"),
                   "gpt_label": bool(r["jailbroken"]),
                   "deepseek_label": (None if label is None else bool(label)),
                   "deepseek_raw": raw}
            done[r["pair_id"]] = rec
        # rewrite cleanly (one row per pair_id, no stale None duplicates)
        with open(out_file, "w", encoding="utf-8") as f:
            for r in rows:
                if r["pair_id"] in done:
                    f.write(json.dumps(done[r["pair_id"]], ensure_ascii=False) + "\n")

    paired = [done[r["pair_id"]] for r in rows
              if r["pair_id"] in done and done[r["pair_id"]]["deepseek_label"] is not None]
    n_parse_fail = len(rows) - len(paired)
    metrics = jailbroken_metrics(paired, n_parse_fail)
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=2)
    o = metrics["overall"]
    logger.info(f"JAILBROKEN: n={o['n']} agreement={o['agreement']:.3f} kappa={o['kappa']:.3f} "
                f"(MAE=MSE={o['mae']:.3f})")


def _jb_overall(items: List[dict]) -> dict:
    g = np.array([1 if x["gpt_label"] else 0 for x in items])
    d = np.array([1 if x["deepseek_label"] else 0 for x in items])
    agree = float(np.mean(g == d)) if len(g) else 0.0
    try:
        kappa = float(cohen_kappa_score(g, d)) if len(set(g.tolist() + d.tolist())) > 1 else 1.0
    except Exception:
        kappa = None
    return {
        "n": int(len(g)),
        "agreement": agree,
        "mae": float(1 - agree),   # == MSE for binary
        "mse": float(1 - agree),
        "kappa": kappa,
        "gpt_jailbroken_rate": float(np.mean(g)) if len(g) else 0.0,
        "deepseek_jailbroken_rate": float(np.mean(d)) if len(d) else 0.0,
        "both_yes": int(np.sum((g == 1) & (d == 1))),
        "both_no": int(np.sum((g == 0) & (d == 0))),
        "gpt_yes_ds_no": int(np.sum((g == 1) & (d == 0))),
        "gpt_no_ds_yes": int(np.sum((g == 0) & (d == 1))),
    }


def jailbroken_metrics(paired: List[dict], n_parse_fail: int) -> dict:
    overall = _jb_overall(paired)
    overall["parse_failures"] = n_parse_fail
    by = {}
    for x in paired:
        by.setdefault(x["attack_type"], []).append(x)
    per_attack = {at: _jb_overall(items) for at, items in sorted(by.items())}
    return {"overall": overall, "per_attack_type": per_attack}


# ── Probe ───────────────────────────────────────────────────────────────────

async def run_probe(args, client, rate_limiter) -> None:
    n = args.probe_n
    # grab some real persona prompts
    scores_dir = Path(args.persona_scores_dir)
    responses_dir = Path(args.responses_dir)
    traits_dir = Path(args.traits_dir)
    sf = sorted(scores_dir.glob("*.json"))[0]
    trait = sf.stem
    eval_prompt = json.load(open(traits_dir / f"{trait}.json")).get("eval_prompt", "")
    resp = load_jsonl(responses_dir / f"{trait}.jsonl")[:n]
    prompts = [eval_prompt.format(question=r["question"], answer=assistant_text(r)) for r in resp]

    t0 = time.time()
    raws = await call_batch(client, prompts, args.judge_model, args.persona_max_tokens,
                            args.temperature, rate_limiter, args.batch_size)
    dt = time.time() - t0
    ok = sum(1 for r in raws if parse_persona_score(r) is not None)
    rate = len(prompts) / dt if dt else 0
    logger.info(f"PROBE: {len(prompts)} calls in {dt:.1f}s -> {rate:.1f} req/s, parsed {ok}/{len(prompts)}")
    logger.info(f"  sample raw outputs: {[r for r in raws[:5]]}")
    if rate > 0:
        for label, count in [("persona 9,600", 9600), ("persona 96,000", 96000), ("jailbroken 3,339", 3339)]:
            logger.info(f"  ETA {label}: {count / rate / 60:.1f} min")


# ── Main ────────────────────────────────────────────────────────────────────

def build_client(args) -> AsyncOpenAI:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        logger.error("OPENROUTER_API_KEY not found in environment/.env")
        sys.exit(1)
    return AsyncOpenAI(base_url=args.base_url, api_key=key)


async def main_async():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["persona", "jailbroken", "both", "probe"], default="both")
    p.add_argument("--judge_model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--base_url", default="https://openrouter.ai/api/v1")
    p.add_argument("--output_root", default="full_trait_output/judge_comparison")
    p.add_argument("--persona_scores_dir", default="full_trait_output/traits40_judge/scores")
    p.add_argument("--responses_dir", default="full_trait_output/traits40_generation/responses")
    p.add_argument("--traits_dir", default="data/traits/instructions")
    p.add_argument("--harmbench_path", default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    p.add_argument("--persona_per_trait", type=int, default=20, help="0 = full set")
    p.add_argument("--traits", nargs="+", help="restrict to these traits")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--persona_max_tokens", type=int, default=256)
    p.add_argument("--jb_max_tokens", type=int, default=16)
    p.add_argument("--requests_per_second", type=float, default=100)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--limit", type=int, default=0, help="cap jailbroken rows (debug)")
    p.add_argument("--probe_n", type=int, default=20)
    args = p.parse_args()

    client = build_client(args)
    rate_limiter = RateLimiter(args.requests_per_second)

    if args.task == "probe":
        await run_probe(args, client, rate_limiter)
        return
    if args.task in ("persona", "both"):
        await run_persona(args, client, rate_limiter)
    if args.task in ("jailbroken", "both"):
        await run_jailbroken(args, client, rate_limiter)
    write_report(Path(args.output_root))


def write_report(root: Path) -> None:
    lines = ["# Judge comparison: GPT-4.1-mini vs DeepSeek-V4-Flash", ""]
    pm = root / "persona" / "metrics.json"
    jm = root / "jailbroken" / "metrics.json"
    if pm.exists():
        o = json.load(open(pm))["overall"]
        lines += ["## Persona trait scores (0-100)", "",
                  f"- n = {o.get('n')}  (parse failures: {o.get('parse_failures')})",
                  f"- **MAE = {o.get('mae'):.2f}**, MSE = {o.get('mse'):.2f}, RMSE = {o.get('rmse'):.2f}",
                  f"- % agreement: exact {100*o.get('agree_exact',0):.1f}%, within±5 {100*o.get('agree_within_5',0):.1f}%, within±10 {100*o.get('agree_within_10',0):.1f}%",
                  f"- Pearson {o.get('pearson')}, Spearman {o.get('spearman')}",
                  f"- mean score: GPT {o.get('gpt_mean'):.1f} vs DeepSeek {o.get('deepseek_mean'):.1f}", ""]
    if jm.exists():
        o = json.load(open(jm))["overall"]
        lines += ["## HarmBench jailbroken labels (binary)", "",
                  f"- n = {o.get('n')}  (parse failures: {o.get('parse_failures')})",
                  f"- **% agreement = {100*o.get('agreement',0):.1f}%**, Cohen's kappa = {o.get('kappa')}",
                  f"- MAE = MSE = {o.get('mae'):.3f}  (= 1 - agreement)",
                  f"- jailbroken rate: GPT {100*o.get('gpt_jailbroken_rate',0):.1f}% vs DeepSeek {100*o.get('deepseek_jailbroken_rate',0):.1f}%",
                  f"- confusion: both-yes {o.get('both_yes')}, both-no {o.get('both_no')}, "
                  f"GPT-yes/DS-no {o.get('gpt_yes_ds_no')}, GPT-no/DS-yes {o.get('gpt_no_ds_yes')}", ""]
    (root).mkdir(parents=True, exist_ok=True)
    (root / "report.md").write_text("\n".join(lines))
    logger.info(f"Wrote {root / 'report.md'}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
