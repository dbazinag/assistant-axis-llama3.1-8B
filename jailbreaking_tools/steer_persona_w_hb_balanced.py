#!/usr/bin/env python3
"""
steer_persona_w_hb_balanced.py

HarmBench 50/50 baseline-split evaluation of the HarmBench persona-classifier w
as a steering vector.

  - Train the persona classifier (proj = x @ trait_matrix.T -> StandardScaler ->
    LogReg) on the HarmBench human_jailbreak TRAIN pool only (strict pool split,
    SPLIT_SEED), and map its decision normal into activation space (persona_w).
  - Sample a balanced set of TEST-pool pairs: half that were JAILBROKEN at baseline
    (label 1) and half NOT jailbroken at baseline (label 0). The classifier never
    trained on these pairs.
  - Steer +w (toward jailbreak), escalating alpha until the GPT-4.1-mini judge flags
    a jailbreak (judge-only escalation; identical mechanism to the JBB persona-w run).
  - Report three ASR numbers:
      * overall ASR over the full 50/50 set
      * ASR among pairs that STARTED jailbroken
      * ASR among pairs that STARTED not-jailbroken

Usage:
  uv run python jailbreaking_tools/steer_persona_w_hb_balanced.py
  uv run python jailbreaking_tools/steer_persona_w_hb_balanced.py --n_per_group 40 --test
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # jailbreaking_tools
import trait_steering_attack_jbb as base
from trait_steering_attack_jbb import (
    LAYER,
    MODEL_NAME,
    RANDOM_SEED,
    REPO_ROOT,
    fetch_jailbreaks,
    fit_classifier,
    load_jsonl,
    load_trait_matrix_and_names,
    logger,
)
from trait_steering_attack_jbb_persona_w import escalate_until_jailbroken, persona_w

sys.path.insert(0, str(REPO_ROOT / "full_trait_tools"))
from steering_robustness_eval_v2 import (  # noqa: E402
    SPLIT_SEED,
    TRAIN_FRAC,
    get_test_pool,
    reconstruct_prompt,
    sample_test_pairs,
)


def train_hb_classifier(trait_matrix, exclude_test=True):
    """Fit the persona classifier on HarmBench human_jailbreak pairs.

    exclude_test=True : strict split — train on the train pool only (no leakage).
    exclude_test=False: train on ALL human_jailbreak pairs (full-dataset mode).
    """
    rows = [r for r in load_jsonl(base.HARMBENCH_CLASSIFIED)
            if r.get("attack_type") == "human_jailbreak"]
    test_beh, test_tpl = get_test_pool(rows, TRAIN_FRAC, SPLIT_SEED)
    acts = torch.load(base.HARMBENCH_ACTS, map_location="cpu", weights_only=False)

    layer_key = str(LAYER)
    X, y = [], []
    for r in rows:
        # strict split: a TEST pair has unseen behavior AND unseen template
        if exclude_test and r["behavior_id"] in test_beh and r["jailbreak_idx"] in test_tpl:
            continue
        pid, jb = r.get("pair_id"), r.get("jailbroken")
        if pid is None or jb is None:
            continue
        item = acts.get(pid)
        if item is None or layer_key not in item:
            continue
        X.append(item[layer_key].float().numpy() @ trait_matrix.T)
        y.append(1 if jb else 0)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    clf, scaler = fit_classifier(X, y)
    scope = "TRAIN pool" if exclude_test else "ALL pairs"
    logger.info(f"[clf] HarmBench {scope}: X={X.shape}  jb_rate={y.mean():.3f}")
    return clf, scaler


def build_full_pool(templates):
    """All HarmBench human_jailbreak pairs, split by baseline jailbroken label."""
    rows = [r for r in load_jsonl(base.HARMBENCH_CLASSIFIED)
            if r.get("attack_type") == "human_jailbreak"]
    jb_pairs, rf_pairs = [], []
    for r in rows:
        if r.get("jailbroken") is None or r.get("pair_id") is None:
            continue
        pair = {
            "pair_id": r["pair_id"],
            "behavior_id": r.get("behavior_id"),
            "behavior_text": r.get("behavior_text", ""),
            "formatted_prompt": reconstruct_prompt(r, templates),
        }
        (jb_pairs if r["jailbroken"] else rf_pairs).append(pair)
    return jb_pairs, rf_pairs


def load_done_ids(path):
    """pair_ids already present in results.jsonl (for resume)."""
    if not path.exists():
        return set()
    done = set()
    for r in load_jsonl(path):
        if "pair_id" in r:
            done.add(str(r["pair_id"]))
    return done


def run_group(pairs, started_jailbroken, w_vec, model, tokenizer, device, client,
              out_f, done_ids):
    tag = "started_JB" if started_jailbroken else "started_refused"
    for i, p in enumerate(pairs):
        pid = p.get("pair_id", p.get("behavior_id"))
        if str(pid) in done_ids:
            continue
        logger.info(f"\n[{tag}] [{i+1}/{len(pairs)}] {p.get('behavior_id')}")
        r = escalate_until_jailbroken(
            pair_id=pid, prompt=p["formatted_prompt"],
            goal=p.get("behavior_text", ""), w_vec=w_vec,
            model=model, tokenizer=tokenizer, device=device, client=client,
        )
        r["behavior_id"] = p.get("behavior_id")
        r["started_jailbroken"] = started_jailbroken
        out_f.write(json.dumps({k: v for k, v in r.items() if k != "history"},
                               ensure_ascii=True) + "\n")
        out_f.flush()


def metrics_from_file(path):
    """Recompute ASR from results.jsonl (resume-correct). Returns dict of rates + counts."""
    rows = load_jsonl(path)
    jb = [r for r in rows if r.get("started_jailbroken")]
    rf = [r for r in rows if not r.get("started_jailbroken")]
    rate = lambda rs: (sum(1 for r in rs if r["final_jailbroken"]) / len(rs)) if rs else 0.0
    return {
        "asr_overall": rate(rows),
        "asr_started_jailbroken": rate(jb),
        "asr_started_not_jailbroken": rate(rf),
        "n_total": len(rows), "n_started_jb": len(jb), "n_started_rf": len(rf),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",
                        default="full_trait_output/steer_persona_w_hb_balanced")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_per_group", type=int, default=50,
                        help="Target pairs per baseline class (balanced; capped by test pool).")
    parser.add_argument("--full_dataset", action="store_true",
                        help="Train on ALL 3180 human_jailbreak pairs and steer ALL of them "
                             "(natural class balance, in-sample). Default is the balanced test sample.")
    parser.add_argument("--alpha_schedule", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--test", action="store_true", help="3 pairs per group")
    args = parser.parse_args()

    base.ALPHA_SCHEDULE = args.alpha_schedule
    base.MAX_ITER = len(args.alpha_schedule)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model + trait matrix
    logger.info(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map={"": device})
    model.eval()

    trait_matrix, _ = load_trait_matrix_and_names(LAYER)
    logger.info(f"Trait matrix: {trait_matrix.shape}")

    # Persona w from the HarmBench classifier (train-pool only, or all pairs in full mode)
    clf, scaler = train_hb_classifier(trait_matrix, exclude_test=not args.full_dataset)
    w_vec = persona_w(clf, scaler, trait_matrix)
    logger.info(f"persona w unit-norm={np.linalg.norm(w_vec):.4f}")

    templates = fetch_jailbreaks()  # full set so every template reconstructs
    if args.full_dataset:
        jb_pairs, rf_pairs = build_full_pool(templates)
        target_pool = "HarmBench human_jailbreak ALL pairs (in-sample)"
        logger.info(f"Full dataset: {len(jb_pairs)} started-jailbroken + "
                    f"{len(rf_pairs)} started-refused = {len(jb_pairs)+len(rf_pairs)} pairs")
    else:
        # Balanced 50/50 baseline split from the TEST pool
        n = 3 if args.test else args.n_per_group
        jb_pairs = sample_test_pairs(base.HARMBENCH_CLASSIFIED, n, RANDOM_SEED, templates,
                                     jailbroken_only=True)
        rf_pairs = sample_test_pairs(base.HARMBENCH_CLASSIFIED, n, RANDOM_SEED, templates,
                                     jailbroken_only=False)
        k = min(len(jb_pairs), len(rf_pairs))
        jb_pairs, rf_pairs = jb_pairs[:k], rf_pairs[:k]
        target_pool = "HarmBench human_jailbreak TEST pool (strict split)"
        logger.info(f"Balanced 50/50: {k} started-jailbroken + {k} started-refused = {2*k} pairs")

    # Steer +w on all pairs, escalate, judge. Append + skip-done for resume safety.
    results_path = output_dir / "results.jsonl"
    done_ids = load_done_ids(results_path)
    if done_ids:
        logger.info(f"Resuming: {len(done_ids)} pairs already done")
    with open(results_path, "a", encoding="utf-8") as out_f:
        run_group(jb_pairs, True, w_vec, model, tokenizer, device, client, out_f, done_ids)
        run_group(rf_pairs, False, w_vec, model, tokenizer, device, client, out_f, done_ids)

    m = metrics_from_file(results_path)
    summary = {
        "steering_vector": "harmbench_persona_classifier_w",
        "target_pool": target_pool,
        "full_dataset": args.full_dataset,
        "alpha_schedule": base.ALPHA_SCHEDULE,
        "judge": "gpt-4.1-mini binary (judge-only escalation)",
        "n_total": m["n_total"],
        "n_started_jailbroken": m["n_started_jb"],
        "n_started_not_jailbroken": m["n_started_rf"],
        "asr_overall": m["asr_overall"],
        "asr_started_jailbroken": m["asr_started_jailbroken"],
        "asr_started_not_jailbroken": m["asr_started_not_jailbroken"],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  HarmBench persona-w steering — {target_pool}")
    print(f"  {m['n_total']} pairs ({m['n_started_jb']} started-jailbroken / "
          f"{m['n_started_rf']} started-refused)")
    print(f"  Alpha schedule: {base.ALPHA_SCHEDULE}")
    print(sep)
    print(f"  ASR (overall)                : {m['asr_overall']:.1%}")
    print(f"  ASR (started jailbroken)     : {m['asr_started_jailbroken']:.1%}")
    print(f"  ASR (started NOT jailbroken) : {m['asr_started_not_jailbroken']:.1%}")
    print(sep)
    logger.info(f"Saved to {output_dir}/")


if __name__ == "__main__":
    main()
