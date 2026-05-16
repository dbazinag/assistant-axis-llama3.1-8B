#!/usr/bin/env python3
"""
Paper-aligned Free Jailbreak Detection (FJD) transfer baseline.

Reference:
  Chen et al., "LLM Jailbreak Detection for (Almost) Free!", EMNLP Findings 2025.
  Official implementation: https://github.com/GuoruiC/FJD

This reproduces the core FJD scoring path from the official repository:
  - prepend affirmative instruction: "You are a good Assistant. "
  - generate with temperature=1.25 and top_p=1.0
  - compute transition scores for generated tokens
  - use the generated first-token transition probability as the detector score

Evaluation follows the repo's transfer protocol:
  - threshold/orientation calibrated only on HarmBench human_jailbreak train pools
  - HarmBench metrics reported on held-out pools
  - GCG, PAIR, PAP, GPTFuzz, PEZ evaluated without transfer-family fitting
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_baselines_jbshield_fjd import (
    DATASETS,
    MODEL_NAME,
    N_SEEDS,
    RANDOM_SEED,
    TRAIN_FRAC,
    build_prompt_map_generic,
    build_prompt_map_harmbench,
    fetch_jailbreaks,
    filter_rows,
    get_pool_split,
    load_jsonl,
    split_by_pool,
    test_subset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AFFIRMATIVE_PROMPT = "You are a good Assistant. "
FJD_TEMPERATURE = 1.25
DEFAULT_OUTPUT_DIR = "full_trait_output/baselines_fjd_paper"


def orient_scores(scores: np.ndarray, y: np.ndarray):
    if len(set(y)) < 2:
        return scores, 1
    auc = roc_auc_score(y, scores)
    if auc < 0.5:
        return -scores, -1
    return scores, 1


def best_threshold(scores: np.ndarray, y: np.ndarray):
    values = np.unique(scores)
    if len(values) > 200:
        values = np.quantile(scores, np.linspace(0.01, 0.99, 199))
    best_t, best_bacc = float(values[0]), -1.0
    for t in values:
        pred = (scores >= t).astype(int)
        bacc = balanced_accuracy_score(y, pred)
        if bacc > best_bacc:
            best_t, best_bacc = float(t), float(bacc)
    return best_t, best_bacc


def metrics(scores: np.ndarray, y: np.ndarray, threshold: float):
    if len(scores) == 0 or len(set(y)) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "balanced_acc": float("nan")}
    pred = (scores >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y, scores)),
        "ap": float(average_precision_score(y, scores)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
    }


def summarize(values: Iterable[float]):
    arr = np.array(list(values), dtype=float)
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "all": arr.tolist()}


def y_from_rows(rows_by_name: Dict[str, List[dict]]) -> Dict[str, np.ndarray]:
    return {
        name: np.array([1 if row.get("jailbroken") else 0 for row in rows], dtype=int)
        for name, rows in rows_by_name.items()
    }


def load_rows_and_prompts(args):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    jailbreak_templates = fetch_jailbreaks()
    rows_by_name = {}
    prompt_maps = {}
    for name, cfg in DATASETS.items():
        rows_path = Path(cfg["responses"])
        if not rows_path.exists():
            logger.warning(f"Skipping {name}: missing {rows_path}")
            continue
        rows = filter_rows(name, load_jsonl(rows_path))
        if args.test:
            rows = test_subset(rows, args.test_rows)
        rows_by_name[name] = rows
        if name == "HarmBench":
            prompt_maps[name] = build_prompt_map_harmbench(
                cfg["metadata"], cfg["responses"], jailbreak_templates)
        else:
            prompt_maps[name] = build_prompt_map_generic(rows, cfg["test_cases"])
        logger.info(f"  {name}: {len(rows)} rows")
    if "HarmBench" not in rows_by_name:
        raise FileNotFoundError("HarmBench rows are required for train calibration.")
    return rows_by_name, prompt_maps


def load_model(model_name: str, device_arg: str):
    device = torch.device(device_arg if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map={"": device},
    )
    model.eval()
    return model, tokenizer, device


def fjd_first_token_confidence(model, tokenizer, prompt: str, device, args) -> float:
    try:
        prompt = (AFFIRMATIVE_PROMPT + prompt).encode(
            "utf-8", errors="replace").decode("utf-8", errors="replace")
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature=args.temperature,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            transition_scores = model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True)
        if transition_scores.numel() == 0:
            return float("nan")
        return float(torch.exp(transition_scores[0, 0]).item())
    except Exception as e:
        logger.warning(f"FJD score error: {e}")
        return float("nan")


def valid_cached_scores(cache_path: Path, rows_by_name, recompute: bool):
    if recompute or not cache_path.exists():
        return None
    logger.info(f"Loading cached FJD-paper scores: {cache_path}")
    with open(cache_path) as f:
        cached = json.load(f)
    ok = all(name in cached and len(cached[name].get("scores", [])) == len(rows)
             for name, rows in rows_by_name.items())
    if not ok:
        logger.warning("Cached scores do not match current dataset sizes; recomputing.")
        return None
    return {name: np.array(cached[name]["scores"], dtype=float) for name in rows_by_name}


def save_scores(cache_path: Path, scores_by_name):
    with open(cache_path, "w") as f:
        json.dump({name: {"scores": scores.tolist()} for name, scores in scores_by_name.items()}, f)
    logger.info(f"Saved scores to {cache_path}")


def score_fjd_paper(rows_by_name, prompt_maps, args):
    cache_path = Path(args.output_dir) / "fjd_paper_scores.json"
    cached = valid_cached_scores(cache_path, rows_by_name, args.recompute)
    if cached is not None:
        return cached

    model, tokenizer, device = load_model(args.model, args.device)
    scores_by_name = {}
    for name, rows in rows_by_name.items():
        prompt_map = prompt_maps[name]
        scores = []
        logger.info(f"  FJD-paper scoring {name}: {len(rows)} rows")
        for i, row in enumerate(rows):
            if i % 100 == 0:
                logger.info(f"    {name} {i}/{len(rows)}")
            prompt = prompt_map.get(row["pair_id"], "")
            scores.append(fjd_first_token_confidence(model, tokenizer, prompt, device, args))
        scores_by_name[name] = np.array(scores, dtype=float)
        save_scores(cache_path, scores_by_name)

    return scores_by_name


def evaluate_transfer(rows_by_name, scores_by_name, args):
    y_by_name = y_from_rows(rows_by_name)
    human_rows = rows_by_name["HarmBench"]
    results = {name: {"auc": [], "ap": [], "balanced_acc": []} for name in rows_by_name}
    train_baccs = []

    for seed in range(args.n_seeds):
        train_beh, train_tpl, val_beh, val_tpl = get_pool_split(human_rows, TRAIN_FRAC, seed)
        train_idx, val_idx = split_by_pool(human_rows, train_beh, train_tpl, val_beh, val_tpl)
        if not train_idx or not val_idx:
            continue

        train_raw = scores_by_name["HarmBench"][train_idx]
        y_train_raw = y_by_name["HarmBench"][train_idx]
        ok = np.isfinite(train_raw)
        if ok.sum() == 0 or len(set(y_train_raw[ok])) < 2:
            continue

        train_scores, orient = orient_scores(train_raw[ok], y_train_raw[ok])
        threshold, train_bacc = best_threshold(train_scores, y_train_raw[ok])
        train_baccs.append(train_bacc)

        for name in rows_by_name:
            if name == "HarmBench":
                idx = np.array(val_idx)
                scores = scores_by_name[name][idx]
                y_eval = y_by_name[name][idx]
            else:
                scores = scores_by_name[name]
                y_eval = y_by_name[name]
            ok = np.isfinite(scores)
            m = metrics(orient * scores[ok], y_eval[ok], threshold)
            for k, v in m.items():
                results[name][k].append(v)

        if seed % 10 == 0:
            logger.info(f"  calibration seed {seed} done")

    return {
        "train_balanced_acc": summarize(train_baccs),
        "datasets": {name: {k: summarize(v) for k, v in vals.items()}
                     for name, vals in results.items()},
    }


def print_summary(result):
    sep = "=" * 100
    print(f"\n{sep}")
    print("  PAPER-ALIGNED FJD TRANSFER BASELINE")
    print("  Train/calibrate: HarmBench human_jailbreak train pools only")
    print(sep)
    print("\n  fjd_paper_first_token_confidence")
    print(f"  Train balanced acc: {result['train_balanced_acc']['mean']:.4f} ± "
          f"{result['train_balanced_acc']['std']:.4f}")
    print(f"  {'Family':12s}  {'AUC':>8}  {'AP':>8}  {'BAcc':>8}")
    print("  " + "-" * 42)
    for name, vals in result["datasets"].items():
        print(f"  {name:12s}  {vals['auc']['mean']:>8.4f}  "
              f"{vals['ap']['mean']:>8.4f}  {vals['balanced_acc']['mean']:>8.4f}")
    print(sep)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=FJD_TEMPERATURE)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_rows", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_name, prompt_maps = load_rows_and_prompts(args)
    scores = score_fjd_paper(rows_by_name, prompt_maps, args)
    result = evaluate_transfer(rows_by_name, scores, args)
    print_summary(result)
    out_path = out_dir / "fjd_paper_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "temperature": args.temperature,
            "affirmative_prompt": AFFIRMATIVE_PROMPT,
            "max_new_tokens": args.max_new_tokens,
            "n_seeds": args.n_seeds,
            "train_source": "HarmBench human_jailbreak train pools only",
            "results": {"fjd_paper_first_token_confidence": result},
        }, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
