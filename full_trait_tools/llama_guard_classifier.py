#!/usr/bin/env python3
"""
llama_guard_classifier.py

Baseline jailbreak detector using Llama-Guard-3-8B.
Runs the guard model on the input prompt for each pair, extracts the
probability of the "unsafe" token as the detection score, then trains
a logistic regression on HarmBench and tests on HarmBench held-out,
GCG, and PAIR — same regime as fast_transfer_classifier.py.

Llama Guard 3 outputs "safe" or "unsafe" as the first token. We extract
the logit difference (or probability of "unsafe") as a soft score for AUC.

HarmBench formatted prompts are reconstructed from pairs_metadata.jsonl
+ jailbreaks.py (same as perplexity_classifier.py).
GCG/PAIR prompts come from their test_cases.json files.

Usage:
  uv run python full_trait_tools/llama_guard_classifier.py
  uv run python full_trait_tools/llama_guard_classifier.py --test
"""

import argparse
import json
import logging
import random
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
GUARD_MODEL            = "meta-llama/Llama-Guard-3-8B"
JAILBREAKS_URL         = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/baselines/human_jailbreaks/jailbreaks.py"
)
SKIP_JAILBREAK_INDICES = {16, 44}
N_JAILBREAK_SAMPLES    = 20
HARMBENCH_SEED         = 42
TRAIN_FRAC             = 0.7
N_SEEDS                = 50
RANDOM_SEED            = 42


# ── Jailbreak template reconstruction (identical to perplexity_classifier.py) ─

def fetch_jailbreaks() -> Dict[int, str]:
    logger.info("Fetching jailbreaks.py from GitHub...")
    with urllib.request.urlopen(JAILBREAKS_URL) as resp:
        source = resp.read().decode("utf-8")
    ns = {}
    exec(source, ns)  # noqa: S102
    raw = ns["JAILBREAKS"]
    valid = {}
    for i, jb in enumerate(raw):
        if i in SKIP_JAILBREAK_INDICES:
            continue
        if "{0}" not in jb:
            continue
        valid[i] = jb
    logger.info(f"  Loaded {len(valid)} valid jailbreak templates")
    return valid


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_harmbench_prompt_map(
    metadata_path: Path,
    classified_path: Path,
    jailbreak_templates: Dict[int, str],
) -> Dict[int, str]:
    meta_rows = load_jsonl(metadata_path)
    classified_rows = load_jsonl(classified_path)
    behavior_lookup = {r["pair_id"]: r["behavior_text"] for r in classified_rows}
    prompt_map = {}
    for row in meta_rows:
        pid = row["pair_id"]
        jb_idx = row["jailbreak_idx"]
        behavior_text = row.get("behavior_text") or behavior_lookup.get(pid, "")
        if jb_idx == -1:
            prompt_map[pid] = behavior_text
        elif jb_idx in jailbreak_templates:
            try:
                prompt_map[pid] = jailbreak_templates[jb_idx].format(behavior_text)
            except Exception:
                prompt_map[pid] = behavior_text
        else:
            prompt_map[pid] = behavior_text
    logger.info(f"  Reconstructed {len(prompt_map)} HarmBench prompts")
    return prompt_map


def build_gcg_prompt_map(classified_path: Path, test_cases_path: Path) -> Dict[int, str]:
    test_cases = json.load(open(test_cases_path))
    rows = load_jsonl(classified_path)
    prompt_map = {}
    for row in rows:
        pid = row["pair_id"]
        bid = row["behavior_id"]
        if bid in test_cases and test_cases[bid]:
            prompt_map[pid] = test_cases[bid][0]
        else:
            prompt_map[pid] = row.get("behavior_text", "")
    return prompt_map


def build_pair_prompt_map(classified_path: Path, test_cases_path: Path) -> Dict[int, str]:
    test_cases = json.load(open(test_cases_path))
    rows = load_jsonl(classified_path)
    prompt_map = {}
    for row in rows:
        pid = row["pair_id"]
        bid = row["behavior_id"]
        attack_type = row.get("attack_type", "")
        if attack_type == "PAIR" and bid in test_cases and test_cases[bid]:
            prompt_map[pid] = test_cases[bid][0]
        else:
            prompt_map[pid] = row.get("behavior_text", "")
    return prompt_map


# ── Llama Guard scoring ────────────────────────────────────────────────────────

def build_guard_prompt(tokenizer, user_message: str, assistant_response: str = None) -> str:
    """
    Format prompt using Llama Guard's chat template.
    If assistant_response provided, evaluates the response (correct mode).
    If not, evaluates the prompt only (prompt-only mode).
    """
    if assistant_response:
        conversation = [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": assistant_response},
        ]
    else:
        conversation = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )


def get_unsafe_score(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    device: torch.device,
    safe_token_id: int,
    unsafe_token_id: int,
    assistant_response: str = None,
) -> float:
    """
    Returns P(unsafe) as a soft detection score in [0, 1].
    Uses prompt + response mode when assistant_response is provided
    (evaluates whether the agent's response is unsafe — correct for your task).
    """
    try:
        prompt_text = prompt_text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        if not prompt_text.strip():
            return float("nan")

        guard_input = build_guard_prompt(tokenizer, prompt_text, assistant_response)
        enc = tokenizer(
            guard_input,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Get logits for the next token (first generated token = safe/unsafe)
        next_token_logits = outputs.logits[0, -1, :]
        safe_logit   = next_token_logits[safe_token_id].float()
        unsafe_logit = next_token_logits[unsafe_token_id].float()

        # Softmax over just these two tokens → P(unsafe)
        probs = torch.softmax(torch.stack([safe_logit, unsafe_logit]), dim=0)
        return float(probs[1].item())

    except Exception as e:
        logger.warning(f"Guard score error: {e}")
        return float("nan")


def get_token_ids(tokenizer) -> Tuple[int, int]:
    """Get token IDs for 'safe' and 'unsafe'."""
    safe_ids   = tokenizer.encode("safe",   add_special_tokens=False)
    unsafe_ids = tokenizer.encode("unsafe", add_special_tokens=False)
    # Take last token in case tokenizer splits the word
    safe_id   = safe_ids[-1]
    unsafe_id = unsafe_ids[-1]
    logger.info(f"  Token IDs — safe: {safe_id}, unsafe: {unsafe_id}")
    logger.info(f"  Verification — safe: '{tokenizer.decode([safe_id])}', unsafe: '{tokenizer.decode([unsafe_id])}'")
    return safe_id, unsafe_id


def compute_all_scores(
    rows: List[dict],
    prompt_map: Dict[int, str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    safe_token_id: int,
    unsafe_token_id: int,
    desc: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """
    Compute Llama Guard scores in a single pass, both modes:
      X_input  [N x 1] — input-only mode (prompt only, pre-generation)
      X_full   [N x 1] — input+output mode (prompt + model response)
    Returns (X_input, X_full, y, valid_rows).
    """
    X_input_list, X_full_list, y_list, valid_rows = [], [], [], []

    for i, row in enumerate(rows):
        if i % 100 == 0:
            logger.info(f"  {desc} {i}/{len(rows)}")

        pid = row["pair_id"]
        jb = row.get("jailbroken")
        if jb is None:
            continue
        prompt = prompt_map.get(pid)
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            continue

        # Input-only score (pre-generation, same setting as activation classifier)
        score_input = get_unsafe_score(
            model, tokenizer, prompt, device, safe_token_id, unsafe_token_id,
            assistant_response=None,
        )

        # Input+output score (post-generation, easier task)
        response_text = row.get("response_text", None)
        if response_text and not response_text.strip():
            response_text = None
        score_full = get_unsafe_score(
            model, tokenizer, prompt, device, safe_token_id, unsafe_token_id,
            assistant_response=response_text,
        )

        if np.isnan(score_input) or np.isnan(score_full):
            continue

        X_input_list.append([score_input])
        X_full_list.append([score_full])
        y_list.append(1 if jb else 0)
        valid_rows.append(row)

    if not X_input_list:
        return np.array([]), np.array([]), np.array([]), []
    return np.stack(X_input_list), np.stack(X_full_list), np.array(y_list), valid_rows


# ── Pool split (identical to fast_transfer_classifier.py) ─────────────────────

def get_pool_split(rows, train_frac, seed):
    rng = random.Random(seed)
    all_behaviors = sorted({r["behavior_id"]  for r in rows})
    all_templates = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(all_behaviors)
    rng.shuffle(all_templates)
    n_beh = max(1, int(len(all_behaviors) * train_frac))
    n_tpl = max(1, int(len(all_templates) * train_frac))
    train_beh = set(all_behaviors[:n_beh])
    train_tpl = set(all_templates[:n_tpl])
    test_beh  = set(all_behaviors[n_beh:])
    test_tpl  = set(all_templates[n_tpl:])
    return train_beh, train_tpl, test_beh, test_tpl


def split_by_pool(rows, train_beh, train_tpl, test_beh, test_tpl):
    train_idx = [i for i, r in enumerate(rows)
                 if r["behavior_id"] in train_beh and r["jailbreak_idx"] in train_tpl]
    test_idx  = [i for i, r in enumerate(rows)
                 if r["behavior_id"] in test_beh  and r["jailbreak_idx"] in test_tpl]
    return train_idx, test_idx


def fit_eval(X_tr, y_tr, X_te, y_te):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, C=1.0)
    clf.fit(X_tr_s, y_tr)
    if len(set(y_te)) < 2:
        return float("nan")
    probs = clf.predict_proba(X_te_s)[:, 1]
    return float(roc_auc_score(y_te, probs))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guard_model",            default=GUARD_MODEL)
    parser.add_argument("--human_classified_path",  default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_metadata_path",    default="full_trait_output/harmbench_activations/pairs_metadata.jsonl")
    parser.add_argument("--gcg_classified_path",    default="full_trait_output/gcg_activations/responses.jsonl")
    parser.add_argument("--gcg_test_cases_path",    default="/dlabscratch1/bazina/HarmBench/results/GCG/llama3_1_8b/test_cases/test_cases.json")
    parser.add_argument("--pair_classified_path",   default="full_trait_output/pair_activations/responses.jsonl")
    parser.add_argument("--pair_test_cases_path",   default="/dlabscratch1/bazina/HarmBench/results/PAIR/llama3_1_8b/test_cases/test_cases.json")
    parser.add_argument("--output_dir",             default="full_trait_output/llama_guard_baseline")
    parser.add_argument("--n_seeds",  type=int,     default=N_SEEDS)
    parser.add_argument("--device",                 default="cuda")
    parser.add_argument("--test",     action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.test:
        output_dir = output_dir.parent / (output_dir.name + "_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load Llama Guard ───────────────────────────────────────────────────────
    logger.info(f"Loading {args.guard_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.guard_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.guard_model, dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()
    logger.info("Guard model loaded.")

    safe_token_id, unsafe_token_id = get_token_ids(tokenizer)

    # ── Reconstruct prompts ────────────────────────────────────────────────────
    logger.info("\n=== Reconstructing prompts ===")
    jailbreak_templates = fetch_jailbreaks()

    human_prompt_map = build_harmbench_prompt_map(
        Path(args.human_metadata_path),
        Path(args.human_classified_path),
        jailbreak_templates,
    )
    gcg_prompt_map  = build_gcg_prompt_map(
        Path(args.gcg_classified_path), Path(args.gcg_test_cases_path)
    )
    pair_prompt_map = build_pair_prompt_map(
        Path(args.pair_classified_path), Path(args.pair_test_cases_path)
    )

    # ── Load metadata ──────────────────────────────────────────────────────────
    logger.info("\n=== Loading metadata ===")
    human_rows_all = load_jsonl(Path(args.human_classified_path))
    human_rows = [r for r in human_rows_all if r.get("attack_type") == "human_jailbreak"]
    gcg_rows   = load_jsonl(Path(args.gcg_classified_path))
    pair_rows  = load_jsonl(Path(args.pair_classified_path))

    if args.test:
        bids = list({r["behavior_id"] for r in human_rows})[:3]
        human_rows = [r for r in human_rows if r["behavior_id"] in bids]
        gcg_rows   = gcg_rows[:20]
        pair_rows  = pair_rows[:20]

    # ── Compute scores ─────────────────────────────────────────────────────────
    logger.info("\n=== Computing Llama Guard scores (both modes, single pass) ===")

    logger.info("HarmBench...")
    X_human_input, X_human_full, y_human, human_valid = compute_all_scores(
        human_rows, human_prompt_map, model, tokenizer, device,
        safe_token_id, unsafe_token_id, "HarmBench"
    )
    logger.info(f"  {len(y_human)} pairs, {y_human.sum():.0f} jailbroken")

    logger.info("GCG...")
    X_gcg_input, X_gcg_full, y_gcg, _ = compute_all_scores(
        gcg_rows, gcg_prompt_map, model, tokenizer, device,
        safe_token_id, unsafe_token_id, "GCG"
    )
    logger.info(f"  {len(y_gcg)} pairs, {y_gcg.sum():.0f} jailbroken")

    logger.info("PAIR...")
    X_pair_input, X_pair_full, y_pair, _ = compute_all_scores(
        pair_rows, pair_prompt_map, model, tokenizer, device,
        safe_token_id, unsafe_token_id, "PAIR"
    )
    logger.info(f"  {len(y_pair)} pairs, {y_pair.sum():.0f} jailbroken")

    def best_auc(X, y):
        if len(set(y)) < 2 or len(X) == 0: return float("nan")
        a = roc_auc_score(y, X[:,0])
        return max(a, 1 - a)

    # Direct AUC — input only
    direct_human_input = best_auc(X_human_input, y_human)
    direct_gcg_input   = best_auc(X_gcg_input,   y_gcg)
    direct_pair_input  = best_auc(X_pair_input,  y_pair)

    # Direct AUC — input + output
    direct_human_full  = best_auc(X_human_full, y_human)
    direct_gcg_full    = best_auc(X_gcg_full,   y_gcg)
    direct_pair_full   = best_auc(X_pair_full,  y_pair)

    logger.info(f"\n  Direct AUC input-only  — Human: {direct_human_input:.4f}, GCG: {direct_gcg_input:.4f}, PAIR: {direct_pair_input:.4f}")
    logger.info(f"  Direct AUC input+output — Human: {direct_human_full:.4f}, GCG: {direct_gcg_full:.4f}, PAIR: {direct_pair_full:.4f}")

    # ── Multi-seed classification ──────────────────────────────────────────────
    logger.info(f"\n=== Classification ({args.n_seeds} seeds) ===")

    gcg_aucs_input, pair_aucs_input, human_test_aucs_input = [], [], []
    gcg_aucs_full,  pair_aucs_full,  human_test_aucs_full  = [], [], []

    for seed in range(args.n_seeds):
        train_beh, train_tpl, test_beh, test_tpl = get_pool_split(
            human_valid, TRAIN_FRAC, seed
        )
        train_idx, test_idx = split_by_pool(
            human_valid, train_beh, train_tpl, test_beh, test_tpl
        )
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        # Input-only mode
        X_tr_i = X_human_input[train_idx]; y_tr = y_human[train_idx]
        X_te_i = X_human_input[test_idx];  y_te = y_human[test_idx]
        gcg_aucs_input.append(fit_eval(X_tr_i, y_tr, X_gcg_input,  y_gcg))
        pair_aucs_input.append(fit_eval(X_tr_i, y_tr, X_pair_input, y_pair))
        human_test_aucs_input.append(fit_eval(X_tr_i, y_tr, X_te_i, y_te))

        # Input+output mode
        X_tr_f = X_human_full[train_idx]
        X_te_f = X_human_full[test_idx]
        gcg_aucs_full.append(fit_eval(X_tr_f, y_tr, X_gcg_full,  y_gcg))
        pair_aucs_full.append(fit_eval(X_tr_f, y_tr, X_pair_full, y_pair))
        human_test_aucs_full.append(fit_eval(X_tr_f, y_tr, X_te_f, y_te))

        if seed % 10 == 0:
            logger.info(f"  Seed {seed} done")

    # ── Summary ────────────────────────────────────────────────────────────────
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  LLAMA GUARD 3  |  {args.guard_model}  |  {args.n_seeds} seeds")
    print(sep)

    print(f"\n  INPUT-ONLY MODE (pre-generation, same setting as activation classifier)")
    print(f"  {'Metric':35s}  {'Direct AUC':>10}  {'Logreg mean':>12}  {'Logreg std':>11}")
    print("  " + "─" * 72)
    print(f"  {'HarmBench':35s}  {direct_human_input:>10.4f}  {np.mean(human_test_aucs_input):>12.4f}  {np.std(human_test_aucs_input):>11.4f}")
    print(f"  {'→ GCG transfer':35s}  {direct_gcg_input:>10.4f}  {np.mean(gcg_aucs_input):>12.4f}  {np.std(gcg_aucs_input):>11.4f}")
    print(f"  {'→ PAIR transfer':35s}  {direct_pair_input:>10.4f}  {np.mean(pair_aucs_input):>12.4f}  {np.std(pair_aucs_input):>11.4f}")

    print(f"\n  INPUT+OUTPUT MODE (post-generation, sees model response)")
    print(f"  {'Metric':35s}  {'Direct AUC':>10}  {'Logreg mean':>12}  {'Logreg std':>11}")
    print("  " + "─" * 72)
    print(f"  {'HarmBench':35s}  {direct_human_full:>10.4f}  {np.mean(human_test_aucs_full):>12.4f}  {np.std(human_test_aucs_full):>11.4f}")
    print(f"  {'→ GCG transfer':35s}  {direct_gcg_full:>10.4f}  {np.mean(gcg_aucs_full):>12.4f}  {np.std(gcg_aucs_full):>11.4f}")
    print(f"  {'→ PAIR transfer':35s}  {direct_pair_full:>10.4f}  {np.mean(pair_aucs_full):>12.4f}  {np.std(pair_aucs_full):>11.4f}")

    print(f"\n  GCG chance:  {y_gcg.mean():.4f}")
    print(f"  PAIR chance: {y_pair.mean():.4f}")
    print(sep)

    out = {
        "guard_model": args.guard_model,
        "n_seeds": args.n_seeds,
        "gcg_chance":  float(y_gcg.mean()),
        "pair_chance": float(y_pair.mean()),
        "input_only": {
            "direct_auc": {"human": direct_human_input, "gcg": direct_gcg_input, "pair": direct_pair_input},
            "logreg": {
                "human_test":    {"mean": float(np.mean(human_test_aucs_input)), "std": float(np.std(human_test_aucs_input))},
                "transfer_gcg":  {"mean": float(np.mean(gcg_aucs_input)),        "std": float(np.std(gcg_aucs_input))},
                "transfer_pair": {"mean": float(np.mean(pair_aucs_input)),       "std": float(np.std(pair_aucs_input))},
            },
        },
        "input_output": {
            "direct_auc": {"human": direct_human_full, "gcg": direct_gcg_full, "pair": direct_pair_full},
            "logreg": {
                "human_test":    {"mean": float(np.mean(human_test_aucs_full)), "std": float(np.std(human_test_aucs_full))},
                "transfer_gcg":  {"mean": float(np.mean(gcg_aucs_full)),        "std": float(np.std(gcg_aucs_full))},
                "transfer_pair": {"mean": float(np.mean(pair_aucs_full)),       "std": float(np.std(pair_aucs_full))},
            },
        },
    }
    out_path = output_dir / "llama_guard_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
