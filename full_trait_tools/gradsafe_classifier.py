#!/usr/bin/env python3
"""
gradsafe_classifier.py

Baseline jailbreak detector using GradSafe (Xie et al., ACL 2024).

GradSafe works by computing gradients of the model loss on a fixed compliance
target ("Sure, here is") given each candidate prompt, then measuring cosine
similarity between these gradients and reference gradients from known
jailbreak/benign prompts. High similarity to jailbreak references → flagged.

Key insight: jailbreak prompts paired with a compliance target produce
gradient signatures that cluster together in parameter space, distinct from
benign prompts.

Implementation:
  - Compliance target: "Sure, here is" (same as original paper)
  - Gradient over: all parameters (flattened, then compared via cosine sim)
  - Reference set: small sample from HarmBench training split
  - Score: cos_sim(grad, jb_ref_grad) - cos_sim(grad, benign_ref_grad)
  - Same strict pool split regime as fast_transfer_classifier.py

Note: gradient vectors are large (~7B params for Llama-3.1-8B). We use
gradients w.r.t. a fixed target subset (last 4 transformer layers MLP weights)
for efficiency — same spirit as the paper's safety-critical parameter selection.

Usage:
  uv run python full_trait_tools/gradsafe_classifier.py
  uv run python full_trait_tools/gradsafe_classifier.py --test
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
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
JAILBREAKS_URL         = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/baselines/human_jailbreaks/jailbreaks.py"
)
SKIP_JAILBREAK_INDICES = {16, 44}
HARMBENCH_SEED         = 42
TRAIN_FRAC             = 0.7
N_SEEDS                = 50
RANDOM_SEED            = 42
MODEL_NAME             = "meta-llama/Llama-3.1-8B-Instruct"
COMPLIANCE_TARGET      = "Sure, here is"
N_REFERENCE_EACH       = 16   # reference jailbreak and benign prompts each
# Use last N transformer layers for gradient computation (efficiency)
GRAD_LAYERS            = list(range(28, 32))  # layers 28-31 (last 4)


# ── Prompt reconstruction (identical to perplexity_classifier.py) ──────────────

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
        prompt_map[pid] = test_cases[bid][0] if bid in test_cases and test_cases[bid] else row.get("behavior_text", "")
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


# ── GradSafe core ──────────────────────────────────────────────────────────────

def get_target_params(model) -> List[torch.nn.Parameter]:
    """
    Select parameters from last GRAD_LAYERS transformer layers.
    These are the "safety-critical parameters" proxy.
    Returns list of parameter tensors.
    """
    params = []
    for layer_idx in GRAD_LAYERS:
        layer = model.model.layers[layer_idx]
        # MLP weights (gate, up, down projections)
        for name, param in layer.mlp.named_parameters():
            if param.requires_grad:
                params.append(param)
    return params


def compute_gradient_vector(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    target_params: List[torch.nn.Parameter],
    device: torch.device,
) -> Optional[np.ndarray]:
    """
    Compute gradient of loss on COMPLIANCE_TARGET given prompt_text.
    Returns flattened gradient vector over target_params, or None on error.
    """
    try:
        prompt_text = prompt_text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        if not prompt_text.strip():
            return None

        # Format prompt with chat template
        conversation = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        # Append compliance target
        full_text = formatted + COMPLIANCE_TARGET

        # Tokenize full text
        full_enc = tokenizer(full_text, return_tensors="pt",
                             add_special_tokens=False).input_ids.to(device)
        # Tokenize prompt only (to know where target starts)
        prompt_enc = tokenizer(formatted, return_tensors="pt",
                               add_special_tokens=False).input_ids.to(device)
        prompt_len = prompt_enc.shape[1]

        if full_enc.shape[1] <= prompt_len:
            return None

        # Labels: -100 for prompt tokens, actual token ids for target tokens
        labels = full_enc.clone()
        labels[0, :prompt_len] = -100

        attention_mask = torch.ones_like(full_enc)

        # Zero existing gradients
        model.zero_grad()

        outputs = model(
            input_ids=full_enc,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        # Collect and flatten gradients
        grad_parts = []
        for param in target_params:
            if param.grad is not None:
                grad_parts.append(param.grad.detach().float().cpu().flatten())

        model.zero_grad()
        torch.cuda.empty_cache()

        if not grad_parts:
            return None

        grad_vec = torch.cat(grad_parts).numpy()
        # Normalize
        norm = np.linalg.norm(grad_vec)
        if norm < 1e-10:
            return None
        return grad_vec / norm

    except Exception as e:
        logger.warning(f"Gradient error: {e}")
        model.zero_grad()
        torch.cuda.empty_cache()
        return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both already unit-normalized


def gradsafe_score(
    grad: np.ndarray,
    jb_refs: List[np.ndarray],
    benign_refs: List[np.ndarray],
) -> float:
    """
    GradSafe score = mean cos_sim with jailbreak refs - mean cos_sim with benign refs.
    Higher → more jailbreak-like gradient signature.
    """
    jb_sim     = np.mean([cosine_sim(grad, r) for r in jb_refs])
    benign_sim = np.mean([cosine_sim(grad, r) for r in benign_refs])
    return float(jb_sim - benign_sim)


# ── Pool split (identical to fast_transfer_classifier.py) ─────────────────────

def get_pool_split(rows, train_frac, seed):
    rng = random.Random(seed)
    all_behaviors = sorted({r["behavior_id"]  for r in rows})
    all_templates = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(all_behaviors)
    rng.shuffle(all_templates)
    n_beh = max(1, int(len(all_behaviors) * train_frac))
    n_tpl = max(1, int(len(all_templates) * train_frac))
    return (set(all_behaviors[:n_beh]), set(all_templates[:n_tpl]),
            set(all_behaviors[n_beh:]),  set(all_templates[n_tpl:]))


def split_by_pool(rows, train_beh, train_tpl, test_beh, test_tpl):
    train_idx = [i for i, r in enumerate(rows)
                 if r["behavior_id"] in train_beh and r["jailbreak_idx"] in train_tpl]
    test_idx  = [i for i, r in enumerate(rows)
                 if r["behavior_id"] in test_beh  and r["jailbreak_idx"] in test_tpl]
    return train_idx, test_idx


def direct_auc(scores: np.ndarray, y: np.ndarray) -> float:
    if len(set(y)) < 2:
        return float("nan")
    a = roc_auc_score(y, scores)
    return max(a, 1 - a)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_classified_path", default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_metadata_path",   default="full_trait_output/harmbench_activations/pairs_metadata.jsonl")
    parser.add_argument("--gcg_classified_path",   default="full_trait_output/gcg_activations/responses.jsonl")
    parser.add_argument("--gcg_test_cases_path",   default="/dlabscratch1/bazina/HarmBench/results/GCG/llama3_1_8b/test_cases/test_cases.json")
    parser.add_argument("--pair_classified_path",  default="full_trait_output/pair_activations/responses.jsonl")
    parser.add_argument("--pair_test_cases_path",  default="/dlabscratch1/bazina/HarmBench/results/PAIR/llama3_1_8b/test_cases/test_cases.json")
    parser.add_argument("--output_dir",            default="full_trait_output/gradsafe_baseline")
    parser.add_argument("--model",                 default=MODEL_NAME)
    parser.add_argument("--n_seeds",  type=int,    default=N_SEEDS)
    parser.add_argument("--n_ref",    type=int,    default=N_REFERENCE_EACH,
                        help="Number of reference prompts per class")
    parser.add_argument("--ref_seed", type=int,    default=0,
                        help="Seed for reference prompt selection")
    parser.add_argument("--device",                default="cuda")
    parser.add_argument("--test",     action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.test:
        output_dir = output_dir.parent / (output_dir.name + "_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model ─────────────────────────────────────────────────────────────
    logger.info(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()
    # Need gradients for GradSafe
    for param in model.parameters():
        param.requires_grad_(True)
    logger.info("Model loaded.")

    target_params = get_target_params(model)
    grad_dim = sum(p.numel() for p in target_params)
    logger.info(f"  Gradient params: {len(target_params)} tensors, {grad_dim:,} total dims")

    # ── Reconstruct prompts ────────────────────────────────────────────────────
    logger.info("\n=== Reconstructing prompts ===")
    jailbreak_templates = fetch_jailbreaks()
    human_prompt_map = build_harmbench_prompt_map(
        Path(args.human_metadata_path), Path(args.human_classified_path), jailbreak_templates)
    gcg_prompt_map  = build_gcg_prompt_map(Path(args.gcg_classified_path),  Path(args.gcg_test_cases_path))
    pair_prompt_map = build_pair_prompt_map(Path(args.pair_classified_path), Path(args.pair_test_cases_path))

    # ── Load metadata ──────────────────────────────────────────────────────────
    logger.info("\n=== Loading metadata ===")
    human_rows_all = load_jsonl(Path(args.human_classified_path))
    human_rows = [r for r in human_rows_all if r.get("attack_type") == "human_jailbreak"]
    gcg_rows   = load_jsonl(Path(args.gcg_classified_path))
    pair_rows  = load_jsonl(Path(args.pair_classified_path))

    if args.test:
        bids = list({r["behavior_id"] for r in human_rows})[:3]
        human_rows = [r for r in human_rows if r["behavior_id"] in bids]
        gcg_rows   = gcg_rows[:10]
        pair_rows  = pair_rows[:10]

    # ── Select reference prompts (from full HarmBench, fixed seed) ─────────────
    logger.info(f"\n=== Selecting {args.n_ref} reference prompts per class ===")
    rng_ref = random.Random(args.ref_seed)
    jb_candidates     = [r for r in human_rows if r.get("jailbroken") and r["pair_id"] in human_prompt_map]
    benign_candidates = [r for r in human_rows if not r.get("jailbroken") and r["pair_id"] in human_prompt_map]

    n_ref = min(args.n_ref, len(jb_candidates), len(benign_candidates))
    jb_ref_rows     = rng_ref.sample(jb_candidates,     n_ref)
    benign_ref_rows = rng_ref.sample(benign_candidates, n_ref)
    logger.info(f"  Selected {n_ref} jailbreak refs, {n_ref} benign refs")

    logger.info("Computing reference gradients...")
    jb_ref_grads, benign_ref_grads = [], []

    for row in jb_ref_rows:
        g = compute_gradient_vector(model, tokenizer, human_prompt_map[row["pair_id"]], target_params, device)
        if g is not None:
            jb_ref_grads.append(g)

    for row in benign_ref_rows:
        g = compute_gradient_vector(model, tokenizer, human_prompt_map[row["pair_id"]], target_params, device)
        if g is not None:
            benign_ref_grads.append(g)

    logger.info(f"  Got {len(jb_ref_grads)} jb refs, {len(benign_ref_grads)} benign refs")
    if len(jb_ref_grads) == 0 or len(benign_ref_grads) == 0:
        logger.error("No valid reference gradients — exiting")
        return

    # ── Compute GradSafe scores for all datasets ───────────────────────────────
    def score_dataset(rows, prompt_map, desc):
        scores_list, y_list, valid_rows = [], [], []
        for i, row in enumerate(rows):
            if i % 50 == 0:
                logger.info(f"  {desc} {i}/{len(rows)}")
            pid = row["pair_id"]
            jb  = row.get("jailbroken")
            if jb is None:
                continue
            prompt = prompt_map.get(pid)
            if not prompt or not prompt.strip():
                continue
            grad = compute_gradient_vector(model, tokenizer, prompt, target_params, device)
            if grad is None:
                continue
            score = gradsafe_score(grad, jb_ref_grads, benign_ref_grads)
            scores_list.append(score)
            y_list.append(1 if jb else 0)
            valid_rows.append(row)
        if not scores_list:
            return np.array([]), np.array([]), []
        return np.array(scores_list), np.array(y_list), valid_rows

    logger.info("\n=== Computing GradSafe scores ===")
    logger.info("HarmBench...")
    scores_human, y_human, human_valid = score_dataset(human_rows, human_prompt_map, "HarmBench")
    logger.info(f"  {len(y_human)} pairs, {y_human.sum():.0f} jailbroken")

    logger.info("GCG...")
    scores_gcg, y_gcg, _ = score_dataset(gcg_rows, gcg_prompt_map, "GCG")
    logger.info(f"  {len(y_gcg)} pairs, {y_gcg.sum():.0f} jailbroken")

    logger.info("PAIR...")
    scores_pair, y_pair, _ = score_dataset(pair_rows, pair_prompt_map, "PAIR")
    logger.info(f"  {len(y_pair)} pairs, {y_pair.sum():.0f} jailbroken")

    # ── Direct AUC ─────────────────────────────────────────────────────────────
    da_human = direct_auc(scores_human, y_human)
    da_gcg   = direct_auc(scores_gcg,   y_gcg)
    da_pair  = direct_auc(scores_pair,  y_pair)
    logger.info(f"\n  Direct AUC — Human: {da_human:.4f}, GCG: {da_gcg:.4f}, PAIR: {da_pair:.4f}")

    # ── Multi-seed logreg ──────────────────────────────────────────────────────
    logger.info(f"\n=== Cross-family logreg ({args.n_seeds} seeds) ===")
    gcg_aucs, pair_aucs, human_test_aucs = [], [], []

    X_human = scores_human.reshape(-1, 1)
    X_gcg   = scores_gcg.reshape(-1, 1)
    X_pair  = scores_pair.reshape(-1, 1)

    for seed in range(args.n_seeds):
        train_beh, train_tpl, test_beh, test_tpl = get_pool_split(human_valid, TRAIN_FRAC, seed)
        train_idx, test_idx = split_by_pool(human_valid, train_beh, train_tpl, test_beh, test_tpl)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        X_tr = X_human[train_idx]; y_tr = y_human[train_idx]
        X_te = X_human[test_idx];  y_te = y_human[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        X_gcg_s  = scaler.transform(X_gcg)
        X_pair_s = scaler.transform(X_pair)

        clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
        clf.fit(X_tr_s, y_tr)

        def auc_if_possible(X, y):
            if len(set(y)) < 2: return float("nan")
            return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))

        gcg_aucs.append(auc_if_possible(X_gcg_s,  y_gcg))
        pair_aucs.append(auc_if_possible(X_pair_s, y_pair))
        human_test_aucs.append(auc_if_possible(X_te_s, y_te))

        if seed % 10 == 0:
            logger.info(f"  Seed {seed} done")

    # ── Summary ────────────────────────────────────────────────────────────────
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  GRADSAFE BASELINE  |  Llama-3.1-8B  |  {args.n_seeds} seeds  |  {n_ref} refs each")
    print(sep)
    print(f"\n  DIRECT AUC (no training, threshold score directly):")
    print(f"  {'HarmBench':30s}  {da_human:>8.4f}")
    print(f"  {'→ GCG':30s}  {da_gcg:>8.4f}")
    print(f"  {'→ PAIR':30s}  {da_pair:>8.4f}")
    print(f"\n  CROSS-FAMILY LOGREG (train HarmBench → test *):")
    print(f"  {'Metric':30s}  {'Mean':>8}  {'Std':>8}")
    print("  " + "─" * 50)
    print(f"  {'→ HarmBench test':30s}  {np.mean(human_test_aucs):>8.4f}  {np.std(human_test_aucs):>8.4f}")
    print(f"  {'→ GCG transfer':30s}  {np.mean(gcg_aucs):>8.4f}  {np.std(gcg_aucs):>8.4f}")
    print(f"  {'→ PAIR transfer':30s}  {np.mean(pair_aucs):>8.4f}  {np.std(pair_aucs):>8.4f}")
    print(f"\n  GCG chance:  {y_gcg.mean():.4f}")
    print(f"  PAIR chance: {y_pair.mean():.4f}")
    print(sep)

    out = {
        "model": args.model, "n_seeds": args.n_seeds, "n_ref": n_ref,
        "gcg_chance": float(y_gcg.mean()), "pair_chance": float(y_pair.mean()),
        "direct_auc": {"human": da_human, "gcg": da_gcg, "pair": da_pair},
        "cross_family_logreg": {
            "human_test":    {"mean": float(np.mean(human_test_aucs)), "std": float(np.std(human_test_aucs))},
            "transfer_gcg":  {"mean": float(np.mean(gcg_aucs)),        "std": float(np.std(gcg_aucs))},
            "transfer_pair": {"mean": float(np.mean(pair_aucs)),       "std": float(np.std(pair_aucs))},
        }
    }
    with open(output_dir / "gradsafe_results.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved to {output_dir}/gradsafe_results.json")


if __name__ == "__main__":
    main()
