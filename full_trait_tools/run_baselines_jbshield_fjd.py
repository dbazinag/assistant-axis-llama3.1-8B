#!/usr/bin/env python3
"""
run_baselines_jbshield_fjd.py

Two transfer-focused jailbreak detection baselines.

Baselines:
  1. JBShield-style representation direction:
     fit a benign->jailbroken mean direction on HarmBench train activations.

  2. FJD-style compliance likelihood:
     score each prompt by log P("Sure" | prompt), then calibrate a threshold on
     HarmBench train scores.

Evaluation:
  - Train/calibrate only on HarmBench human_jailbreak train pools.
  - Report held-out HarmBench validation performance.
  - Apply the same fitted detector unchanged to GCG, PAIR, PAP, GPTFuzz, PEZ.

Usage:
  uv run python full_trait_tools/run_baselines_jbshield_fjd.py
  uv run python full_trait_tools/run_baselines_jbshield_fjd.py --baseline jbshield
  uv run python full_trait_tools/run_baselines_jbshield_fjd.py --baseline fjd --test
"""

import argparse
import json
import logging
import random
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME             = "meta-llama/Llama-3.1-8B-Instruct"
COMPLIANCE_TARGET      = "Sure"
JAILBREAKS_URL         = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench"
                          "/main/baselines/human_jailbreaks/jailbreaks.py")
SKIP_JAILBREAK_INDICES = {16, 44}
HARMBENCH_ROOT         = "/dlabscratch1/bazina/HarmBench"
TRAIN_FRAC             = 0.7
N_SEEDS                = 50
RANDOM_SEED            = 42
LAYER                  = 16
TEST_ROWS_PER_DATASET  = 200

DATASETS = {
    "HarmBench": {
        "responses":   "full_trait_output/harmbench_activations/classified_responses.jsonl",
        "metadata":    "full_trait_output/harmbench_activations/pairs_metadata.jsonl",
        "activations": "full_trait_output/harmbench_activations/activations.pt",
        "test_cases":  None,
        "attack_type": "human_jailbreak",
    },
    "GCG": {
        "responses":   "full_trait_output/gcg_activations/responses.jsonl",
        "activations": "full_trait_output/gcg_activations/activations.pt",
        "test_cases":  f"{HARMBENCH_ROOT}/results/GCG/llama3_1_8b/test_cases/test_cases.json",
    },
    "PAIR": {
        "responses":   "full_trait_output/pair_activations/responses.jsonl",
        "activations": "full_trait_output/pair_activations/activations.pt",
        "test_cases":  f"{HARMBENCH_ROOT}/results/PAIR/llama3_1_8b/test_cases/test_cases.json",
    },
    "PAP": {
        "responses":   "full_trait_output/pap_activations/responses.jsonl",
        "activations": "full_trait_output/pap_activations/activations.pt",
        "test_cases":  f"{HARMBENCH_ROOT}/results/PAP/llama3_1_8b/test_cases/test_cases.json",
    },
    "GPTFuzz": {
        "responses":   "full_trait_output/gptfuzz_activations/responses.jsonl",
        "activations": "full_trait_output/gptfuzz_activations/activations.pt",
        "test_cases":  f"{HARMBENCH_ROOT}/results/GPTFuzz/llama3_1_8b/test_cases/test_cases.json",
    },
    "PEZ": {
        "responses":   "full_trait_output/pez_activations/responses.jsonl",
        "activations": "full_trait_output/pez_activations/activations.pt",
        "test_cases":  f"{HARMBENCH_ROOT}/results/PEZ/llama3_1_8b/test_cases/test_cases.json",
    },
}

MODEL_NAME_OLMO3 = "allenai/OLMo-3-7B-Instruct"

# OLMo-3 target responses + activations; attack-family prompts rebuilt from the
# per-behavior OLMo test-case files (exp subdir varies per family). GCG excluded
# (not yet collected for OLMo).
DATASETS_OLMO3 = {
    "HarmBench": {
        "responses":   "full_trait_output/harmbench_activations_olmo3/classified_responses.jsonl",
        "metadata":    "full_trait_output/harmbench_activations_olmo3/pairs_metadata.jsonl",
        "activations": "full_trait_output/harmbench_activations_olmo3/activations.pt",
        "test_cases":  None,
        "attack_type": "human_jailbreak",
    },
    "PAIR":    {"responses": "full_trait_output/pair_activations_olmo3_hb/classified_responses.jsonl",    "activations": "full_trait_output/pair_activations_olmo3_hb/activations.pt",    "exp": "olmo3_7b_harmbench"},
    "PAP":     {"responses": "full_trait_output/pap_activations_olmo3_hb/classified_responses.jsonl",     "activations": "full_trait_output/pap_activations_olmo3_hb/activations.pt",     "exp": "olmo3_7b_top5"},
    "GPTFuzz": {"responses": "full_trait_output/gptfuzz_activations_olmo3_hb/classified_responses.jsonl", "activations": "full_trait_output/gptfuzz_activations_olmo3_hb/activations.pt", "exp": "olmo3_7b"},
    "PEZ":     {"responses": "full_trait_output/pez_activations_olmo3_hb/classified_responses.jsonl",     "activations": "full_trait_output/pez_activations_olmo3_hb/activations.pt",     "exp": "olmo3_7b"},
}

# Gemma-4-31B-it. Single layer 30 was collected (shape 5376), so JBShield uses --layer 30.
MODEL_NAME_GEMMA = ("/mnt/dlabscratch1/bazina/.cache/huggingface/hub/"
                    "models--google--gemma-4-31B-it/snapshots/"
                    "fb9ae262347c3945692f09a612f8bb189def854f")
GEMMA_LAYER = 30


def _olmo3_to_gemma(cfg):
    """Gemma reuses the OLMo3 test_cases/exp; only the full_trait_output data paths change."""
    out = {}
    for name, c in cfg.items():
        c2 = dict(c)
        for k, v in c2.items():
            if isinstance(v, str) and "full_trait_output" in v:
                c2[k] = v.replace("_olmo3", "_gemma")
        out[name] = c2
    return out


DATASETS_GEMMA = _olmo3_to_gemma(DATASETS_OLMO3)


def load_olmo3_test_cases(attack_upper, exp):
    """Merge per-behavior test_cases.json files (OLMo attack-collector layout)."""
    ind_dir = (Path(HARMBENCH_ROOT) / "results" / f"HarmBench_{attack_upper}" / exp
               / "test_cases" / "test_cases_individual_behaviors")
    if not ind_dir.exists():
        raise FileNotFoundError(f"Not found: {ind_dir}")
    merged = {}
    for bdir in sorted(ind_dir.iterdir()):
        if not bdir.is_dir():
            continue
        tc_file = bdir / "test_cases.json"
        if not tc_file.exists():
            continue
        with tc_file.open() as f:
            d = json.load(f)
        for bid, prompts in d.items():
            flat = []
            for p in prompts:
                if isinstance(p, list):
                    flat.extend(p)
                elif isinstance(p, str):
                    flat.append(p)
            if flat:
                merged[bid] = merged.get(bid, []) + flat
    return merged


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fetch_jailbreaks() -> Dict[int, str]:
    with urllib.request.urlopen(JAILBREAKS_URL) as resp:
        source = resp.read().decode("utf-8")
    ns = {}
    exec(source, ns)  # noqa: S102
    raw = ns["JAILBREAKS"]
    valid = {}
    for i, jb in enumerate(raw):
        if i in SKIP_JAILBREAK_INDICES or "{0}" not in jb:
            continue
        valid[i] = jb
    return valid


def build_prompt_map_harmbench(metadata_path, responses_path, jailbreak_templates):
    meta = load_jsonl(Path(metadata_path))
    resp = load_jsonl(Path(responses_path))
    btext = {r["pair_id"]: r["behavior_text"] for r in resp}
    pm = {}
    for row in meta:
        pid = row["pair_id"]
        jb_idx = row["jailbreak_idx"]
        bt = row.get("behavior_text") or btext.get(pid, "")
        if jb_idx == -1:
            pm[pid] = bt
        elif jb_idx in jailbreak_templates:
            try:
                pm[pid] = jailbreak_templates[jb_idx].format(bt)
            except Exception:
                pm[pid] = bt
        else:
            pm[pid] = bt
    return pm


def build_prompt_map_generic(rows, tc):
    pm = {}
    behavior_counts = {}
    for row in rows:
        pid = row["pair_id"]
        bid = row["behavior_id"]
        behavior_counts.setdefault(bid, 0)
        prompts = tc.get(bid, [])
        flat = []
        for p in prompts:
            if isinstance(p, list):
                flat.extend(p)
            elif isinstance(p, str):
                flat.append(p)
        if flat:
            pm[pid] = flat[behavior_counts[bid] % len(flat)]
        else:
            pm[pid] = row.get("behavior_text", "")
        behavior_counts[bid] += 1
    return pm


def filter_rows(cfg: dict, rows: List[dict]) -> List[dict]:
    filter_type = cfg.get("attack_type")
    if filter_type:
        rows = [r for r in rows if r.get("attack_type") == filter_type]
    return rows


def test_subset(rows: List[dict], n: int = TEST_ROWS_PER_DATASET) -> List[dict]:
    if len(rows) <= n:
        return rows
    positives = [r for r in rows if r.get("jailbroken")]
    negatives = [r for r in rows if not r.get("jailbroken")]
    half = n // 2
    if len(positives) >= half and len(negatives) >= n - half:
        return positives[:half] + negatives[: n - half]
    return rows[:n]


def get_pool_split(rows, train_frac, seed):
    rng = random.Random(seed)
    all_behaviors = sorted({r["behavior_id"] for r in rows})
    all_templates = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(all_behaviors)
    rng.shuffle(all_templates)
    n_beh = max(1, int(len(all_behaviors) * train_frac))
    n_tpl = max(1, int(len(all_templates) * train_frac))
    return (
        set(all_behaviors[:n_beh]),
        set(all_templates[:n_tpl]),
        set(all_behaviors[n_beh:]),
        set(all_templates[n_tpl:]),
    )


def split_by_pool(rows, train_beh, train_tpl, val_beh, val_tpl):
    train_idx = [
        i for i, r in enumerate(rows)
        if r["behavior_id"] in train_beh and r["jailbreak_idx"] in train_tpl
    ]
    val_idx = [
        i for i, r in enumerate(rows)
        if r["behavior_id"] in val_beh and r["jailbreak_idx"] in val_tpl
    ]
    return train_idx, val_idx


def load_activations(path: Path) -> dict:
    logger.info(f"Loading activations: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def build_activation_matrix(rows: List[dict], activations: dict, layer: int):
    layer_key = str(layer)
    X_list, y_list, valid_rows = [], [], []
    for row in rows:
        pid = row["pair_id"]
        jb = row.get("jailbroken")
        if jb is None or pid not in activations or layer_key not in activations[pid]:
            continue
        X_list.append(activations[pid][layer_key].float().numpy())
        y_list.append(1 if jb else 0)
        valid_rows.append(row)
    return np.stack(X_list), np.array(y_list), valid_rows


def orient_scores(scores, y):
    if len(set(y)) < 2:
        return scores, 1
    auc = roc_auc_score(y, scores)
    if auc < 0.5:
        return -scores, -1
    return scores, 1


def best_threshold(scores, y):
    values = np.unique(scores)
    if len(values) > 200:
        values = np.quantile(scores, np.linspace(0.01, 0.99, 199))
    best_t, best_bacc = float(values[0]), -1.0
    for t in values:
        pred = (scores >= t).astype(int)
        bacc = balanced_accuracy_score(y, pred)
        if bacc > best_bacc:
            best_t = float(t)
            best_bacc = float(bacc)
    return best_t, best_bacc


def metrics(scores, y, threshold):
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


class MeanDirectionDetector:
    def fit(self, X, y):
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        mu_pos = Xs[y == 1].mean(axis=0)
        mu_neg = Xs[y == 0].mean(axis=0)
        w = mu_pos - mu_neg
        self.w = w / (np.linalg.norm(w) + 1e-12)
        return self

    def score(self, X):
        Xs = self.scaler.transform(X)
        return Xs @ self.w


def run_jbshield(rows_by_name, acts_by_name, args, wjb_X=None, wjb_y=None):
    logger.info("\n=== JBShield-style mean direction ===")
    X_by_name, y_by_name = {}, {}
    human_valid = None
    for name, rows in rows_by_name.items():
        X, y, valid_rows = build_activation_matrix(rows, acts_by_name[name], args.layer)
        X_by_name[name] = X
        y_by_name[name] = y
        if name == "HarmBench":
            human_valid = valid_rows
        logger.info(f"  {name}: X={X.shape}, jb={int(y.sum())}/{len(y)}")

    results = {name: {"auc": [], "ap": [], "balanced_acc": []} for name in rows_by_name}
    train_baccs = []

    for seed in range(args.n_seeds):
        train_beh, train_tpl, val_beh, val_tpl = get_pool_split(human_valid, TRAIN_FRAC, seed)
        train_idx, val_idx = split_by_pool(human_valid, train_beh, train_tpl, val_beh, val_tpl)
        if not train_idx or not val_idx:
            continue

        X_h = X_by_name["HarmBench"]
        y_h = y_by_name["HarmBench"]
        X_tr, y_tr = X_h[train_idx], y_h[train_idx]
        if wjb_X is not None:  # fold WJB into the calibration pool (HB+WJB)
            X_tr = np.vstack([X_tr, wjb_X])
            y_tr = np.concatenate([y_tr, wjb_y])
        det = MeanDirectionDetector().fit(X_tr, y_tr)
        train_scores = det.score(X_tr)
        train_scores, orient = orient_scores(train_scores, y_tr)
        threshold, train_bacc = best_threshold(train_scores, y_tr)
        train_baccs.append(train_bacc)

        for name in rows_by_name:
            if name == "HarmBench":
                X_eval = X_h[val_idx]
                y_eval = y_h[val_idx]
            else:
                X_eval = X_by_name[name]
                y_eval = y_by_name[name]
            scores = orient * det.score(X_eval)
            m = metrics(scores, y_eval, threshold)
            for k, v in m.items():
                results[name][k].append(v)

        if seed % 10 == 0:
            logger.info(f"  JBShield seed {seed} done")

    return {
        "train_balanced_acc": summarize(train_baccs),
        "datasets": {name: {k: summarize(v) for k, v in vals.items()}
                     for name, vals in results.items()},
    }


def compliance_logprob(model, tokenizer, prompt_text, device, max_length=2048):
    try:
        prompt_text = prompt_text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        conversation = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        full_text = formatted + COMPLIANCE_TARGET
        full_enc = tokenizer(
            full_text, return_tensors="pt", truncation=True,
            max_length=max_length, add_special_tokens=False).input_ids.to(device)
        prompt_enc = tokenizer(
            formatted, return_tensors="pt", truncation=True,
            max_length=max_length, add_special_tokens=False).input_ids.to(device)
        prompt_len = prompt_enc.shape[1]
        if full_enc.shape[1] <= prompt_len:
            return float("nan")
        labels = full_enc.clone()
        labels[:, :prompt_len] = -100
        with torch.no_grad():
            out = model(input_ids=full_enc, attention_mask=torch.ones_like(full_enc), labels=labels)
        n_target = int((labels != -100).sum().item())
        return float(-out.loss.item() * max(n_target, 1))
    except Exception as e:
        logger.warning(f"Compliance score error: {e}")
        return float("nan")


def score_prompts_fjd(rows_by_name, prompt_maps, args, wjb_rows=None, wjb_prompt_map=None):
    # WJB calibration prompts are scored alongside the eval families under a reserved
    # "__WJB__" key, then split back out so they only feed the per-seed train pool.
    score_targets = dict(rows_by_name)
    pm_targets = dict(prompt_maps)
    if wjb_rows is not None:
        score_targets["__WJB__"] = wjb_rows
        pm_targets["__WJB__"] = wjb_prompt_map

    cache_path = Path(args.output_dir) / "fjd_compliance_scores.json"
    if cache_path.exists() and not args.recompute_fjd:
        logger.info(f"Loading cached FJD scores: {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        cache_ok = all(
            name in cached and len(cached[name].get("scores", [])) == len(rows)
            for name, rows in score_targets.items()
        )
        if cache_ok:
            all_scores = {name: np.array(cached[name]["scores"], dtype=float) for name in cached}
            wjb_scores = all_scores.pop("__WJB__", None)
            return all_scores, wjb_scores
        logger.warning("Cached FJD scores do not match current dataset sizes; recomputing.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    remote = bool(getattr(args, "olmo3", False) or getattr(args, "gemma", False))
    logger.info(f"Loading {args.model} for FJD-style compliance scores...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=remote, trust_remote_code=remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(args, "gemma", False):
        # Gemma-4-31B needs sdpa + chunked attention; single-device bf16 (~62GB).
        from chunked_sdpa import chunked_sdpa_scope
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map="cuda:0", trust_remote_code=True)
        scope_cm = chunked_sdpa_scope
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map={"": device}, trust_remote_code=remote)
        scope_cm = nullcontext
    model.eval()

    scores_by_name = {}
    serializable = {}
    with scope_cm():
        for name, rows in score_targets.items():
            prompt_map = pm_targets[name]
            scores = []
            logger.info(f"  FJD scoring {name}: {len(rows)} rows")
            for i, row in enumerate(rows):
                if i % 100 == 0:
                    logger.info(f"    {name} {i}/{len(rows)}")
                prompt = prompt_map.get(row["pair_id"], "")
                scores.append(compliance_logprob(model, tokenizer, prompt, device) if prompt else float("nan"))
            arr = np.array(scores, dtype=float)
            scores_by_name[name] = arr
            serializable[name] = {"scores": arr.tolist()}

    with open(cache_path, "w") as f:
        json.dump(serializable, f)
    logger.info(f"Saved FJD scores to {cache_path}")
    wjb_scores = scores_by_name.pop("__WJB__", None)
    return scores_by_name, wjb_scores


def run_fjd(rows_by_name, prompt_maps, args, wjb_rows=None, wjb_prompt_map=None):
    logger.info("\n=== FJD-style compliance likelihood ===")
    scores_by_name, wjb_scores = score_prompts_fjd(rows_by_name, prompt_maps, args,
                                                   wjb_rows=wjb_rows, wjb_prompt_map=wjb_prompt_map)
    wjb_y = (np.array([1 if r.get("jailbroken") else 0 for r in wjb_rows], dtype=int)
             if wjb_rows is not None else None)
    y_by_name = {
        name: np.array([1 if r.get("jailbroken") else 0 for r in rows], dtype=int)
        for name, rows in rows_by_name.items()
    }
    human_rows = rows_by_name["HarmBench"]
    results = {name: {"auc": [], "ap": [], "balanced_acc": []} for name in rows_by_name}
    train_baccs = []

    for seed in range(args.n_seeds):
        train_beh, train_tpl, val_beh, val_tpl = get_pool_split(human_rows, TRAIN_FRAC, seed)
        train_idx, val_idx = split_by_pool(human_rows, train_beh, train_tpl, val_beh, val_tpl)
        if not train_idx or not val_idx:
            continue

        s_train_raw = scores_by_name["HarmBench"][train_idx]
        y_train = y_by_name["HarmBench"][train_idx]
        ok = np.isfinite(s_train_raw)
        s_train = s_train_raw[ok]
        y_train = y_train[ok]
        if wjb_scores is not None:  # fold WJB into the calibration pool (HB+WJB)
            wok = np.isfinite(wjb_scores)
            s_train = np.concatenate([s_train, wjb_scores[wok]])
            y_train = np.concatenate([y_train, wjb_y[wok]])
        s_train, orient = orient_scores(s_train, y_train)
        threshold, train_bacc = best_threshold(s_train, y_train)
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
            logger.info(f"  FJD seed {seed} done")

    return {
        "train_balanced_acc": summarize(train_baccs),
        "datasets": {name: {k: summarize(v) for k, v in vals.items()}
                     for name, vals in results.items()},
    }


def print_summary(results):
    sep = "=" * 100
    print(f"\n{sep}")
    print("  JBSHIELD / FJD TRANSFER BASELINES")
    print("  Train/calibrate: HarmBench human_jailbreak train pools only")
    print(sep)
    for baseline, data in results.items():
        print(f"\n  {baseline}")
        print(f"  Train balanced acc: {data['train_balanced_acc']['mean']:.4f} ± "
              f"{data['train_balanced_acc']['std']:.4f}")
        print(f"  {'Family':12s}  {'AUC':>8}  {'AP':>8}  {'BAcc':>8}")
        print("  " + "-" * 42)
        for name, vals in data["datasets"].items():
            print(f"  {name:12s}  {vals['auc']['mean']:>8.4f}  "
                  f"{vals['ap']['mean']:>8.4f}  {vals['balanced_acc']['mean']:>8.4f}")
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["both", "jbshield", "fjd"], default="both")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output_dir", default="full_trait_output/baselines_jbshield_fjd")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recompute_fjd", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--olmo3", action="store_true",
                        help="Use OLMo-3-7B + OLMo activations/responses (GCG excluded; not yet collected)")
    parser.add_argument("--gemma", action="store_true",
                        help="Use Gemma-4-31B + Gemma activations/responses (GCG excluded; layer 30)")
    parser.add_argument("--augment_wjb", action="store_true",
                        help="Fold WildJailbreak into the per-seed calibration pool (HB+WJB) for "
                             "JBShield (acts) and FJD (compliance score); eval families unchanged. "
                             "Output dir gets a _hbwjb suffix. Gemma/layer-30 only.")
    args = parser.parse_args()

    if args.gemma:
        if args.model == MODEL_NAME:
            args.model = MODEL_NAME_GEMMA
        if args.output_dir == "full_trait_output/baselines_jbshield_fjd":
            args.output_dir = "full_trait_output/baselines_jbshield_fjd_gemma"
        if args.layer == LAYER:
            args.layer = GEMMA_LAYER
    if args.augment_wjb:
        args.output_dir = args.output_dir + "_hbwjb"
    elif args.olmo3:
        if args.model == MODEL_NAME:
            args.model = MODEL_NAME_OLMO3
        if args.output_dir == "full_trait_output/baselines_jbshield_fjd":
            args.output_dir = "full_trait_output/baselines_jbshield_fjd_olmo3"
    datasets = DATASETS_GEMMA if args.gemma else (DATASETS_OLMO3 if args.olmo3 else DATASETS)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading rows and prompt maps...")
    jailbreak_templates = fetch_jailbreaks()
    rows_by_name = {}
    prompt_maps = {}
    for name, cfg in datasets.items():
        rows_path = Path(cfg["responses"])
        if not rows_path.exists():
            logger.warning(f"Skipping {name}: missing {rows_path}")
            continue
        rows = filter_rows(cfg, load_jsonl(rows_path))
        if args.test:
            rows = test_subset(rows)
        rows_by_name[name] = rows
        if name == "HarmBench":
            prompt_maps[name] = build_prompt_map_harmbench(
                cfg["metadata"], cfg["responses"], jailbreak_templates)
        elif args.olmo3 or args.gemma:
            prompt_maps[name] = build_prompt_map_generic(rows, load_olmo3_test_cases(name, cfg["exp"]))
        else:
            prompt_maps[name] = build_prompt_map_generic(rows, json.load(open(cfg["test_cases"])))
        logger.info(f"  {name}: {len(rows)} rows")

    # WJB calibration channel (HB+WJB): rows for FJD scoring + acts for JBShield. Folded
    # into the per-seed TRAIN pool only; never added to rows_by_name (would hit eval loop).
    wjb_rows = wjb_prompt_map = None
    if args.augment_wjb:
        wjb_rows = load_jsonl(Path("full_trait_output/wildjailbreak_activations_gemma/classified_responses.jsonl"))
        if args.test:
            wjb_rows = wjb_rows[:20]
        wjb_prompt_map = {r["pair_id"]: r["behavior_text"] for r in wjb_rows}
        logger.info(f"  WJB calibration augment: {len(wjb_rows)} rows")

    results = {}
    if args.baseline in ("both", "jbshield"):
        acts_by_name = {}
        for name in rows_by_name:
            acts_path = Path(datasets[name]["activations"])
            if not acts_path.exists():
                raise FileNotFoundError(f"Missing activations for {name}: {acts_path}")
            acts_by_name[name] = load_activations(acts_path)
        wjb_X = wjb_y = None
        if args.augment_wjb:
            wjb_acts = load_activations(Path("full_trait_output/wildjailbreak_activations_gemma/activations.pt"))
            wjb_X, wjb_y, _ = build_activation_matrix(wjb_rows, wjb_acts, args.layer)
            logger.info(f"  JBShield WJB augment: X={wjb_X.shape}, jb={int(wjb_y.sum())}/{len(wjb_y)}")
        results["jbshield_mean_direction"] = run_jbshield(rows_by_name, acts_by_name, args,
                                                          wjb_X=wjb_X, wjb_y=wjb_y)

    if args.baseline in ("both", "fjd"):
        results["fjd_sure_logprob"] = run_fjd(rows_by_name, prompt_maps, args,
                                              wjb_rows=wjb_rows, wjb_prompt_map=wjb_prompt_map)

    print_summary(results)
    out_path = output_dir / "jbshield_fjd_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "layer": args.layer,
            "n_seeds": args.n_seeds,
            "train_source": ("HarmBench human_jailbreak + WildJailbreak (HB+WJB) train pools"
                             if args.augment_wjb else "HarmBench human_jailbreak train pools only"),
            "results": results,
        }, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
