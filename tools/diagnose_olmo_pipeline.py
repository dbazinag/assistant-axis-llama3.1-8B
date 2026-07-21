#!/usr/bin/env python3
"""
Diagnose why OLMo trait features underperform raw activations.

Run on cluster:
  uv run python tools/diagnose_olmo_pipeline.py

Checks:
  1. Attack transfer-set label distributions (are jailbroken labels real?)
  2. pair_id consistency between JSONL and activations.pt
  3. OLMo vs Llama trait matrix shape + effective rank
  4. Centroid-AUC: how well do trait projections vs raw activations separate
     jailbroken from non-jailbroken rows in EACH dataset
     (uses only dot-product with mean-difference direction, no training)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

BASE = Path("full_trait_output")
LAYER = 16


# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_auc(y, s) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    a = float(roc_auc_score(y, s))
    return max(a, 1.0 - a)


def load_acts(path: Path) -> dict:
    print(f"  loading {path.name} ({path.stat().st_size / 1e6:.0f} MB) ...", flush=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_matrix(mat_path: Path) -> np.ndarray | None:
    if not mat_path.exists():
        return None
    m = np.load(mat_path).astype(np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


# ── label-distribution check ──────────────────────────────────────────────────

def check_labels(name: str, jsonl_path: Path) -> dict:
    if not jsonl_path.exists():
        print(f"  {name}: MISSING {jsonl_path}")
        return {}
    rows = load_jsonl(jsonl_path)
    total = len(rows)
    n_jb = sum(1 for r in rows if r.get("jailbroken") is True)
    n_notjb = sum(1 for r in rows if r.get("jailbroken") is False)
    n_none = total - n_jb - n_notjb
    # Were rows actually judged by GPT? (judge adds llm_judge_raw field)
    n_judged = sum(1 for r in rows if r.get("llm_judge_raw") is not None)
    print(
        f"  {name:40s}  n={total:5d}  jb={n_jb:5d} ({100*n_jb/total:5.1f}%)  "
        f"not_jb={n_notjb:5d}  none={n_none}  gpt_judged={n_judged}"
    )
    return {"total": total, "n_jb": n_jb, "n_judged": n_judged}


# ── pair_id consistency check ─────────────────────────────────────────────────

def check_pair_ids(name: str, jsonl_path: Path, acts_path: Path) -> None:
    if not jsonl_path.exists() or not acts_path.exists():
        print(f"  {name}: MISSING files, skipping pair_id check")
        return
    rows = load_jsonl(jsonl_path)
    acts = torch.load(acts_path, map_location="cpu", weights_only=False)
    layer_key = str(LAYER)
    matched = sum(
        1 for r in rows
        if acts.get(r["pair_id"]) is not None
        and layer_key in acts.get(r["pair_id"], {})
    )
    print(f"  {name:40s}  rows={len(rows)}  acts_keys={len(acts)}  matched_at_L{LAYER}={matched}")


# ── effective rank of a matrix ────────────────────────────────────────────────

def effective_rank(matrix: np.ndarray) -> int:
    sv = np.linalg.svd(matrix, compute_uv=False)
    return int((sv > sv[0] * 0.01).sum())


# ── centroid AUC: raw vs trait ────────────────────────────────────────────────

def centroid_auc(name: str, jsonl_path: Path, acts_path: Path,
                 matrix: np.ndarray | None) -> None:
    if not jsonl_path.exists() or not acts_path.exists():
        print(f"  {name}: files missing")
        return
    rows = [r for r in load_jsonl(jsonl_path) if r.get("jailbroken") is not None]
    if not rows:
        print(f"  {name}: no labelled rows")
        return

    acts = load_acts(acts_path)
    layer_key = str(LAYER)
    xs_raw, ys = [], []
    for r in rows:
        item = acts.get(r["pair_id"])
        if item is None or layer_key not in item:
            continue
        xs_raw.append(item[layer_key].float().numpy())
        ys.append(1 if r["jailbroken"] else 0)

    if len(ys) < 20 or len(set(ys)) < 2:
        print(f"  {name}: n={len(ys)} or single-class, skipping AUC")
        return

    xs = np.stack(xs_raw).astype(np.float32)
    ys = np.array(ys)
    pos = ys == 1
    neg = ys == 0

    # Raw centroid direction
    raw_dir = xs[pos].mean(0) - xs[neg].mean(0)
    raw_auc = safe_auc(ys, xs @ raw_dir)

    if matrix is not None:
        proj = xs @ matrix.T  # [N, n_traits]
        trait_dir = proj[pos].mean(0) - proj[neg].mean(0)
        trait_auc = safe_auc(ys, proj @ trait_dir)
        print(
            f"  {name:40s}  n={len(ys):5d}  jb={pos.sum():5d}  "
            f"raw_centroid_AUC={raw_auc:.4f}  trait_centroid_AUC={trait_auc:.4f}"
        )
    else:
        print(f"  {name:40s}  n={len(ys):5d}  jb={pos.sum():5d}  raw_centroid_AUC={raw_auc:.4f}  (no trait matrix)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    sep = "=" * 80

    # ── 1. Label distributions ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  1. LABEL DISTRIBUTIONS (are jailbroken labels real?)")
    print(sep)
    datasets = [
        # OLMo training
        ("OLMo HarmBench (train)",
         BASE / "harmbench_activations_olmo3/classified_responses.jsonl"),
        # OLMo attack transfer sets
        ("OLMo PAIR",
         BASE / "pair_activations_olmo3_hb/classified_responses.jsonl"),
        ("OLMo PAP",
         BASE / "pap_activations_olmo3_hb/classified_responses.jsonl"),
        ("OLMo GPTFuzz",
         BASE / "gptfuzz_activations_olmo3_hb/classified_responses.jsonl"),
        ("OLMo PEZ",
         BASE / "pez_activations_olmo3_hb/classified_responses.jsonl"),
        # Llama training
        ("Llama HarmBench (train)",
         BASE / "harmbench_activations/classified_responses.jsonl"),
        # Llama attack transfer sets (if they exist)
        ("Llama PAIR",
         BASE / "pair_activations/classified_responses.jsonl"),
        ("Llama PAP",
         BASE / "pap_activations/classified_responses.jsonl"),
        ("Llama GPTFuzz",
         BASE / "gptfuzz_activations/classified_responses.jsonl"),
        ("Llama PEZ",
         BASE / "pez_activations/classified_responses.jsonl"),
    ]
    for name, path in datasets:
        check_labels(name, path)

    # ── 2. pair_id consistency ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  2. PAIR_ID CONSISTENCY (JSONL ↔ activations.pt)")
    print(sep)
    consistency_checks = [
        ("OLMo HarmBench",
         BASE / "harmbench_activations_olmo3/classified_responses.jsonl",
         BASE / "harmbench_activations_olmo3/activations.pt"),
        ("OLMo PAIR",
         BASE / "pair_activations_olmo3_hb/classified_responses.jsonl",
         BASE / "pair_activations_olmo3_hb/activations.pt"),
        ("OLMo PAP",
         BASE / "pap_activations_olmo3_hb/classified_responses.jsonl",
         BASE / "pap_activations_olmo3_hb/activations.pt"),
        ("OLMo GPTFuzz",
         BASE / "gptfuzz_activations_olmo3_hb/classified_responses.jsonl",
         BASE / "gptfuzz_activations_olmo3_hb/activations.pt"),
        ("OLMo PEZ",
         BASE / "pez_activations_olmo3_hb/classified_responses.jsonl",
         BASE / "pez_activations_olmo3_hb/activations.pt"),
    ]
    for name, jp, ap in consistency_checks:
        check_pair_ids(name, jp, ap)

    # ── 3. Trait matrix quality ────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  3. TRAIT MATRIX QUALITY")
    print(sep)
    matrices_to_check = [
        ("Llama trait (cached, all-traits)",
         BASE / f"trait_matrix_layer{LAYER}.npy"),
        ("OLMo trait (all-traits sweep cache)",
         BASE / f"all_traits_sweep_v2_olmo3_all_traits/trait_matrix_layer{LAYER}.npy"),
        ("OLMo trait (corrected sweep cache)",
         BASE / f"all_traits_sweep_v2_olmo3_corrected/trait_matrix_layer{LAYER}.npy"),
    ]
    for name, path in matrices_to_check:
        if not path.exists():
            print(f"  {name:45s}  MISSING")
            continue
        m = np.load(path).astype(np.float32)
        norms = np.linalg.norm(m, axis=1)
        erank = effective_rank(m)
        print(
            f"  {name:45s}  shape={m.shape}  "
            f"norm_mean={norms.mean():.4f}  norm_std={norms.std():.4f}  "
            f"effective_rank={erank}"
        )

    # ── 4. Centroid AUC: raw vs trait ──────────────────────────────────────────
    print(f"\n{sep}")
    print("  4. CENTROID AUC — raw 4096-dim vs trait-projected (layer 16)")
    print("     (centroid direction = mean(jailbroken) - mean(not jailbroken))")
    print("     Higher = better linear separability in that feature space")
    print(sep)

    olmo_mat_path = BASE / f"all_traits_sweep_v2_olmo3_all_traits/trait_matrix_layer{LAYER}.npy"
    llama_mat_path = BASE / f"trait_matrix_layer{LAYER}.npy"
    olmo_matrix = load_matrix(olmo_mat_path)
    llama_matrix = load_matrix(llama_mat_path)

    centroid_datasets = [
        # OLMo HarmBench training set (human jailbreak only — filter applied inside)
        ("OLMo HarmBench (all attack_types)",
         BASE / "harmbench_activations_olmo3/classified_responses.jsonl",
         BASE / "harmbench_activations_olmo3/activations.pt",
         olmo_matrix),
        # OLMo attack transfer sets
        ("OLMo PAIR",
         BASE / "pair_activations_olmo3_hb/classified_responses.jsonl",
         BASE / "pair_activations_olmo3_hb/activations.pt",
         olmo_matrix),
        ("OLMo PAP",
         BASE / "pap_activations_olmo3_hb/classified_responses.jsonl",
         BASE / "pap_activations_olmo3_hb/activations.pt",
         olmo_matrix),
        ("OLMo GPTFuzz",
         BASE / "gptfuzz_activations_olmo3_hb/classified_responses.jsonl",
         BASE / "gptfuzz_activations_olmo3_hb/activations.pt",
         olmo_matrix),
        ("OLMo PEZ",
         BASE / "pez_activations_olmo3_hb/classified_responses.jsonl",
         BASE / "pez_activations_olmo3_hb/activations.pt",
         olmo_matrix),
        # Llama HarmBench for comparison
        ("Llama HarmBench (all attack_types)",
         BASE / "harmbench_activations/classified_responses.jsonl",
         BASE / "harmbench_activations/activations.pt",
         llama_matrix),
        # Llama attack transfer sets if available
        ("Llama PAIR",
         BASE / "pair_activations/classified_responses.jsonl",
         BASE / "pair_activations/activations.pt",
         llama_matrix),
        ("Llama GPTFuzz",
         BASE / "gptfuzz_activations/classified_responses.jsonl",
         BASE / "gptfuzz_activations/activations.pt",
         llama_matrix),
    ]
    for name, jp, ap, mat in centroid_datasets:
        centroid_auc(name, jp, ap, mat)

    print(f"\n{sep}")
    print("  INTERPRETATION GUIDE")
    print(sep)
    print("  Section 1: if gpt_judged << total for attack sets → labels are default False → invalid transfer AUCs")
    print("  Section 2: if matched_at_L16 << rows → silent data loss in sweep → inflated/deflated AUCs")
    print("  Section 3: OLMo effective_rank << Llama → OLMo trait directions are more collinear (noisy)")
    print("  Section 4: if raw_centroid_AUC >> trait_centroid_AUC → jailbreak signal lives outside trait subspace")
    print(sep + "\n")


if __name__ == "__main__":
    main()
