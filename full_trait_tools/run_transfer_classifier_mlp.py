#!/usr/bin/env python3
"""
run_transfer_classifier_mlp.py

MLP-based transfer classifier. Same as fast_transfer_classifier.py but replaces
logistic regression with a 3-layer MLP (input → 256 → 128 → 64 → 1).

Trains on HarmBench strict pool split, tests on all attack families in one run.
Same 50 seeds, balanced 50/50 test set, best-direction AUC throughout.

Architecture details:
  - 3 hidden layers: 256 → 128 → 64
  - BatchNorm + ReLU + Dropout(0.3) per layer
  - BCEWithLogitsLoss with class reweighting
  - Adam optimizer, cosine LR schedule
  - Early stopping on validation AUC (patience=20, 10% of train as val)

Usage:
  uv run python full_trait_tools/run_transfer_classifier_mlp.py
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

LAYER       = 16
N_PCA       = 4
TOP_K       = 20
TRAIN_FRAC  = 0.7
N_SEEDS     = 50
RANDOM_SEED = 42

HIDDEN_DIMS  = [256, 128, 64]
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 200
PATIENCE     = 20
BATCH_SIZE   = 64


# ── MLP ────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp(X_tr, y_tr, X_val, y_val, input_dim, seed, device="cpu"):
    torch.manual_seed(seed)
    model = MLP(input_dim, HIDDEN_DIMS, DROPOUT).to(device)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                               dtype=torch.float32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_EPOCHS)

    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32).to(device)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_val_auc  = 0.0
    best_state    = None
    patience_left = PATIENCE

    for epoch in range(MAX_EPOCHS):
        model.train()
        idx = torch.randperm(len(X_tr_t), generator=torch.Generator().manual_seed(seed + epoch))
        for start in range(0, len(idx), BATCH_SIZE):
            batch = idx[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t[batch]), y_tr_t[batch])
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).cpu().numpy()
        if len(set(y_val)) >= 2:
            val_auc = roc_auc_score(y_val, val_logits)
            val_auc = max(val_auc, 1 - val_auc)
            if val_auc > best_val_auc:
                best_val_auc  = val_auc
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                patience_left = PATIENCE
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_mlp(model, X_te, y_te, device="cpu"):
    idx_pos = np.where(y_te == 1)[0]
    idx_neg = np.where(y_te == 0)[0]
    n = min(len(idx_pos), len(idx_neg))
    if n == 0 or len(set(y_te)) < 2:
        return float("nan")
    rng = np.random.RandomState(RANDOM_SEED)
    idx = np.concatenate([rng.choice(idx_pos, n, replace=False),
                          rng.choice(idx_neg, n, replace=False)])
    X_b, y_b = X_te[idx], y_te[idx]
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_b, dtype=torch.float32).to(device)).cpu().numpy()
    auc = roc_auc_score(y_b, logits)
    return max(auc, 1 - auc)


# ── I/O ────────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_activations(path):
    print(f"  Loading {path}...", flush=True)
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  Done loading {Path(path).name}", flush=True)
    return data


def load_trait_matrix(vectors_dir, layer):
    vecs, names = [], []
    for pt_file in sorted(Path(vectors_dir).glob("*.pt")):
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            v = data["vector"][layer].float().numpy()
            norm = np.linalg.norm(v)
            if norm > 1e-8:
                vecs.append(v / norm)
                names.append(pt_file.stem)
        except Exception:
            pass
    return np.stack(vecs), names


def load_hyperplane(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    v = data["vector"].float().numpy()
    return v / (np.linalg.norm(v) + 1e-12)


def build_activation_matrix(rows, activations, layer):
    layer_key = str(layer)
    X_list, y_list, valid_rows = [], [], []
    for row in rows:
        pid = row["pair_id"]
        jb  = row.get("jailbroken")
        if jb is None:
            continue
        if pid not in activations or layer_key not in activations[pid]:
            continue
        X_list.append(activations[pid][layer_key].float().numpy())
        y_list.append(1 if jb else 0)
        valid_rows.append(row)
    if not X_list:
        return np.array([]), np.array([]), []
    return np.stack(X_list), np.array(y_list), valid_rows


def project_traits(X_raw, trait_matrix):
    return X_raw @ trait_matrix.T


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


def get_features(mode, X_raw, X_traits, X_top):
    if mode in ("raw", "pca"):
        return X_raw
    elif mode == "all_traits":
        return X_traits
    return X_top


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_classified_path",  default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_activations_path", default="full_trait_output/harmbench_activations/activations.pt")
    parser.add_argument("--gcg_classified_path",    default="full_trait_output/gcg_activations/responses.jsonl")
    parser.add_argument("--gcg_activations_path",   default="full_trait_output/gcg_activations/activations.pt")
    parser.add_argument("--pair_classified_path",   default="full_trait_output/pair_activations/responses.jsonl")
    parser.add_argument("--pair_activations_path",  default="full_trait_output/pair_activations/activations.pt")
    parser.add_argument("--pap_classified_path",    default="full_trait_output/pap_activations/responses.jsonl")
    parser.add_argument("--pap_activations_path",   default="full_trait_output/pap_activations/activations.pt")
    parser.add_argument("--gptfuzz_classified_path",  default="full_trait_output/gptfuzz_activations/responses.jsonl")
    parser.add_argument("--gptfuzz_activations_path", default="full_trait_output/gptfuzz_activations/activations.pt")
    parser.add_argument("--pez_classified_path",    default="full_trait_output/pez_activations/responses.jsonl")
    parser.add_argument("--pez_activations_path",   default="full_trait_output/pez_activations/activations.pt")
    parser.add_argument("--trait_vectors_dir", default="full_trait_output/traits40_vectors/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total")
    parser.add_argument("--hyperplane_path",   default="full_trait_output/harmbench_logreg/stable_hyperplane_layer16.pt")
    parser.add_argument("--output_dir",        default="full_trait_output/transfer_results_mlp")
    parser.add_argument("--layer",    type=int, default=LAYER)
    parser.add_argument("--n_pca",    type=int, default=N_PCA)
    parser.add_argument("--top_k",    type=int, default=TOP_K)
    parser.add_argument("--n_seeds",  type=int, default=N_SEEDS)
    parser.add_argument("--device",             default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # ── Load all datasets ──────────────────────────────────────────────────────
    print("\n=== Loading data ===", flush=True)
    print("Loading human responses...", flush=True)
    human_rows_all = load_jsonl(args.human_classified_path)
    human_rows = [r for r in human_rows_all if r.get("attack_type") == "human_jailbreak"]
    print(f"  Human responses: {len(human_rows)} rows", flush=True)

    print("Loading human activations...", flush=True)
    human_acts = load_activations(args.human_activations_path)
    print(f"  Human activations loaded: {len(human_acts)} pairs", flush=True)

    transfer_datasets = {}
    for name, cp, ap in [
        ("GCG",      args.gcg_classified_path,     args.gcg_activations_path),
        ("PAIR",     args.pair_classified_path,    args.pair_activations_path),
        ("PAP",      args.pap_classified_path,     args.pap_activations_path),
        ("GPTFuzz",  args.gptfuzz_classified_path, args.gptfuzz_activations_path),
        ("PEZ",      args.pez_classified_path,     args.pez_activations_path),
    ]:
        if Path(cp).exists() and Path(ap).exists():
            print(f"Loading {name} responses...", flush=True)
            rows = load_jsonl(cp)
            print(f"Loading {name} activations...", flush=True)
            acts = load_activations(ap)
            transfer_datasets[name] = (rows, acts)
            print(f"  {name}: {len(rows)} rows loaded", flush=True)
        else:
            print(f"  Skipping {name} — files not found", flush=True)

    print("Loading trait vectors...", flush=True)
    trait_matrix, trait_names = load_trait_matrix(args.trait_vectors_dir, args.layer)
    w_vec = load_hyperplane(args.hyperplane_path)
    cos_w = np.abs(trait_matrix @ w_vec)
    top_k_idx = np.argsort(cos_w)[::-1][:args.top_k]
    print(f"  {len(trait_names)} traits, top-{args.top_k}: {[trait_names[i] for i in top_k_idx[:3]]}...")

    # ── Activation matrices ───────────────────────────────────────────────────
    print("\n=== Building activation matrices ===")
    X_human_raw, y_human, human_valid = build_activation_matrix(
        human_rows, human_acts, args.layer)
    print(f"  Human: {len(y_human)} ({y_human.sum():.0f} jb)")

    transfer_matrices = {}
    for name, (rows, acts) in transfer_datasets.items():
        X, y, _ = build_activation_matrix(rows, acts, args.layer)
        transfer_matrices[name] = (X, y)
        print(f"  {name}: {len(y)} ({y.sum():.0f} jb), chance={y.mean():.3f}")

    # ── Trait projections ─────────────────────────────────────────────────────
    print("\n=== Pre-computing trait projections ===")
    X_human_traits = project_traits(X_human_raw, trait_matrix)
    X_human_top    = X_human_traits[:, top_k_idx]

    transfer_traits = {}
    for name, (X, y) in transfer_matrices.items():
        X_traits = project_traits(X, trait_matrix)
        X_top    = X_traits[:, top_k_idx]
        transfer_traits[name] = (X, X_traits, X_top, y)

    modes = ["raw", "pca", "all_traits", "top_traits"]

    # ── Multi-seed MLP ────────────────────────────────────────────────────────
    print(f"\n=== MLP transfer ({args.n_seeds} seeds) ===")
    print(f"  Architecture: input → {HIDDEN_DIMS} → 1 | Dropout={DROPOUT} | patience={PATIENCE}")

    results = {mode: {name: [] for name in list(transfer_matrices.keys()) + ["human_test"]}
               for mode in modes}

    for seed in range(args.n_seeds):
        train_beh, train_tpl, test_beh, test_tpl = get_pool_split(
            human_valid, TRAIN_FRAC, seed)
        train_idx, test_idx = split_by_pool(
            human_valid, train_beh, train_tpl, test_beh, test_tpl)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        # 10% of train as validation for early stopping
        rng = np.random.RandomState(seed)
        n_val = max(1, len(train_idx) // 10)
        val_mask = set(rng.choice(len(train_idx), n_val, replace=False))
        val_idx  = [train_idx[i] for i in val_mask]
        tr_idx   = [train_idx[i] for i in range(len(train_idx)) if i not in val_mask]

        for mode in modes:
            X_h = get_features(mode, X_human_raw, X_human_traits, X_human_top)

            X_tr  = X_h[tr_idx];   y_tr  = y_human[tr_idx]
            X_val = X_h[val_idx];  y_val = y_human[val_idx]
            X_te  = X_h[test_idx]; y_te  = y_human[test_idx]

            scaler = StandardScaler()
            X_tr_s  = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            X_te_s  = scaler.transform(X_te)

            pca_obj = None
            if mode == "pca":
                pca_obj = PCA(n_components=args.n_pca, random_state=RANDOM_SEED)
                X_tr_s  = pca_obj.fit_transform(X_tr_s)
                X_val_s = pca_obj.transform(X_val_s)
                X_te_s  = pca_obj.transform(X_te_s)

            mlp = train_mlp(X_tr_s, y_tr, X_val_s, y_val, X_tr_s.shape[1], seed, device)
            results[mode]["human_test"].append(eval_mlp(mlp, X_te_s, y_te, device))

            for name, (X_raw, X_traits, X_top, y_t) in transfer_traits.items():
                X_transfer = get_features(mode, X_raw, X_traits, X_top)
                X_t_s = scaler.transform(X_transfer)
                if pca_obj is not None:
                    X_t_s = pca_obj.transform(X_t_s)
                results[mode][name].append(eval_mlp(mlp, X_t_s, y_t, device))

        if seed % 5 == 0:
            print(f"  Seed {seed} done")

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "=" * 110
    print(f"\n\n{sep}")
    print(f"  MLP TRANSFER SUMMARY  |  Layer {args.layer}  |  {args.n_seeds} seeds  |  {HIDDEN_DIMS} hidden dims")
    print(sep)

    families = list(transfer_matrices.keys())
    header = f"  {'Mode':12s}  {'HumanTest':>10}" + "".join(f"  {'→'+n:>12}" for n in families)
    print(f"\n{header}")
    print("  " + "─" * (12 + 12 + 14 * len(families)))

    summary = {}
    for mode in modes:
        ht = float(np.nanmean(results[mode]["human_test"]))
        row = f"  {mode:12s}  {ht:>10.4f}"
        summary[mode] = {"human_test": {"mean": ht}}
        for name in families:
            m = float(np.nanmean(results[mode][name]))
            s = float(np.nanstd(results[mode][name]))
            row += f"  {m:>12.4f}"
            summary[mode][f"transfer_{name}"] = {"mean": m, "std": s}
        print(row)

    print(f"\n  Chances: " + ", ".join(
        f"{n}={transfer_matrices[n][1].mean():.3f}" for n in families))
    print(sep)

    with open(output_dir / "transfer_results_mlp.json", "w") as f:
        json.dump({"architecture": HIDDEN_DIMS, "dropout": DROPOUT,
                   "n_seeds": args.n_seeds, "layer": args.layer,
                   "modes": summary}, f, indent=2)
    print(f"\nSaved to {output_dir}/transfer_results_mlp.json")


if __name__ == "__main__":
    main()
