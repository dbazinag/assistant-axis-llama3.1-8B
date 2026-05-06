#!/usr/bin/env python3
"""
run_transfer_classifier_mlp.py

Same as run_transfer_classifier.py but uses an MLP instead of logistic regression.
Architecture: input → 256 → 128 → 64 → 1 (3 hidden layers with dropout + BN)
Same strict pool split, 50 seeds, balanced test set, best-direction AUC.

Usage:
  uv run python full_trait_tools/run_transfer_classifier_mlp.py \
    --transfer1_name GCG \
    --transfer1_classified_path full_trait_output/gcg_activations/responses.jsonl \
    --transfer1_activations_path full_trait_output/gcg_activations/activations.pt \
    --transfer2_name PAIR \
    --transfer2_classified_path full_trait_output/pair_activations/responses.jsonl \
    --transfer2_activations_path full_trait_output/pair_activations/activations.pt \
    --output_dir full_trait_output/transfer_results_mlp_gcg_pair
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

# MLP hyperparameters
HIDDEN_DIMS  = [256, 128, 64]
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 200
PATIENCE     = 20       # early stopping patience
BATCH_SIZE   = 64


# ── MLP definition ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp(X_tr, y_tr, X_val, y_val, input_dim, device="cpu"):
    """Train MLP with early stopping on validation AUC."""
    model = MLP(input_dim, HIDDEN_DIMS, DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)]).to(device)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_EPOCHS)

    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32).to(device)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val_auc  = 0.0
    best_state    = None
    patience_left = PATIENCE

    for epoch in range(MAX_EPOCHS):
        model.train()
        # Mini-batch SGD
        idx = torch.randperm(len(X_tr_t))
        for start in range(0, len(idx), BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(X_tr_t[batch_idx])
            loss   = criterion(logits, y_tr_t[batch_idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation AUC
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
    """Evaluate MLP on balanced test set with best-direction AUC."""
    # Balance test set
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
    print(f"  Loading {path} ({Path(path).stat().st_size / 1e6:.1f} MB)...")
    t0 = time.time()
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  Loaded in {time.time()-t0:.1f}s")
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
    matrix = np.stack(vecs)
    print(f"  Loaded {len(names)} trait vectors → {matrix.shape}")
    return matrix, names


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


# ── Pool split ─────────────────────────────────────────────────────────────────

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
    elif mode == "top_traits":
        return X_top


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_classified_path",  default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_activations_path", default="full_trait_output/harmbench_activations/activations.pt")
    parser.add_argument("--transfer1_name",             required=True)
    parser.add_argument("--transfer1_classified_path",  required=True)
    parser.add_argument("--transfer1_activations_path", required=True)
    parser.add_argument("--transfer2_name",             default=None)
    parser.add_argument("--transfer2_classified_path",  default=None)
    parser.add_argument("--transfer2_activations_path", default=None)
    parser.add_argument("--trait_vectors_dir", default="full_trait_output/traits40_vectors/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total")
    parser.add_argument("--hyperplane_path",   default="full_trait_output/harmbench_logreg/stable_hyperplane_layer16.pt")
    parser.add_argument("--output_dir",        default="full_trait_output/transfer_results_mlp")
    parser.add_argument("--layer",    type=int, default=LAYER)
    parser.add_argument("--n_pca",    type=int, default=N_PCA)
    parser.add_argument("--top_k",    type=int, default=TOP_K)
    parser.add_argument("--n_seeds",  type=int, default=N_SEEDS)
    parser.add_argument("--device",             default="cpu",
                        help="cpu is fine — MLP trains in seconds on this data size")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n=== Loading data ===")
    human_rows_all = load_jsonl(args.human_classified_path)
    human_rows = [r for r in human_rows_all if r.get("attack_type") == "human_jailbreak"]
    human_acts = load_activations(args.human_activations_path)

    t1_rows = load_jsonl(args.transfer1_classified_path)
    t1_acts = load_activations(args.transfer1_activations_path)
    print(f"  {args.transfer1_name}: {len(t1_rows)} rows")

    t2_rows, t2_acts = None, None
    if args.transfer2_name:
        t2_rows = load_jsonl(args.transfer2_classified_path)
        t2_acts = load_activations(args.transfer2_activations_path)
        print(f"  {args.transfer2_name}: {len(t2_rows)} rows")

    print("Trait vectors...")
    trait_matrix, trait_names = load_trait_matrix(args.trait_vectors_dir, args.layer)
    w_vec = load_hyperplane(args.hyperplane_path)
    cos_w = np.abs(trait_matrix @ w_vec)
    top_k_idx = np.argsort(cos_w)[::-1][:args.top_k]

    # ── Activation matrices ───────────────────────────────────────────────────
    print("\n=== Building activation matrices ===")
    X_human_raw, y_human, human_valid = build_activation_matrix(human_rows, human_acts, args.layer)
    X_t1_raw, y_t1, _ = build_activation_matrix(t1_rows, t1_acts, args.layer)
    print(f"  Human: {len(y_human)} ({y_human.sum():.0f} jb)")
    print(f"  {args.transfer1_name}: {len(y_t1)} ({y_t1.sum():.0f} jb), chance={y_t1.mean():.3f}")

    X_t2_raw, y_t2 = None, None
    if t2_rows:
        X_t2_raw, y_t2, _ = build_activation_matrix(t2_rows, t2_acts, args.layer)
        print(f"  {args.transfer2_name}: {len(y_t2)} ({y_t2.sum():.0f} jb), chance={y_t2.mean():.3f}")

    # ── Trait projections ─────────────────────────────────────────────────────
    print("\n=== Pre-computing trait projections ===")
    X_human_traits = project_traits(X_human_raw, trait_matrix)
    X_t1_traits    = project_traits(X_t1_raw,    trait_matrix)
    X_human_top    = X_human_traits[:, top_k_idx]
    X_t1_top       = X_t1_traits[:,   top_k_idx]

    if X_t2_raw is not None:
        X_t2_traits = project_traits(X_t2_raw, trait_matrix)
        X_t2_top    = X_t2_traits[:, top_k_idx]

    modes = ["raw", "pca", "all_traits", "top_traits"]

    # ── Multi-seed MLP training ───────────────────────────────────────────────
    print(f"\n=== MLP transfer experiment ({args.n_seeds} seeds) ===")
    print(f"    Architecture: {X_human_raw.shape[1]} → {HIDDEN_DIMS} → 1")
    print(f"    Dropout={DROPOUT}, LR={LR}, WD={WEIGHT_DECAY}, EarlyStop patience={PATIENCE}")

    results = {mode: {"t1": [], "t2": [], "human_test": []} for mode in modes}

    for seed in range(args.n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_beh, train_tpl, test_beh, test_tpl = get_pool_split(
            human_valid, TRAIN_FRAC, seed)
        train_idx, test_idx = split_by_pool(
            human_valid, train_beh, train_tpl, test_beh, test_tpl)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        # Use 10% of train as validation for early stopping
        n_val = max(1, len(train_idx) // 10)
        rng = np.random.RandomState(seed)
        val_mask = rng.choice(len(train_idx), n_val, replace=False)
        val_set  = set(val_mask)
        val_idx  = [train_idx[i] for i in val_mask]
        tr_idx   = [train_idx[i] for i in range(len(train_idx)) if i not in val_set]

        for mode in modes:
            X_h  = get_features(mode, X_human_raw, X_human_traits, X_human_top)
            X_t1 = get_features(mode, X_t1_raw, X_t1_traits, X_t1_top)

            X_tr_raw = X_h[tr_idx];   y_tr = y_human[tr_idx]
            X_val_raw = X_h[val_idx]; y_val = y_human[val_idx]
            X_te_raw  = X_h[test_idx]; y_te = y_human[test_idx]

            # Scale
            scaler = StandardScaler()
            X_tr_s  = scaler.fit_transform(X_tr_raw)
            X_val_s = scaler.transform(X_val_raw)
            X_te_s  = scaler.transform(X_te_raw)
            X_t1_s  = scaler.transform(X_t1)

            # PCA reduction if needed
            if mode == "pca":
                pca = PCA(n_components=args.n_pca, random_state=RANDOM_SEED)
                X_tr_s  = pca.fit_transform(X_tr_s)
                X_val_s = pca.transform(X_val_s)
                X_te_s  = pca.transform(X_te_s)
                X_t1_s  = pca.transform(X_t1_s)

            input_dim = X_tr_s.shape[1]

            # Train MLP
            mlp = train_mlp(X_tr_s, y_tr, X_val_s, y_val, input_dim, device)

            # Evaluate
            results[mode]["t1"].append(eval_mlp(mlp, X_t1_s, y_t1, device))
            results[mode]["human_test"].append(eval_mlp(mlp, X_te_s, y_te, device))

            if X_t2_raw is not None:
                X_t2 = get_features(mode, X_t2_raw, X_t2_traits, X_t2_top)
                X_t2_s = scaler.transform(X_t2)
                if mode == "pca":
                    X_t2_s = pca.transform(X_t2_s)
                results[mode]["t2"].append(eval_mlp(mlp, X_t2_s, y_t2, device))

        if seed % 5 == 0:
            print(f"  Seed {seed} done")

    # ── Summary ───────────────────────────────────────────────────────────────
    t1n = args.transfer1_name
    t2n = args.transfer2_name or ""
    sep = "=" * 100
    print(f"\n\n{sep}")
    print(f"  MLP TRANSFER SUMMARY  |  Train: HarmBench → Test: {t1n}"
          + (f" + {t2n}" if t2n else "")
          + f"  |  Layer {args.layer}  |  {args.n_seeds} seeds")
    print(f"  Architecture: input → {HIDDEN_DIMS} → 1  |  Dropout={DROPOUT}  |  EarlyStop patience={PATIENCE}")
    print(sep)

    header = f"  {'Mode':12s}  {'→'+t1n+' mean':>14}  {'→'+t1n+' std':>12}"
    if t2n:
        header += f"  {'→'+t2n+' mean':>14}  {'→'+t2n+' std':>12}"
    header += f"  {'Human test':>12}"
    print(f"\n{header}")
    print("  " + "─" * 96)

    summary = {}
    for mode in modes:
        t1_mean = float(np.nanmean(results[mode]["t1"]))
        t1_std  = float(np.nanstd(results[mode]["t1"]))
        ht_mean = float(np.nanmean(results[mode]["human_test"]))

        row = f"  {mode:12s}  {t1_mean:>14.4f}  {t1_std:>12.4f}"
        s = {
            f"transfer_{t1n}": {"mean": t1_mean, "std": t1_std},
            "human_test": {"mean": ht_mean},
        }

        if t2n and results[mode]["t2"]:
            t2_mean = float(np.nanmean(results[mode]["t2"]))
            t2_std  = float(np.nanstd(results[mode]["t2"]))
            row += f"  {t2_mean:>14.4f}  {t2_std:>12.4f}"
            s[f"transfer_{t2n}"] = {"mean": t2_mean, "std": t2_std}

        row += f"  {ht_mean:>12.4f}"
        print(row)
        summary[mode] = s

    print(f"\n  {t1n} chance: {y_t1.mean():.4f}")
    if y_t2 is not None:
        print(f"  {t2n} chance: {y_t2.mean():.4f}")
    print(sep)

    out = {
        "architecture": HIDDEN_DIMS,
        "dropout": DROPOUT, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "n_seeds": args.n_seeds, "layer": args.layer,
        "transfer1_name": t1n, "transfer2_name": t2n,
        f"{t1n}_chance": float(y_t1.mean()),
        "modes": summary,
    }
    if y_t2 is not None:
        out[f"{t2n}_chance"] = float(y_t2.mean())

    out_path = output_dir / "transfer_results_mlp.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
