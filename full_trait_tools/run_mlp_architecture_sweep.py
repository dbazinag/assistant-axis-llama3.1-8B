#!/usr/bin/env python3
"""
run_mlp_architecture_sweep.py

Sweeps MLP architectures from very simple (high bias) to very complex
(high variance) to find the bias-variance sweet spot for jailbreak transfer.

Tests 20 architectures × 10 seeds × 4 modes × all attack families.

Architectures ordered simple → complex:
  Very simple:  [8], [16], [32], [64], [128]
  Shallow 2L:   [32,16], [64,32], [128,64], [256,128]
  Medium 3L:    [64,32,16], [128,64,32], [256,128,64]*, [512,256,128]
  Deep 4L:      [128,64,32,16], [256,128,64,32], [512,256,128,64], [1024,512,256,128]
  Very deep 5L: [256,128,64,32,16], [512,256,128,64,32], [1024,512,256,128,64]
  * current baseline

Usage:
  uv run python full_trait_tools/run_mlp_architecture_sweep.py
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

LAYER        = 16
N_PCA        = 4
TOP_K        = 20
TRAIN_FRAC   = 0.7
N_SEEDS      = 10
RANDOM_SEED  = 42
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 200
PATIENCE     = 20
BATCH_SIZE   = 64

ARCHITECTURES = [
    # Very simple — high bias
    [8],
    [16],
    [32],
    [64],
    [128],
    # Shallow 2-layer
    [32, 16],
    [64, 32],
    [128, 64],
    [256, 128],
    # Medium 3-layer
    [64, 32, 16],
    [128, 64, 32],
    [256, 128, 64],        # current baseline
    [512, 256, 128],
    # Deep 4-layer
    [128, 64, 32, 16],
    [256, 128, 64, 32],
    [512, 256, 128, 64],
    [1024, 512, 256, 128],
    # Very deep 5-layer — high variance
    [256, 128, 64, 32, 16],
    [512, 256, 128, 64, 32],
    [1024, 512, 256, 128, 64],
]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp(X_tr, y_tr, X_val, y_val, hidden_dims, seed, device="cpu"):
    torch.manual_seed(seed)
    model = MLP(X_tr.shape[1], hidden_dims, DROPOUT).to(device)
    pos_weight = torch.tensor(
        [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
        dtype=torch.float32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_EPOCHS)

    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32).to(device)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_val_auc, best_state, patience_left = 0.0, None, PATIENCE

    for epoch in range(MAX_EPOCHS):
        model.train()
        idx = torch.randperm(len(X_tr_t),
                             generator=torch.Generator().manual_seed(seed + epoch))
        for start in range(0, len(idx), BATCH_SIZE):
            batch = idx[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            criterion(model(X_tr_t[batch]), y_tr_t[batch]).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).cpu().numpy()
        if len(set(y_val)) >= 2:
            val_auc = max(roc_auc_score(y_val, val_logits),
                          1 - roc_auc_score(y_val, val_logits))
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
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_te[idx], dtype=torch.float32).to(device)).cpu().numpy()
    auc = roc_auc_score(y_te[idx], logits)
    return max(auc, 1 - auc)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_activations(path):
    print(f"  Loading {Path(path).name}...", flush=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_trait_matrix(layer):
    cache_mat   = Path(f"full_trait_output/trait_matrix_layer{layer}.npy")
    cache_names = Path(f"full_trait_output/trait_names_layer{layer}.json")
    if not cache_mat.exists():
        raise FileNotFoundError("Run cache script first to generate trait_matrix_layer16.npy")
    print("  Loading cached trait matrix...", flush=True)
    return np.load(str(cache_mat)), json.load(open(cache_names))


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
    all_beh = sorted({r["behavior_id"]  for r in rows})
    all_tpl = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(all_beh); rng.shuffle(all_tpl)
    n_beh = max(1, int(len(all_beh) * train_frac))
    n_tpl = max(1, int(len(all_tpl) * train_frac))
    return (set(all_beh[:n_beh]), set(all_tpl[:n_tpl]),
            set(all_beh[n_beh:]),  set(all_tpl[n_tpl:]))


def split_by_pool(rows, train_beh, train_tpl, test_beh, test_tpl):
    tr  = [i for i, r in enumerate(rows)
           if r["behavior_id"] in train_beh and r["jailbreak_idx"] in train_tpl]
    te  = [i for i, r in enumerate(rows)
           if r["behavior_id"] in test_beh  and r["jailbreak_idx"] in test_tpl]
    return tr, te


def get_features(mode, X_raw, X_traits, X_top):
    if mode in ("raw", "pca"):    return X_raw
    if mode == "all_traits":      return X_traits
    return X_top


def n_params(input_dim, hidden_dims):
    dims = [input_dim] + hidden_dims + [1]
    return sum(dims[i] * dims[i+1] for i in range(len(dims)-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_classified_path",   default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_activations_path",  default="full_trait_output/harmbench_activations/activations.pt")
    parser.add_argument("--gcg_classified_path",     default="full_trait_output/gcg_activations/responses.jsonl")
    parser.add_argument("--gcg_activations_path",    default="full_trait_output/gcg_activations/activations.pt")
    parser.add_argument("--pair_classified_path",    default="full_trait_output/pair_activations/responses.jsonl")
    parser.add_argument("--pair_activations_path",   default="full_trait_output/pair_activations/activations.pt")
    parser.add_argument("--pap_classified_path",     default="full_trait_output/pap_activations/responses.jsonl")
    parser.add_argument("--pap_activations_path",    default="full_trait_output/pap_activations/activations.pt")
    parser.add_argument("--gptfuzz_classified_path", default="full_trait_output/gptfuzz_activations/responses.jsonl")
    parser.add_argument("--gptfuzz_activations_path",default="full_trait_output/gptfuzz_activations/activations.pt")
    parser.add_argument("--pez_classified_path",     default="full_trait_output/pez_activations/responses.jsonl")
    parser.add_argument("--pez_activations_path",    default="full_trait_output/pez_activations/activations.pt")
    parser.add_argument("--hyperplane_path", default="full_trait_output/harmbench_logreg/stable_hyperplane_layer16.pt")
    parser.add_argument("--output_dir",      default="full_trait_output/mlp_sweep")
    parser.add_argument("--layer",   type=int, default=LAYER)
    parser.add_argument("--n_pca",   type=int, default=N_PCA)
    parser.add_argument("--top_k",   type=int, default=TOP_K)
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    parser.add_argument("--device",           default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n=== Loading ===", flush=True)
    human_rows = [r for r in load_jsonl(args.human_classified_path)
                  if r.get("attack_type") == "human_jailbreak"]
    human_acts = load_activations(args.human_activations_path)

    transfer_datasets = {}
    for name, cp, ap in [
        ("GCG",     args.gcg_classified_path,     args.gcg_activations_path),
        ("PAIR",    args.pair_classified_path,    args.pair_activations_path),
        ("PAP",     args.pap_classified_path,     args.pap_activations_path),
        ("GPTFuzz", args.gptfuzz_classified_path, args.gptfuzz_activations_path),
        ("PEZ",     args.pez_classified_path,     args.pez_activations_path),
    ]:
        if Path(cp).exists() and Path(ap).exists():
            transfer_datasets[name] = (load_jsonl(cp), load_activations(ap))
            print(f"  {name} loaded", flush=True)

    trait_matrix, trait_names = load_trait_matrix(args.layer)
    w_vec     = load_hyperplane(args.hyperplane_path)
    top_k_idx = np.argsort(np.abs(trait_matrix @ w_vec))[::-1][:args.top_k]

    # ── Feature matrices ──────────────────────────────────────────────────────
    print("\n=== Building feature matrices ===", flush=True)
    X_human_raw, y_human, human_valid = build_activation_matrix(
        human_rows, human_acts, args.layer)
    X_human_traits = project_traits(X_human_raw, trait_matrix)
    X_human_top    = X_human_traits[:, top_k_idx]

    transfer_feats = {}
    for name, (rows, acts) in transfer_datasets.items():
        X, y, _ = build_activation_matrix(rows, acts, args.layer)
        X_tr = project_traits(X, trait_matrix)
        transfer_feats[name] = (X, X_tr, X_tr[:, top_k_idx], y)
        print(f"  {name}: {len(y)} pairs, chance={y.mean():.3f}", flush=True)

    modes    = ["raw", "pca", "all_traits", "top_traits"]
    families = list(transfer_feats.keys())

    # ── Sweep ─────────────────────────────────────────────────────────────────
    print(f"\n=== Sweeping {len(ARCHITECTURES)} architectures × {args.n_seeds} seeds ===",
          flush=True)

    all_results = {}

    for arch_idx, hidden_dims in enumerate(ARCHITECTURES):
        arch_str = str(hidden_dims)
        np_raw    = n_params(X_human_raw.shape[1],    hidden_dims)
        np_traits = n_params(X_human_traits.shape[1], hidden_dims)
        print(f"\n[{arch_idx+1}/{len(ARCHITECTURES)}] {hidden_dims} "
              f"| raw_params={np_raw:,} traits_params={np_traits:,}", flush=True)

        arch_res = {mode: {f: [] for f in families + ["human_test"]}
                    for mode in modes}

        for seed in range(args.n_seeds):
            tr_beh, tr_tpl, te_beh, te_tpl = get_pool_split(human_valid, TRAIN_FRAC, seed)
            tr_idx, te_idx = split_by_pool(human_valid, tr_beh, tr_tpl, te_beh, te_tpl)
            if not tr_idx or not te_idx:
                continue

            rng     = np.random.RandomState(seed)
            n_val   = max(1, len(tr_idx) // 10)
            val_set = set(rng.choice(len(tr_idx), n_val, replace=False))
            val_idx = [tr_idx[i] for i in val_set]
            tr_idx2 = [tr_idx[i] for i in range(len(tr_idx)) if i not in val_set]

            for mode in modes:
                X_h   = get_features(mode, X_human_raw, X_human_traits, X_human_top)
                X_tr  = X_h[tr_idx2]; y_tr  = y_human[tr_idx2]
                X_val = X_h[val_idx]; y_val = y_human[val_idx]
                X_te  = X_h[te_idx];  y_te  = y_human[te_idx]

                sc      = StandardScaler()
                X_tr_s  = sc.fit_transform(X_tr)
                X_val_s = sc.transform(X_val)
                X_te_s  = sc.transform(X_te)

                pca_obj = None
                if mode == "pca":
                    pca_obj = PCA(n_components=args.n_pca, random_state=RANDOM_SEED)
                    X_tr_s  = pca_obj.fit_transform(X_tr_s)
                    X_val_s = pca_obj.transform(X_val_s)
                    X_te_s  = pca_obj.transform(X_te_s)

                mlp = train_mlp(X_tr_s, y_tr, X_val_s, y_val, hidden_dims, seed, args.device)
                arch_res[mode]["human_test"].append(eval_mlp(mlp, X_te_s, y_te, args.device))

                for fname, (Xr, Xtr, Xto, yt) in transfer_feats.items():
                    Xf = get_features(mode, Xr, Xtr, Xto)
                    Xs = sc.transform(Xf)
                    if pca_obj is not None:
                        Xs = pca_obj.transform(Xs)
                    arch_res[mode][fname].append(eval_mlp(mlp, Xs, yt, args.device))

        # Print per-arch summary
        for mode in modes:
            ht  = np.nanmean(arch_res[mode]["human_test"])
            avt = np.nanmean([np.nanmean(arch_res[mode][f]) for f in families])
            fam = "  ".join(f"{f}={np.nanmean(arch_res[mode][f]):.3f}" for f in families)
            print(f"  {mode:12s}: human={ht:.3f} avg_transfer={avt:.3f}  {fam}", flush=True)

        all_results[arch_str] = arch_res

    # ── Summary tables ─────────────────────────────────────────────────────────
    sep = "=" * 130
    print(f"\n\n{sep}")
    print(f"  ARCHITECTURE SWEEP  |  {args.n_seeds} seeds  |  Layer {args.layer}")
    print(sep)

    for mode in modes:
        print(f"\n  [{mode}]")
        header = f"  {'Architecture':28s}  {'Params':>10}  {'HumanTest':>10}  {'AvgXfer':>10}" + \
                 "".join(f"  {f:>9}" for f in families)
        print(header)
        print("  " + "─" * (28 + 10 + 10 + 10 + 11 * len(families)))
        for hidden_dims in ARCHITECTURES:
            arch_str = str(hidden_dims)
            np_ = n_params(
                X_human_raw.shape[1] if mode in ("raw","pca") else X_human_traits.shape[1],
                hidden_dims)
            r   = all_results[arch_str][mode]
            ht  = np.nanmean(r["human_test"])
            fav = [np.nanmean(r[f]) for f in families]
            avg = np.nanmean(fav)
            marker = " ←baseline" if hidden_dims == [256, 128, 64] else ""
            row = f"  {arch_str:28s}  {np_:>10,}  {ht:>10.4f}  {avg:>10.4f}"
            row += "".join(f"  {v:>9.4f}" for v in fav)
            print(row + marker)

    print(f"\n{sep}")

    # Save
    out = {
        "architectures": [str(a) for a in ARCHITECTURES],
        "n_seeds": args.n_seeds, "layer": args.layer, "families": families,
        "results": {
            arch_str: {
                mode: {f: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)), "all": v}
                       for f, v in mode_res.items()}
                for mode, mode_res in arch_res.items()
            }
            for arch_str, arch_res in all_results.items()
        }
    }
    out_path = output_dir / "mlp_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
