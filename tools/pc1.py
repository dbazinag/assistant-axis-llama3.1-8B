#!/usr/bin/env python3
# Fixes the torch slicing bug and also computes how many PCs are needed to reach 70% variance (per-layer).

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def torch_load_cpu(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def is_default_vector(role: str, vtype: str) -> bool:
    r = (role or "").lower()
    t = (vtype or "").lower()
    return ("default" in r) or (t == "mean") or (t == "default")


def load_role_vectors(vectors_dir: Path) -> Tuple[List[str], torch.Tensor]:
    """
    Returns:
      roles: list[str] length N
      X: Tensor shape (N, L, D) float32
    """
    roles: List[str] = []
    vecs: List[torch.Tensor] = []

    for f in sorted(vectors_dir.glob("*.pt")):
        d = torch_load_cpu(f)
        if not isinstance(d, dict) or "vector" not in d:
            continue
        role = str(d.get("role", f.stem))
        vtype = str(d.get("type", "unknown"))
        if is_default_vector(role, vtype):
            continue

        v = d["vector"]
        if not torch.is_tensor(v) or v.ndim != 2:
            continue

        roles.append(role)
        vecs.append(v.float())

    if not vecs:
        raise RuntimeError(f"No role vectors found in: {vectors_dir}")

    X = torch.stack(vecs, dim=0)  # (N, L, D)
    return roles, X


def pc_stats_per_layer(X: torch.Tensor, target: float = 0.70) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X: (N, L, D) float
    Returns:
      evr_pc1: (L,) float64  -- PC1 explained variance ratio per layer
      k_to_target: (L,) int64 -- number of PCs needed to reach 'target' cumulative variance per layer
    """
    N, L, D = X.shape
    evr_pc1 = torch.empty(L, dtype=torch.float64)
    k_to_target = torch.empty(L, dtype=torch.int64)

    for l in range(L):
        A = X[:, l, :]  # (N, D)
        A = A - A.mean(dim=0, keepdim=True)  # center across roles

        # SVD: A = U diag(S) V^T
        # Variance along PCs proportional to S^2
        _, S, _ = torch.linalg.svd(A, full_matrices=False)
        S2 = (S.double() ** 2)
        total = S2.sum() + 1e-12

        evr = S2 / total
        evr_pc1[l] = evr[0].item()

        cum = torch.cumsum(evr, dim=0)
        # first index where cum >= target
        idx = int(torch.searchsorted(cum, torch.tensor(target, dtype=cum.dtype)).item())
        k_to_target[l] = idx + 1  # convert 0-based idx to count

    return evr_pc1, k_to_target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors_dir", type=str, required=True, help="Path to vectors_q50 (pt files).")
    ap.add_argument("--out_path", type=str, default="", help="Optional JSON output path.")
    ap.add_argument("--print_topk", type=int, default=5, help="How many best/worst layers to print.")
    ap.add_argument("--target", type=float, default=0.70, help="Target cumulative variance for k PCs.")
    args = ap.parse_args()

    vectors_dir = Path(args.vectors_dir)
    roles, X = load_role_vectors(vectors_dir)

    evr, k70 = pc_stats_per_layer(X, target=args.target)

    L = int(evr.numel())
    evr_mean = float(evr.mean().item())
    evr_min = float(evr.min().item())
    evr_max = float(evr.max().item())
    argmin = int(torch.argmin(evr).item())
    argmax = int(torch.argmax(evr).item())

    k_mean = float(k70.double().mean().item())
    k_min = int(k70.min().item())
    k_max = int(k70.max().item())
    k_argmin = int(torch.argmin(k70).item())
    k_argmax = int(torch.argmax(k70).item())

    print("=" * 60)
    print("PC explained-variance stats (per-layer)")
    print("=" * 60)
    print(f"vectors_dir: {vectors_dir}")
    print(f"n_roles: {len(roles)}")
    print(f"n_layers: {L}")
    print(f"PC1 EVR mean: {evr_mean:.6f}")
    print(f"PC1 EVR min : {evr_min:.6f} (layer {argmin})")
    print(f"PC1 EVR max : {evr_max:.6f} (layer {argmax})")
    print(f"\nPCs to reach {args.target*100:.0f}% variance:")
    print(f"k mean: {k_mean:.3f}")
    print(f"k min : {k_min} (layer {k_argmin})")
    print(f"k max : {k_max} (layer {k_argmax})")

    k = max(1, min(args.print_topk, L))
    order_evr = torch.argsort(evr)  # ascending
    worst = [(int(i), float(evr[i].item())) for i in order_evr[:k]]
    best = [(int(i), float(evr[i].item())) for i in torch.flip(order_evr[-k:], dims=[0])]

    print(f"\nWorst {k} layers by PC1 EVR (layer, evr):")
    for l, v in worst:
        print(f"  {l:>2d}: {v:.6f}")

    print(f"\nBest {k} layers by PC1 EVR (layer, evr):")
    for l, v in best:
        print(f"  {l:>2d}: {v:.6f}")

    # Also show k-to-target best/worst
    order_k = torch.argsort(k70)  # ascending
    best_k = [(int(i), int(k70[i].item())) for i in order_k[:k]]
    worst_k = [(int(i), int(k70[i].item())) for i in torch.flip(order_k[-k:], dims=[0])]

    print(f"\nBest {k} layers by fewest PCs for {args.target*100:.0f}% (layer, k):")
    for l, kv in best_k:
        print(f"  {l:>2d}: {kv}")

    print(f"\nWorst {k} layers by most PCs for {args.target*100:.0f}% (layer, k):")
    for l, kv in worst_k:
        print(f"  {l:>2d}: {kv}")

    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {
            "vectors_dir": str(vectors_dir),
            "n_roles": len(roles),
            "n_layers": L,
            "target_cumulative_variance": float(args.target),
            "pc1_explained_variance_ratio_by_layer": [float(x) for x in evr.tolist()],
            "pcs_needed_to_reach_target_by_layer": [int(x) for x in k70.tolist()],
            "summary": {
                "pc1_evr": {
                    "mean": evr_mean,
                    "min": {"value": evr_min, "layer": argmin},
                    "max": {"value": evr_max, "layer": argmax},
                },
                "k_to_target": {
                    "mean": k_mean,
                    "min": {"value": k_min, "layer": k_argmin},
                    "max": {"value": k_max, "layer": k_argmax},
                },
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {out_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()