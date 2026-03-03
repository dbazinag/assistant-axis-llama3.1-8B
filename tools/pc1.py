#!/usr/bin/env python3
# Computes PC1 explained variance ratio (per-layer) from saved role vectors (PCA via SVD).

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

    # (N, L, D)
    X = torch.stack(vecs, dim=0)
    return roles, X


def explained_variance_ratio_pc1_per_layer(X: torch.Tensor) -> torch.Tensor:
    """
    X: (N, L, D) float
    Returns: evr: (L,) float64, PC1 explained variance ratio per layer.
    """
    N, L, D = X.shape
    evr = torch.empty(L, dtype=torch.float64)

    for l in range(L):
        A = X[:, l, :]  # (N, D)
        A = A - A.mean(dim=0, keepdim=True)  # center across roles
        # singular values S: (min(N,D),)
        # explained variance proportional to S^2
        _, S, _ = torch.linalg.svd(A, full_matrices=False)
        S2 = (S.double() ** 2)
        evr[l] = (S2[0] / (S2.sum() + 1e-12)).item()

    return evr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors_dir", type=str, required=True, help="Path to vectors_q50 (pt files).")
    ap.add_argument("--out_path", type=str, required=False, default="", help="Optional JSON output path.")
    ap.add_argument("--print_topk", type=int, default=5, help="How many best/worst layers to print.")
    args = ap.parse_args()

    vectors_dir = Path(args.vectors_dir)
    roles, X = load_role_vectors(vectors_dir)

    evr = explained_variance_ratio_pc1_per_layer(X)  # (L,)

    L = evr.numel()
    evr_mean = float(evr.mean().item())
    evr_min = float(evr.min().item())
    evr_max = float(evr.max().item())
    argmin = int(torch.argmin(evr).item())
    argmax = int(torch.argmax(evr).item())

    print("=" * 60)
    print("PC1 Explained Variance Ratio (per-layer)")
    print("=" * 60)
    print(f"vectors_dir: {vectors_dir}")
    print(f"n_roles: {len(roles)}")
    print(f"n_layers: {L}")
    print(f"EVR mean: {evr_mean:.6f}")
    print(f"EVR min : {evr_min:.6f} (layer {argmin})")
    print(f"EVR max : {evr_max:.6f} (layer {argmax})")

    k = max(1, min(args.print_topk, L))
    order = torch.argsort(evr)  # ascending
    worst = [(int(i), float(evr[i].item())) for i in order[:k]]
    best = [(int(i), float(evr[i].item())) for i in order[-k:][::-1]]

    print(f"\nWorst {k} layers (layer, evr):")
    for l, v in worst:
        print(f"  {l:>2d}: {v:.6f}")

    print(f"\nBest {k} layers (layer, evr):")
    for l, v in best:
        print(f"  {l:>2d}: {v:.6f}")

    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {
            "vectors_dir": str(vectors_dir),
            "n_roles": len(roles),
            "n_layers": L,
            "pc1_explained_variance_ratio_by_layer": [float(x) for x in evr.tolist()],
            "summary": {
                "mean": evr_mean,
                "min": {"value": evr_min, "layer": argmin},
                "max": {"value": evr_max, "layer": argmax},
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {out_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()