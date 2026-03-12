#!/usr/bin/env python3
"""
Compute assistant axis via PCA over all trait vectors.

Method (same idea as assistant-axis paper):
1. Load all trait vectors
2. Stack them
3. Run PCA per layer
4. Take PC1 as assistant axis

Also prints:
- variance explained
- PCs needed for 70% / 90% variance
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm


def load_vector(path: Path):
    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        return data["vector"].float()

    return data.float()


def pca_stats(X):
    """
    X shape: (n_vectors, hidden_dim)
    """

    X = X - X.mean(dim=0, keepdim=True)

    U, S, V = torch.pca_lowrank(X, q=min(X.shape) - 1)

    var = S**2
    var_ratio = var / var.sum()

    cumulative = torch.cumsum(var_ratio, dim=0)

    k70 = (cumulative >= 0.70).nonzero()[0].item() + 1
    k90 = (cumulative >= 0.90).nonzero()[0].item() + 1

    return V[:, 0], var_ratio, k70, k90


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors_dir", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    vectors_dir = Path(args.vectors_dir)
    output_path = Path(args.output)

    files = sorted(vectors_dir.glob("*.pt"))

    print(f"\nFound {len(files)} trait vectors")

    vectors = []

    for f in tqdm(files):

        vec = load_vector(f)

        vectors.append(vec)

    stacked = torch.stack(vectors)

    n_traits, n_layers, hidden = stacked.shape

    print("\nStacked tensor shape:", stacked.shape)

    axis_layers = []
    pc1_variance = []

    k70_list = []
    k90_list = []

    print("\nRunning PCA per layer...")

    for layer in range(n_layers):

        X = stacked[:, layer, :]

        pc1, var_ratio, k70, k90 = pca_stats(X)

        axis_layers.append(pc1)

        pc1_variance.append(var_ratio[0].item())
        k70_list.append(k70)
        k90_list.append(k90)

    axis = torch.stack(axis_layers)

    print("\nAxis shape:", axis.shape)

    print("\n=== PCA STATS ===")

    print("\nPC1 variance explained (first 10 layers):")
    for i in range(min(10, n_layers)):
        print(f"Layer {i:2d}: {pc1_variance[i]*100:.2f}%")

    print("\nAverage PC1 variance explained:")
    print(f"{sum(pc1_variance)/len(pc1_variance)*100:.2f}%")

    print("\nComponents needed for 70% variance:")
    print(f"mean = {sum(k70_list)/len(k70_list):.2f}")

    print("\nComponents needed for 90% variance:")
    print(f"mean = {sum(k90_list)/len(k90_list):.2f}")

    norms = axis.norm(dim=1)

    print("\nAxis norms per layer (top 10):")

    top = torch.topk(norms, 10)

    for idx in top.indices:
        print(f"L{idx.item():2d} = {norms[idx].item():.4f}")

    torch.save(axis, output_path)

    print(f"\nSaved assistant axis → {output_path}")


if __name__ == "__main__":
    main()