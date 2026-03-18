#!/usr/bin/env python3
# Interview note: computes a PCA-based assistant-like axis (PC1 per layer) from a filtered trait-vector set and saves rich metadata + README.

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_vector_file(path: Path) -> Tuple[torch.Tensor, str]:
    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        if "vector" not in data:
            raise ValueError(f"{path} is missing 'vector'")
        trait_name = data.get("trait", path.stem)
        return data["vector"].float(), trait_name

    return data.float(), path.stem


def pca_stats(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    # X shape: (n_vectors, hidden_dim)
    X = X - X.mean(dim=0, keepdim=True)

    q = min(X.shape[0], X.shape[1]) - 1
    if q < 1:
        raise ValueError("Need at least 2 vectors to run PCA")

    _, S, V = torch.pca_lowrank(X, q=q)

    var = S**2
    var_ratio = var / var.sum()
    cumulative = torch.cumsum(var_ratio, dim=0)

    k70 = int((cumulative >= 0.70).nonzero()[0].item() + 1)
    k90 = int((cumulative >= 0.90).nonzero()[0].item() + 1)

    pc1 = V[:, 0]
    return pc1, var_ratio, k70, k90


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def build_readme_text(input_dir: Path, output_file: Path) -> str:
    return (
        "PCA-based assistant-like axis\n"
        "============================\n\n"
        "This folder stores a per-layer PCA axis computed from a filtered set of trait vectors.\n"
        "Method:\n"
        "1. Load all trait vectors from the input directory.\n"
        "2. Stack vectors into shape [n_traits, n_layers, hidden_dim].\n"
        "3. For each layer independently, run PCA over the trait vectors.\n"
        "4. Take PC1 at each layer as the saved axis.\n\n"
        "Important note:\n"
        "This is a PCA-based assistant-like axis (PC1 of trait space), not the paper's contrast-vector Assistant Axis.\n\n"
        f"Input directory: {input_dir}\n"
        f"Output file: {output_file}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Compute PCA-based assistant-like axis from trait vectors")
    parser.add_argument("--vectors_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", default="assistant_axis_pc1")
    args = parser.parse_args()

    vectors_dir = Path(args.vectors_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(vectors_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt vector files found in {vectors_dir}")

    vectors: List[torch.Tensor] = []
    trait_names: List[str] = []

    print(f"\nFound {len(files)} trait vectors in {vectors_dir}")

    for f in tqdm(files, desc="Loading vectors"):
        vec, trait_name = load_vector_file(f)
        vectors.append(vec)
        trait_names.append(trait_name)

    stacked = torch.stack(vectors)  # [n_traits, n_layers, hidden_dim]
    n_traits, n_layers, hidden_dim = stacked.shape

    print("\nStacked tensor shape:", tuple(stacked.shape))

    axis_layers = []
    pc1_variance = []
    k70_list = []
    k90_list = []
    top_traits_by_layer = {}

    print("\nRunning PCA per layer...")

    for layer in range(n_layers):
        X = stacked[:, layer, :]
        pc1, var_ratio, k70, k90 = pca_stats(X)

        axis_layers.append(pc1)
        pc1_variance.append(float(var_ratio[0].item()))
        k70_list.append(int(k70))
        k90_list.append(int(k90))

        sims = []
        for i, trait_name in enumerate(trait_names):
            sims.append((trait_name, cosine_similarity(stacked[i, layer, :], pc1)))

        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
        top_traits_by_layer[str(layer)] = {
            "top_10": [{"trait": name, "cosine": sim} for name, sim in sims_sorted[:10]],
            "bottom_10": [{"trait": name, "cosine": sim} for name, sim in sims_sorted[-10:]],
        }

    axis = torch.stack(axis_layers)  # [n_layers, hidden_dim]

    norms = axis.norm(dim=1)
    top_norm_layers = torch.topk(norms, min(10, len(norms)))

    save_payload = {
        "axis": axis,
        "method": "pc1_per_layer_over_trait_vectors",
        "type": "assistant_axis_pc1",
        "n_traits": n_traits,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "trait_names": trait_names,
        "pc1_variance_per_layer": pc1_variance,
        "pc1_variance_mean": sum(pc1_variance) / len(pc1_variance),
        "k70_per_layer": k70_list,
        "k70_mean": sum(k70_list) / len(k70_list),
        "k90_per_layer": k90_list,
        "k90_mean": sum(k90_list) / len(k90_list),
        "top_norm_layers": [
            {"layer": int(idx.item()), "norm": float(norms[idx].item())}
            for idx in top_norm_layers.indices
        ],
        "top_traits_by_layer": top_traits_by_layer,
        "input_vectors_dir": str(vectors_dir.resolve()),
        "created_at_utc": utc_now_iso(),
        "git_commit": safe_git_commit(),
    }

    output_file = output_dir / f"{args.output_name}.pt"
    torch.save(save_payload, output_file)

    write_json(output_dir / f"{args.output_name}_summary.json", {
        "method": save_payload["method"],
        "type": save_payload["type"],
        "n_traits": n_traits,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "pc1_variance_mean": save_payload["pc1_variance_mean"],
        "k70_mean": save_payload["k70_mean"],
        "k90_mean": save_payload["k90_mean"],
        "top_norm_layers": save_payload["top_norm_layers"],
        "input_vectors_dir": save_payload["input_vectors_dir"],
        "created_at_utc": save_payload["created_at_utc"],
        "git_commit": save_payload["git_commit"],
    })

    write_text(output_dir / "README.txt", build_readme_text(vectors_dir, output_file))

    print("\nAxis shape:", tuple(axis.shape))
    print("\n=== PCA STATS ===")
    print("\nPC1 variance explained (first 10 layers):")
    for i in range(min(10, n_layers)):
        print(f"Layer {i:2d}: {pc1_variance[i] * 100:.2f}%")

    print("\nAverage PC1 variance explained:")
    print(f"{(sum(pc1_variance) / len(pc1_variance)) * 100:.2f}%")

    print("\nComponents needed for 70% variance:")
    print(f"mean = {sum(k70_list) / len(k70_list):.2f}")

    print("\nComponents needed for 90% variance:")
    print(f"mean = {sum(k90_list) / len(k90_list):.2f}")

    print("\nAxis norms per layer (top 10):")
    for idx in top_norm_layers.indices:
        print(f"L{idx.item():2d} = {norms[idx].item():.4f}")

    print(f"\nSaved PCA-based assistant-like axis → {output_file}")


if __name__ == "__main__":
    main()