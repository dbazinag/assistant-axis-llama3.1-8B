#!/usr/bin/env python3
# Computes a PCA-based assistant-like axis from selected-layer trait vectors.
# Compatible with Gemma traits40 vectors shaped [n_selected_layers, hidden_dim],
# where rows correspond to --vector_layers, e.g. 15,20,25,30,35,40,45.

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from tqdm import tqdm


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_git_commit() -> Optional[str]:
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


def parse_vector_layers(s: Optional[str]) -> Optional[List[int]]:
    if s is None or not s.strip():
        return None
    layers = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not layers:
        raise ValueError("--vector_layers was provided but parsed to an empty list")
    return layers


def load_vector_file(path: Path) -> Tuple[torch.Tensor, str, Dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)

    metadata = {}

    if isinstance(data, dict):
        if "vector" not in data:
            raise ValueError(f"{path} is missing 'vector'. Keys: {list(data.keys())}")
        trait_name = data.get("trait", path.stem)
        metadata = {k: v for k, v in data.items() if k != "vector"}
        return data["vector"].float(), trait_name, metadata

    return data.float(), path.stem, metadata


def pca_stats(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    # X shape: [n_vectors, hidden_dim]
    X = X.float()
    X = X - X.mean(dim=0, keepdim=True)

    q = min(X.shape[0], X.shape[1]) - 1
    if q < 1:
        raise ValueError("Need at least 2 vectors to run PCA")

    _, S, V = torch.pca_lowrank(X, q=q)

    var = S**2
    var_ratio = var / (var.sum() + 1e-12)
    cumulative = torch.cumsum(var_ratio, dim=0)

    k70 = int((cumulative >= 0.70).nonzero()[0].item() + 1)
    k90 = int((cumulative >= 0.90).nonzero()[0].item() + 1)

    pc1 = V[:, 0]
    return pc1, var_ratio, k70, k90


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().unsqueeze(0),
        b.float().unsqueeze(0),
    ).item()


def build_readme_text(
    input_dir: Path,
    output_file: Path,
    vector_layers: Optional[List[int]],
) -> str:
    if vector_layers is None:
        layer_text = (
            "The vector rows are treated as row indices 0..n_layers-1 because no --vector_layers "
            "mapping was provided.\n"
        )
    else:
        layer_text = (
            "The saved axis rows correspond to selected original model layers:\n"
            f"{vector_layers}\n\n"
            "For example, if vector_layers = [15,20,25,30,35,40,45], then axis row 3 is model layer 30.\n"
        )

    return (
        "PCA-based assistant-like axis\n"
        "=============================\n\n"
        "This folder stores a PCA axis computed from a set of trait vectors.\n\n"
        "Method:\n"
        "1. Load all trait vectors from the input directory.\n"
        "2. Stack vectors into shape [n_traits, n_selected_layers, hidden_dim].\n"
        "3. For each selected layer row independently, run PCA over the trait vectors.\n"
        "4. Take PC1 at each selected layer row as the saved axis.\n\n"
        "Important note:\n"
        "This is a PCA-based assistant-like axis, PC1 of trait-vector space.\n"
        "It is not necessarily the paper's exact contrast-vector Assistant Axis.\n\n"
        f"{layer_text}\n"
        f"Input directory: {input_dir}\n"
        f"Output file: {output_file}\n"
    )


def validate_vectors(vectors: List[torch.Tensor], files: List[Path]) -> Tuple[int, int]:
    if not vectors:
        raise ValueError("No vectors loaded")

    first_shape = tuple(vectors[0].shape)

    if len(first_shape) != 2:
        raise ValueError(f"Expected 2D vectors, got first vector shape {first_shape}")

    for vec, path in zip(vectors, files):
        if tuple(vec.shape) != first_shape:
            raise ValueError(
                f"Shape mismatch. Expected {first_shape}, got {tuple(vec.shape)} for {path}"
            )

    return first_shape[0], first_shape[1]


def main():
    parser = argparse.ArgumentParser(
        description="Compute PCA-based assistant-like axis from selected-layer trait vectors"
    )
    parser.add_argument("--vectors_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", default="assistant_axis_pc1")
    parser.add_argument(
        "--vector_layers",
        type=str,
        default=None,
        help="Comma-separated original model layers represented by vector rows, e.g. 15,20,25,30,35,40,45",
    )
    args = parser.parse_args()

    vectors_dir = Path(args.vectors_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vector_layers = parse_vector_layers(args.vector_layers)

    files = sorted(vectors_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt vector files found in {vectors_dir}")

    vectors: List[torch.Tensor] = []
    trait_names: List[str] = []
    per_trait_metadata: Dict[str, Dict] = {}

    print(f"\nFound {len(files)} trait vectors in {vectors_dir}")

    for f in tqdm(files, desc="Loading vectors"):
        vec, trait_name, metadata = load_vector_file(f)
        vectors.append(vec)
        trait_names.append(trait_name)
        per_trait_metadata[trait_name] = {
            "source_file": str(f.resolve()),
            "activation_position": metadata.get("activation_position"),
            "filter_name": metadata.get("filter_name"),
            "passed_filter": metadata.get("passed_filter"),
            "positive_mean_score": metadata.get("positive_mean_score"),
            "negative_mean_score": metadata.get("negative_mean_score"),
            "overall_score_diff": metadata.get("overall_score_diff"),
        }

    n_selected_layers, hidden_dim = validate_vectors(vectors, files)

    if vector_layers is not None and len(vector_layers) != n_selected_layers:
        raise ValueError(
            f"--vector_layers has {len(vector_layers)} entries, but vectors have "
            f"{n_selected_layers} selected-layer rows"
        )

    if vector_layers is None:
        vector_layers = list(range(n_selected_layers))

    stacked = torch.stack(vectors)  # [n_traits, n_selected_layers, hidden_dim]
    n_traits, n_layers_from_tensor, hidden_dim_from_tensor = stacked.shape

    print("\nStacked tensor shape:", tuple(stacked.shape))
    print("Vector row → original model layer mapping:")
    for row_idx, model_layer in enumerate(vector_layers):
        print(f"  row {row_idx} -> model layer {model_layer}")

    axis_layers = []
    pc1_variance = []
    k70_list = []
    k90_list = []
    top_traits_by_layer = {}

    print("\nRunning PCA per selected layer row...")

    for row_idx in range(n_selected_layers):
        model_layer = vector_layers[row_idx]
        X = stacked[:, row_idx, :]

        pc1, var_ratio, k70, k90 = pca_stats(X)

        axis_layers.append(pc1)
        pc1_variance.append(float(var_ratio[0].item()))
        k70_list.append(int(k70))
        k90_list.append(int(k90))

        sims = []
        for i, trait_name in enumerate(trait_names):
            sim = cosine_similarity(stacked[i, row_idx, :], pc1)
            sims.append((trait_name, sim))

        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)

        top_traits_by_layer[str(model_layer)] = {
            "row_index": row_idx,
            "model_layer": model_layer,
            "top_10": [{"trait": name, "cosine": sim} for name, sim in sims_sorted[:10]],
            "bottom_10": [{"trait": name, "cosine": sim} for name, sim in sims_sorted[-10:]],
        }

    axis = torch.stack(axis_layers)  # [n_selected_layers, hidden_dim]

    norms = axis.norm(dim=1)
    top_norm_rows = torch.topk(norms, min(10, len(norms)))

    save_payload = {
        "axis": axis,
        "vector": axis,
        "method": "pc1_per_selected_layer_over_trait_vectors",
        "type": "assistant_axis_pc1",
        "n_traits": n_traits,
        "n_layers": n_selected_layers,
        "n_selected_layers": n_selected_layers,
        "hidden_dim": hidden_dim,
        "hidden_dim_from_tensor": hidden_dim_from_tensor,
        "vector_layers": vector_layers,
        "row_to_model_layer": {str(i): int(layer) for i, layer in enumerate(vector_layers)},
        "trait_names": trait_names,
        "pc1_variance_per_row": pc1_variance,
        "pc1_variance_per_model_layer": {
            str(layer): pc1_variance[i] for i, layer in enumerate(vector_layers)
        },
        "pc1_variance_mean": sum(pc1_variance) / len(pc1_variance),
        "k70_per_row": k70_list,
        "k70_per_model_layer": {
            str(layer): k70_list[i] for i, layer in enumerate(vector_layers)
        },
        "k70_mean": sum(k70_list) / len(k70_list),
        "k90_per_row": k90_list,
        "k90_per_model_layer": {
            str(layer): k90_list[i] for i, layer in enumerate(vector_layers)
        },
        "k90_mean": sum(k90_list) / len(k90_list),
        "top_norm_layers": [
            {
                "row_index": int(idx.item()),
                "model_layer": int(vector_layers[int(idx.item())]),
                "norm": float(norms[idx].item()),
            }
            for idx in top_norm_rows.indices
        ],
        "top_traits_by_model_layer": top_traits_by_layer,
        "per_trait_metadata": per_trait_metadata,
        "input_vectors_dir": str(vectors_dir.resolve()),
        "created_at_utc": utc_now_iso(),
        "git_commit": safe_git_commit(),
    }

    output_file = output_dir / f"{args.output_name}.pt"
    torch.save(save_payload, output_file)

    write_json(
        output_dir / f"{args.output_name}_summary.json",
        {
            "method": save_payload["method"],
            "type": save_payload["type"],
            "n_traits": n_traits,
            "n_selected_layers": n_selected_layers,
            "hidden_dim": hidden_dim,
            "vector_layers": vector_layers,
            "row_to_model_layer": save_payload["row_to_model_layer"],
            "pc1_variance_mean": save_payload["pc1_variance_mean"],
            "k70_mean": save_payload["k70_mean"],
            "k90_mean": save_payload["k90_mean"],
            "top_norm_layers": save_payload["top_norm_layers"],
            "input_vectors_dir": save_payload["input_vectors_dir"],
            "created_at_utc": save_payload["created_at_utc"],
            "git_commit": save_payload["git_commit"],
        },
    )

    write_text(output_dir / "README.txt", build_readme_text(vectors_dir, output_file, vector_layers))

    print("\nAxis shape:", tuple(axis.shape))

    print("\n=== PCA STATS ===")
    print("\nPC1 variance explained by selected model layer:")
    for row_idx, model_layer in enumerate(vector_layers):
        print(f"row {row_idx:2d} / model layer {model_layer:2d}: {pc1_variance[row_idx] * 100:.2f}%")

    print("\nAverage PC1 variance explained:")
    print(f"{(sum(pc1_variance) / len(pc1_variance)) * 100:.2f}%")

    print("\nComponents needed for 70% variance:")
    print(f"mean = {sum(k70_list) / len(k70_list):.2f}")

    print("\nComponents needed for 90% variance:")
    print(f"mean = {sum(k90_list) / len(k90_list):.2f}")

    print("\nAxis norms by selected model layer:")
    for row_idx, model_layer in enumerate(vector_layers):
        print(f"row {row_idx:2d} / model layer {model_layer:2d}: {norms[row_idx].item():.4f}")

    print("\nTop norm rows:")
    for idx in top_norm_rows.indices:
        row_idx = int(idx.item())
        print(
            f"row {row_idx:2d} / model layer {vector_layers[row_idx]:2d} = "
            f"{norms[row_idx].item():.4f}"
        )

    print(f"\nSaved PCA-based assistant-like axis → {output_file}")


if __name__ == "__main__":
    main()