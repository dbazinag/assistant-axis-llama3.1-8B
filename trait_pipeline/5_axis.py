#!/usr/bin/env python3
# Builds per-trait contrast axes from trait vector files: axis = mean(pos) - mean(neg)

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_vector_file(vector_file: Path) -> dict:
    """Load one trait vector payload from disk."""
    data = torch.load(vector_file, map_location="cpu", weights_only=False)
    if isinstance(data, torch.Tensor):
        return {"vector": data, "name": vector_file.stem}
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported payload type in {vector_file}: {type(data)}")
    if "vector" not in data:
        raise KeyError(f"Missing 'vector' in {vector_file}. Keys: {list(data.keys())}")
    return data


def infer_trait_name(data: dict, vector_file: Path) -> str:
    """Infer trait name from metadata or filename."""
    for key in ("trait", "name"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    stem = vector_file.stem
    for suffix in ("_pos", "_neg", "-pos", "-neg", ".pos", ".neg"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def infer_polarity(data: dict, vector_file: Path) -> str | None:
    """
    Infer whether a file is positive or negative for a trait.
    Expected values:
      - metadata: polarity/label/side in {"pos","positive","neg","negative"}
      - filename endings like *_pos.pt or *_neg.pt
    """
    for key in ("polarity", "label", "side"):
        val = data.get(key)
        if isinstance(val, str):
            v = val.strip().lower()
            if v in {"pos", "positive", "+"}:
                return "pos"
            if v in {"neg", "negative", "-"}:
                return "neg"

    stem = vector_file.stem.lower()
    if stem.endswith(("_pos", "-pos", ".pos")):
        return "pos"
    if stem.endswith(("_neg", "-neg", ".neg")):
        return "neg"

    return None


def ensure_2d_vector(data: dict, vector_file: Path) -> torch.Tensor:
    """Return float tensor of shape (n_layers, hidden_dim)."""
    vector = data["vector"].float()
    if vector.ndim != 2:
        raise ValueError(f"Expected 2D vector in {vector_file}, got shape {tuple(vector.shape)}")
    return vector


def save_axis_payload(
    output_path: Path,
    axis: torch.Tensor,
    trait: str,
    pos_count: int,
    neg_count: int,
    pos_mean: torch.Tensor,
    neg_mean: torch.Tensor,
) -> None:
    """Save a richer payload so downstream scripts can inspect metadata if needed."""
    payload = {
        "vector": axis.cpu(),
        "trait": trait,
        "type": "contrast_axis",
        "formula": "mean(pos) - mean(neg)",
        "n_pos": int(pos_count),
        "n_neg": int(neg_count),
        "pos_mean": pos_mean.cpu(),
        "neg_mean": neg_mean.cpu(),
    }
    torch.save(payload, output_path)


def print_axis_stats(trait: str, axis: torch.Tensor) -> None:
    norms = axis.norm(dim=1)
    print(f"\nTrait: {trait}")
    print(f"Axis shape: {tuple(axis.shape)}")
    print("Axis norms per layer (first 10):")
    for i, norm in enumerate(norms[:10]):
        print(f"  Layer {i}: {norm:.4f}")
    print("  ...")
    print(f"  Mean norm: {norms.mean():.4f}")
    print(f"  Max norm: {norms.max():.4f} (layer {norms.argmax().item()})")


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-trait contrast axes from positive/negative trait vectors"
    )
    parser.add_argument(
        "--vectors_dir",
        type=str,
        required=True,
        help="Directory containing trait vector .pt files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where per-trait axis files will be written",
    )
    parser.add_argument(
        "--trait",
        type=str,
        default=None,
        help="Optional single trait to process",
    )
    args = parser.parse_args()

    vectors_dir = Path(args.vectors_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vector_files = sorted(vectors_dir.glob("*.pt"))
    print(f"Found {len(vector_files)} vector files in {vectors_dir}")

    if not vector_files:
        print("Error: No .pt files found")
        sys.exit(1)

    grouped: dict[str, dict[str, list[torch.Tensor]]] = {}

    for vec_file in tqdm(vector_files, desc="Loading vectors"):
        data = load_vector_file(vec_file)
        trait = infer_trait_name(data, vec_file)
        polarity = infer_polarity(data, vec_file)

        if args.trait is not None and trait != args.trait:
            continue

        if polarity is None:
            print(f"Skipping {vec_file.name}: could not infer pos/neg polarity")
            continue

        vector = ensure_2d_vector(data, vec_file)

        if trait not in grouped:
            grouped[trait] = {"pos": [], "neg": []}
        grouped[trait][polarity].append(vector)

        print(f"  {vec_file.name}: trait={trait}, polarity={polarity}, shape={tuple(vector.shape)}")

    if not grouped:
        print("Error: No usable trait vectors found")
        sys.exit(1)

    print(f"\nFound {len(grouped)} trait group(s): {sorted(grouped.keys())}")

    any_saved = False

    for trait, buckets in grouped.items():
        pos_vectors = buckets["pos"]
        neg_vectors = buckets["neg"]

        print(
            f"\nProcessing trait '{trait}' "
            f"(pos={len(pos_vectors)}, neg={len(neg_vectors)})"
        )

        if not pos_vectors:
            print(f"  Skipping '{trait}': no positive vectors found")
            continue
        if not neg_vectors:
            print(f"  Skipping '{trait}': no negative vectors found")
            continue

        pos_stacked = torch.stack(pos_vectors)  # (n_pos, n_layers, hidden_dim)
        neg_stacked = torch.stack(neg_vectors)  # (n_neg, n_layers, hidden_dim)

        pos_mean = pos_stacked.mean(dim=0)  # (n_layers, hidden_dim)
        neg_mean = neg_stacked.mean(dim=0)  # (n_layers, hidden_dim)

        axis = pos_mean - neg_mean  # points from negative toward positive trait behavior

        print_axis_stats(trait, axis)

        output_path = output_dir / f"{trait}.pt"
        save_axis_payload(
            output_path=output_path,
            axis=axis,
            trait=trait,
            pos_count=len(pos_vectors),
            neg_count=len(neg_vectors),
            pos_mean=pos_mean,
            neg_mean=neg_mean,
        )
        print(f"  Saved axis to {output_path}")
        any_saved = True

    if not any_saved:
        print("\nError: No trait axes were saved")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()