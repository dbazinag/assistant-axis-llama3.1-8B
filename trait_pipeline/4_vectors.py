#!/usr/bin/env python3
"""
Compute per-trait vectors from activations and scores.

Mirrors scripts/4_vectors.py. The differences from the role version:
  - No score filtering on individual rollouts.
    (Paper Appendix C: "we collected the post-MLP residual stream activation of
    ALL response tokens and subtracted the mean negatively-elicited response
    activation from the mean positively-elicited activation")
  - Instead, the judge scores are used at the TRAIT level: a trait is only kept
    if its pos and neg rollouts show enough separation (mean score difference
    >= min_score_diff on a 0-100 scale).
  - Vector = mean(all positive activations) - mean(all negative activations)

Usage:
    uv run trait_pipeline/4_vectors.py \
        --activations_dir outputs/llama-3.1-8b/trait_activations \
        --scores_dir outputs/llama-3.1-8b/trait_scores \
        --output_dir outputs/llama-3.1-8b/trait_vectors \
        --min_score_diff 20
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_scores(scores_file: Path) -> dict:
    with open(scores_file, 'r') as f:
        return json.load(f)


def load_activations(activations_file: Path) -> dict:
    return torch.load(activations_file, map_location="cpu", weights_only=False)


def compute_trait_vector(
    activations: dict,
    scores: dict,
    min_score_diff: float,
) -> torch.Tensor:
    """
    Compute trait vector = mean(all positive activations) - mean(all negative activations).

    The trait is rejected (ValueError) if the mean score difference between
    positive and negative rollouts is below min_score_diff.

    Keys in activations follow the format set by 1_generate.py:
        positive_p{prompt_index}_q{question_index}
        negative_p{prompt_index}_q{question_index}
    """
    pos_acts = []
    neg_acts = []
    pos_scores = []
    neg_scores = []

    for key, act in activations.items():
        score = scores.get(key)
        if score is None:
            continue
        if key.startswith('positive'):
            pos_acts.append(act)
            pos_scores.append(score)
        elif key.startswith('negative'):
            neg_acts.append(act)
            neg_scores.append(score)

    if not pos_acts:
        raise ValueError("No positive activations found")
    if not neg_acts:
        raise ValueError("No negative activations found")

    mean_pos_score = sum(pos_scores) / len(pos_scores)
    mean_neg_score = sum(neg_scores) / len(neg_scores)
    score_diff = mean_pos_score - mean_neg_score

    if score_diff < min_score_diff:
        raise ValueError(
            f"Score diff {score_diff:.1f} < min_score_diff {min_score_diff} "
            f"(pos_mean={mean_pos_score:.1f}, neg_mean={mean_neg_score:.1f})"
        )

    pos_mean = torch.stack(pos_acts).mean(dim=0)  # (n_layers, hidden_dim)
    neg_mean = torch.stack(neg_acts).mean(dim=0)

    return pos_mean - neg_mean


def main():
    parser = argparse.ArgumentParser(description="Compute per-trait vectors")
    parser.add_argument("--activations_dir", type=str, required=True)
    parser.add_argument("--scores_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--min_score_diff", type=float, default=20.0,
                        help="Minimum mean(pos_scores) - mean(neg_scores) to keep a trait (0-100 scale, default: 20)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activations_dir = Path(args.activations_dir)
    scores_dir = Path(args.scores_dir)

    activation_files = sorted(activations_dir.glob("*.pt"))
    print(f"Found {len(activation_files)} trait activation files")

    successful = 0
    skipped = 0
    failed = 0
    failed_names = []

    for act_file in tqdm(activation_files, desc="Computing trait vectors"):
        trait = act_file.stem
        output_file = output_dir / f"{trait}.pt"

        if output_file.exists() and not args.overwrite:
            skipped += 1
            continue

        activations = load_activations(act_file)
        if not activations:
            print(f"Warning: No activations for {trait}")
            failed += 1
            continue

        scores_file = scores_dir / f"{trait}.json"
        if not scores_file.exists():
            print(f"Warning: No scores file for {trait}")
            failed += 1
            failed_names.append(f"{trait}: no scores")
            continue

        scores = load_scores(scores_file)

        try:
            vector = compute_trait_vector(activations, scores, args.min_score_diff)

            save_data = {
                "vector": vector,   # (n_layers, hidden_dim)
                "type": "trait",
                "trait": trait,
            }
            torch.save(save_data, output_file)
            successful += 1

        except ValueError as e:
            print(f"  Skipping {trait}: {e}")
            failed += 1
            failed_names.append(f"{trait}: {e}")

    print(f"\nSummary: {successful} successful, {skipped} skipped, {failed} failed")
    if failed_names:
        print(f"\nFailed traits ({min(10, len(failed_names))} shown):")
        for name in failed_names[:10]:
            print(f"  {name}")
        if len(failed_names) > 10:
            print(f"  ... and {len(failed_names) - 10} more")


if __name__ == "__main__":
    main()
