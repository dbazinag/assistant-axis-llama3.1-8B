#!/usr/bin/env python3
# Interview note: builds trait vectors from multiple activation positions, saves unfiltered + multiple paper-inspired filtered variants, and writes README files so future-you knows exactly what each folder means.

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm


ACTIVATION_POSITIONS = [
    "answer_mean",
    "pre_generation_last_token",
    "assistant_header_mean",
]


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


def load_scores(scores_file: Path) -> Dict[str, int]:
    with open(scores_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_activations(activations_file: Path) -> Dict[str, torch.Tensor]:
    return torch.load(activations_file, map_location="cpu", weights_only=False)


def parse_label(label: str) -> Tuple[str, int, int]:
    # format: positive_p{prompt_index}_q{question_index}
    # or     negative_p{prompt_index}_q{question_index}
    polarity, p_part, q_part = label.split("_")
    prompt_index = int(p_part[1:])
    question_index = int(q_part[1:])
    return polarity, prompt_index, question_index


def split_by_polarity(
    activations: Dict[str, torch.Tensor],
    scores: Dict[str, int],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int], List[int]]:
    pos_acts, neg_acts, pos_scores, neg_scores = [], [], [], []

    for label, act in activations.items():
        if label not in scores:
            continue
        score = scores[label]
        polarity, _, _ = parse_label(label)
        if polarity == "positive":
            pos_acts.append(act)
            pos_scores.append(score)
        elif polarity == "negative":
            neg_acts.append(act)
            neg_scores.append(score)

    return pos_acts, neg_acts, pos_scores, neg_scores


def compute_overall_score_diff(scores: Dict[str, int]) -> Dict[str, float]:
    pos_scores, neg_scores = [], []
    for label, score in scores.items():
        polarity, _, _ = parse_label(label)
        if polarity == "positive":
            pos_scores.append(score)
        elif polarity == "negative":
            neg_scores.append(score)

    if not pos_scores or not neg_scores:
        raise ValueError("Missing positive or negative scores")

    pos_mean = sum(pos_scores) / len(pos_scores)
    neg_mean = sum(neg_scores) / len(neg_scores)
    return {
        "positive_mean_score": pos_mean,
        "negative_mean_score": neg_mean,
        "overall_score_diff": pos_mean - neg_mean,
    }


def compute_matched_pair_stats(scores: Dict[str, int]) -> Dict:
    matched_diffs = {}
    per_prompt_pair_counts_ge_50 = {p: 0 for p in range(5)}

    for prompt_index in range(5):
        for question_index in range(40):
            pos_label = f"positive_p{prompt_index}_q{question_index}"
            neg_label = f"negative_p{prompt_index}_q{question_index}"

            if pos_label not in scores or neg_label not in scores:
                continue

            diff = scores[pos_label] - scores[neg_label]
            matched_diffs[(prompt_index, question_index)] = diff

            if diff >= 50:
                per_prompt_pair_counts_ge_50[prompt_index] += 1

    total_matched_pairs = len(matched_diffs)
    total_matched_pairs_ge_50 = sum(1 for d in matched_diffs.values() if d >= 50)
    prompt_pairs_with_at_least_10 = sum(1 for c in per_prompt_pair_counts_ge_50.values() if c >= 10)

    return {
        "total_matched_pairs": total_matched_pairs,
        "total_matched_pairs_ge_50": total_matched_pairs_ge_50,
        "per_prompt_pair_counts_ge_50": per_prompt_pair_counts_ge_50,
        "prompt_pairs_with_at_least_10_question_wins": prompt_pairs_with_at_least_10,
    }


def compute_vector(
    activations: Dict[str, torch.Tensor],
    scores: Dict[str, int],
) -> torch.Tensor:
    pos_acts, neg_acts, _, _ = split_by_polarity(activations, scores)

    if not pos_acts:
        raise ValueError("No positive activations found")
    if not neg_acts:
        raise ValueError("No negative activations found")

    pos_mean = torch.stack(pos_acts).mean(dim=0)
    neg_mean = torch.stack(neg_acts).mean(dim=0)
    return pos_mean - neg_mean


def build_save_payload(
    trait: str,
    activation_position: str,
    vector: torch.Tensor,
    score_stats: Dict,
    matched_stats: Dict,
    filter_name: str,
    passed_filter: bool,
) -> Dict:
    return {
        "vector": vector,
        "type": "trait",
        "trait": trait,
        "activation_position": activation_position,
        "filter_name": filter_name,
        "passed_filter": passed_filter,
        **score_stats,
        **matched_stats,
    }


def folder_explanation_text(filter_name: str, activation_position: str) -> str:
    explanations = {
        "all_traits_no_filter": (
            "Vector construction: mean(all positive activations) - mean(all negative activations).\n"
            "No score-based filtering is applied. This is useful if paper filtering was only dataset curation.\n"
        ),
        "filter_overall_mean_score_diff_ge_50": (
            "Trait kept only if overall mean positive judge score minus overall mean negative judge score >= 50.\n"
            "This is a simple trait-level separation filter.\n"
        ),
        "filter_matched_pairs_ge_50_count_ge_10_total": (
            "Trait kept only if at least 10 matched (prompt_index, question_index) pairs satisfy:\n"
            "positive_score - negative_score >= 50.\n"
            "This is a paper-inspired interpretation of the ambiguous appendix wording.\n"
        ),
        "filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs": (
            "For each prompt pair p in {0..4}, count how many of its 40 matched questions satisfy:\n"
            "positive_score - negative_score >= 50.\n"
            "Trait kept only if at least 3 of the 5 prompt pairs each have >= 10 such matched question wins.\n"
            "This is a stricter paper-inspired interpretation.\n"
        ),
    }

    position_text = {
        "answer_mean": (
            "Activation source = answer_mean.\n"
            "This is the main paper-faithful source because the paper says trait vectors use all response tokens.\n"
        ),
        "pre_generation_last_token": (
            "Activation source = pre_generation_last_token.\n"
            "This is a prompt-side ablation using the final token before generation begins.\n"
        ),
        "assistant_header_mean": (
            "Activation source = assistant_header_mean.\n"
            "This is a prompt-side ablation averaging over the assistant header span.\n"
        ),
    }

    return (
        f"Folder purpose\n"
        f"==============\n\n"
        f"{position_text[activation_position]}\n"
        f"{explanations[filter_name]}\n"
        f"Saved file format\n"
        f"=================\n\n"
        f"Each .pt file contains a dict with:\n"
        f"- vector: Tensor[n_layers, hidden_dim]\n"
        f"- trait: trait name\n"
        f"- activation_position\n"
        f"- filter_name\n"
        f"- passed_filter\n"
        f"- score statistics\n"
        f"- matched pair statistics\n"
    )


def get_filter_decisions(score_stats: Dict, matched_stats: Dict) -> Dict[str, bool]:
    return {
        "all_traits_no_filter": True,
        "filter_overall_mean_score_diff_ge_50": score_stats["overall_score_diff"] >= 50,
        "filter_matched_pairs_ge_50_count_ge_10_total": matched_stats["total_matched_pairs_ge_50"] >= 10,
        "filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs": (
            matched_stats["prompt_pairs_with_at_least_10_question_wins"] >= 3
        ),
    }


def ensure_readmes(output_root: Path) -> None:
    for activation_position in ACTIVATION_POSITIONS:
        for filter_name in [
            "all_traits_no_filter",
            "filter_overall_mean_score_diff_ge_50",
            "filter_matched_pairs_ge_50_count_ge_10_total",
            "filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs",
        ]:
            folder = output_root / activation_position / filter_name
            folder.mkdir(parents=True, exist_ok=True)
            readme_path = folder / "README.txt"
            if not readme_path.exists():
                write_text(readme_path, folder_explanation_text(filter_name, activation_position))


def main():
    parser = argparse.ArgumentParser(description="Compute per-trait vectors from multiple activation positions")
    parser.add_argument(
        "--activations_root",
        type=str,
        default="full_trait_output/traits40_activations",
        help="Root folder containing subfolders like answer_mean, pre_generation_last_token, assistant_header_mean",
    )
    parser.add_argument(
        "--scores_dir",
        type=str,
        default="full_trait_output/traits40_judge/scores",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="full_trait_output/traits40_vectors",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    activations_root = Path(args.activations_root)
    scores_dir = Path(args.scores_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ensure_readmes(output_root)

    manifest = {
        "created_at_utc": utc_now_iso(),
        "activations_root": str(activations_root.resolve()),
        "scores_dir": str(scores_dir.resolve()),
        "output_root": str(output_root.resolve()),
        "activation_positions": ACTIVATION_POSITIONS,
        "filters": [
            "all_traits_no_filter",
            "filter_overall_mean_score_diff_ge_50",
            "filter_matched_pairs_ge_50_count_ge_10_total",
            "filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs",
        ],
        "git_commit": safe_git_commit(),
    }
    write_json(output_root / "manifests" / "run_config.json", manifest)

    overall_summary = {}

    for activation_position in ACTIVATION_POSITIONS:
        activation_dir = activations_root / activation_position
        activation_files = sorted(activation_dir.glob("*.pt"))

        if not activation_files:
            print(f"Warning: no activation files found in {activation_dir}")
            continue

        print(f"\n=== Processing activation position: {activation_position} ===")
        print(f"Found {len(activation_files)} activation files")

        summary = {
            "activation_position": activation_position,
            "total_traits_seen": 0,
            "vectors_saved_all_traits_no_filter": 0,
            "vectors_saved_filter_overall_mean_score_diff_ge_50": 0,
            "vectors_saved_filter_matched_pairs_ge_50_count_ge_10_total": 0,
            "vectors_saved_filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs": 0,
            "failed_traits": [],
        }

        for act_file in tqdm(activation_files, desc=f"Vectors[{activation_position}]"):
            trait = act_file.stem
            summary["total_traits_seen"] += 1

            try:
                activations = load_activations(act_file)
                if not activations:
                    raise ValueError("No activations loaded")

                scores_file = scores_dir / f"{trait}.json"
                if not scores_file.exists():
                    raise ValueError("No scores file found")

                scores = load_scores(scores_file)

                score_stats = compute_overall_score_diff(scores)
                matched_stats = compute_matched_pair_stats(scores)
                vector = compute_vector(activations, scores)
                filter_decisions = get_filter_decisions(score_stats, matched_stats)

                for filter_name, passed in filter_decisions.items():
                    if filter_name != "all_traits_no_filter" and not passed:
                        continue

                    out_dir = output_root / activation_position / filter_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / f"{trait}.pt"

                    if out_file.exists() and not args.overwrite:
                        continue

                    payload = build_save_payload(
                        trait=trait,
                        activation_position=activation_position,
                        vector=vector,
                        score_stats=score_stats,
                        matched_stats=matched_stats,
                        filter_name=filter_name,
                        passed_filter=passed,
                    )
                    torch.save(payload, out_file)

                    if filter_name == "all_traits_no_filter":
                        summary["vectors_saved_all_traits_no_filter"] += 1
                    elif filter_name == "filter_overall_mean_score_diff_ge_50":
                        summary["vectors_saved_filter_overall_mean_score_diff_ge_50"] += 1
                    elif filter_name == "filter_matched_pairs_ge_50_count_ge_10_total":
                        summary["vectors_saved_filter_matched_pairs_ge_50_count_ge_10_total"] += 1
                    elif filter_name == "filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs":
                        summary["vectors_saved_filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs"] += 1

            except Exception as e:
                summary["failed_traits"].append(f"{trait}: {str(e)}")

        write_json(output_root / activation_position / "summary.json", summary)
        overall_summary[activation_position] = summary

        print(
            f"{activation_position}: "
            f"all={summary['vectors_saved_all_traits_no_filter']}, "
            f"overall50={summary['vectors_saved_filter_overall_mean_score_diff_ge_50']}, "
            f"matched10total={summary['vectors_saved_filter_matched_pairs_ge_50_count_ge_10_total']}, "
            f"promptpair10_require3of5={summary['vectors_saved_filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs']}, "
            f"failed={len(summary['failed_traits'])}"
        )

    write_json(output_root / "manifests" / "overall_summary.json", overall_summary)
    print("\nDone.")


if __name__ == "__main__":
    main()