# plots layer-wise assistant-axis separation and projection distributions
# interview-sized note: focuses on last_prompt_proj_all_layers because that is the cleanest pre-response jailbreak-state signal

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


VALID_LABELS = [
    "benign",
    "successful_jailbreak",
    "unsuccessful_jailbreak",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create assistant-axis layerwise and distribution plots from projection_summary.jsonl."
    )
    parser.add_argument(
        "--summary_jsonl",
        type=str,
        required=True,
        help="Path to projection_summary.jsonl",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for PNG figures",
    )
    parser.add_argument(
        "--metric_key",
        type=str,
        default="last_prompt_proj_all_layers",
        choices=[
            "last_prompt_proj_all_layers",
            "mean_response_proj_all_layers",
            "delta_proj_all_layers",
            "last_prompt_cos_all_layers",
            "mean_response_cos_all_layers",
            "delta_cos_all_layers",
        ],
        help="Which axis metric to plot.",
    )
    parser.add_argument(
        "--focus_layer",
        type=int,
        default=16,
        help="Primary layer to use for the main distribution plots.",
    )
    parser.add_argument(
        "--avg_mode",
        type=str,
        default="mean_all_layers",
        choices=["mean_all_layers", "none"],
        help="Whether to also make a distribution plot of the mean across all 32 layers.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of bins for histograms.",
    )
    parser.add_argument(
        "--make_all_layer_hists",
        action="store_true",
        help="If set, save one histogram per layer (0..31).",
    )
    parser.add_argument(
        "--title_prefix",
        type=str,
        default="assistant_axis",
        help="Prefix used in plot titles and filenames.",
    )
    return parser.parse_args()


def load_rows(summary_jsonl: str, metric_key: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(summary_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            label = obj.get("prompt_label")
            if label not in VALID_LABELS:
                raise ValueError(
                    f"Unexpected prompt_label at line {line_idx + 1}: {label!r}"
                )

            axis = obj.get("axis", {})
            values = axis.get(metric_key)
            if values is None:
                raise KeyError(
                    f"Missing axis[{metric_key!r}] at line {line_idx + 1}"
                )
            if not isinstance(values, list):
                raise TypeError(
                    f"axis[{metric_key!r}] must be a list at line {line_idx + 1}"
                )
            if len(values) != 32:
                raise ValueError(
                    f"axis[{metric_key!r}] must have length 32 at line {line_idx + 1}, got {len(values)}"
                )

            arr = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"Non-finite values found in axis[{metric_key!r}] at line {line_idx + 1}"
                )

            rows.append(
                {
                    "prompt_id": obj.get("prompt_id", f"row_{line_idx:05d}"),
                    "prompt_label": label,
                    "method": obj.get("method", "unknown"),
                    "values": arr,
                }
            )
    return rows


def group_by_label(rows: List[Dict]) -> Dict[str, np.ndarray]:
    grouped: Dict[str, List[np.ndarray]] = {label: [] for label in VALID_LABELS}
    for row in rows:
        grouped[row["prompt_label"]].append(row["values"])

    out: Dict[str, np.ndarray] = {}
    for label, items in grouped.items():
        if len(items) == 0:
            raise ValueError(f"No rows found for label={label!r}")
        out[label] = np.stack(items, axis=0)  # [n_prompts, 32]
    return out


def stderr(x: np.ndarray, axis: int = 0) -> np.ndarray:
    n = x.shape[axis]
    if n <= 1:
        return np.zeros_like(np.mean(x, axis=axis))
    return np.std(x, axis=axis, ddof=1) / math.sqrt(n)


def sanitize_filename(s: str) -> str:
    return (
        s.replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def metric_pretty_name(metric_key: str) -> str:
    mapping = {
        "last_prompt_proj_all_layers": "Last Prompt Projection",
        "mean_response_proj_all_layers": "Mean Response Projection",
        "delta_proj_all_layers": "Projection Delta",
        "last_prompt_cos_all_layers": "Last Prompt Cosine Similarity",
        "mean_response_cos_all_layers": "Mean Response Cosine Similarity",
        "delta_cos_all_layers": "Cosine Delta",
    }
    return mapping.get(metric_key, metric_key)


def make_layerwise_mean_plot(
    grouped: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
    y_label: str,
) -> None:
    layers = np.arange(32)

    plt.figure(figsize=(11, 6))
    for label in VALID_LABELS:
        arr = grouped[label]  # [n, 32]
        mean = arr.mean(axis=0)
        se = stderr(arr, axis=0)
        plt.plot(layers, mean, label=label, linewidth=2)
        plt.fill_between(layers, mean - se, mean + se, alpha=0.2)

    plt.axvline(16, linestyle="--", linewidth=1, alpha=0.8)
    plt.xticks(np.arange(0, 32, 2))
    plt.xlabel("Layer")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_layerwise_boxplot(
    grouped: Dict[str, np.ndarray],
    focus_layer: int,
    out_path: Path,
    title: str,
    y_label: str,
) -> None:
    data = [grouped[label][:, focus_layer] for label in VALID_LABELS]

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=VALID_LABELS, showfliers=False)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_histogram_overlay(
    grouped_1d: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
    x_label: str,
    bins: int,
) -> None:
    all_values = np.concatenate([grouped_1d[label] for label in VALID_LABELS], axis=0)
    x_min = float(np.min(all_values))
    x_max = float(np.max(all_values))
    bin_edges = np.linspace(x_min, x_max, bins + 1)

    plt.figure(figsize=(10, 6))
    for label in VALID_LABELS:
        plt.hist(
            grouped_1d[label],
            bins=bin_edges,
            density=True,
            alpha=0.4,
            label=label,
        )

    plt.xlabel(x_label)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_violin_plot(
    grouped_1d: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
    y_label: str,
) -> None:
    data = [grouped_1d[label] for label in VALID_LABELS]

    plt.figure(figsize=(8, 6))
    plt.violinplot(data, showmeans=True, showmedians=False, showextrema=True)
    plt.xticks(np.arange(1, len(VALID_LABELS) + 1), VALID_LABELS)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def extract_single_layer(grouped: Dict[str, np.ndarray], layer_idx: int) -> Dict[str, np.ndarray]:
    return {label: grouped[label][:, layer_idx] for label in VALID_LABELS}


def extract_mean_all_layers(grouped: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {label: grouped[label].mean(axis=1) for label in VALID_LABELS}


def save_summary_stats(
    grouped: Dict[str, np.ndarray],
    metric_key: str,
    focus_layer: int,
    out_path: Path,
) -> None:
    stats: Dict[str, Dict] = {
        "metric_key": metric_key,
        "focus_layer": focus_layer,
        "labels": {},
    }

    for label in VALID_LABELS:
        arr = grouped[label]
        stats["labels"][label] = {
            "n": int(arr.shape[0]),
            "per_layer_mean": arr.mean(axis=0).tolist(),
            "per_layer_std": arr.std(axis=0, ddof=1).tolist(),
            "focus_layer_mean": float(arr[:, focus_layer].mean()),
            "focus_layer_std": float(arr[:, focus_layer].std(ddof=1)),
            "all_layer_mean_of_prompt_means": float(arr.mean(axis=1).mean()),
            "all_layer_std_of_prompt_means": float(arr.mean(axis=1).std(ddof=1)),
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def main() -> None:
    args = parse_args()

    if not (0 <= args.focus_layer < 32):
        raise ValueError("--focus_layer must be in [0, 31]")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.summary_jsonl, args.metric_key)
    grouped = group_by_label(rows)

    pretty_metric = metric_pretty_name(args.metric_key)
    prefix = sanitize_filename(args.title_prefix)
    metric_slug = sanitize_filename(args.metric_key)

    save_summary_stats(
        grouped=grouped,
        metric_key=args.metric_key,
        focus_layer=args.focus_layer,
        out_path=out_dir / f"{prefix}_{metric_slug}_summary_stats.json",
    )

    make_layerwise_mean_plot(
        grouped=grouped,
        out_path=out_dir / f"{prefix}_{metric_slug}_layerwise_mean_separation.png",
        title=f"{args.title_prefix}: {pretty_metric} Across Layers",
        y_label=pretty_metric,
    )

    make_layerwise_boxplot(
        grouped=grouped,
        focus_layer=args.focus_layer,
        out_path=out_dir / f"{prefix}_{metric_slug}_layer_{args.focus_layer}_boxplot.png",
        title=f"{args.title_prefix}: {pretty_metric} at Layer {args.focus_layer}",
        y_label=pretty_metric,
    )

    focus_grouped_1d = extract_single_layer(grouped, args.focus_layer)

    make_histogram_overlay(
        grouped_1d=focus_grouped_1d,
        out_path=out_dir / f"{prefix}_{metric_slug}_layer_{args.focus_layer}_histogram.png",
        title=f"{args.title_prefix}: {pretty_metric} Distribution at Layer {args.focus_layer}",
        x_label=pretty_metric,
        bins=args.bins,
    )

    make_violin_plot(
        grouped_1d=focus_grouped_1d,
        out_path=out_dir / f"{prefix}_{metric_slug}_layer_{args.focus_layer}_violin.png",
        title=f"{args.title_prefix}: {pretty_metric} at Layer {args.focus_layer}",
        y_label=pretty_metric,
    )

    if args.avg_mode == "mean_all_layers":
        avg_grouped_1d = extract_mean_all_layers(grouped)

        make_histogram_overlay(
            grouped_1d=avg_grouped_1d,
            out_path=out_dir / f"{prefix}_{metric_slug}_all_layers_mean_histogram.png",
            title=f"{args.title_prefix}: Mean {pretty_metric} Across All Layers",
            x_label=f"Mean {pretty_metric} Across All Layers",
            bins=args.bins,
        )

        make_violin_plot(
            grouped_1d=avg_grouped_1d,
            out_path=out_dir / f"{prefix}_{metric_slug}_all_layers_mean_violin.png",
            title=f"{args.title_prefix}: Mean {pretty_metric} Across All Layers",
            y_label=f"Mean {pretty_metric} Across All Layers",
        )

    if args.make_all_layer_hists:
        all_layers_dir = out_dir / f"{prefix}_{metric_slug}_all_layer_histograms"
        all_layers_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx in range(32):
            grouped_1d = extract_single_layer(grouped, layer_idx)
            make_histogram_overlay(
                grouped_1d=grouped_1d,
                out_path=all_layers_dir / f"layer_{layer_idx:02d}_histogram.png",
                title=f"{args.title_prefix}: {pretty_metric} Distribution at Layer {layer_idx}",
                x_label=pretty_metric,
                bins=args.bins,
            )

    print(f"Loaded rows: {len(rows)}")
    for label in VALID_LABELS:
        print(f"{label}: {grouped[label].shape[0]} prompts")

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()