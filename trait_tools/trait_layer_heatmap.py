# small interview note: fit one standardized logistic-regression model per layer, then visualize signed trait coefficients across layers in a multi-page PDF heatmap

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VALID_LABELS = {
    "benign",
    "successful_jailbreak",
    "unsuccessful_jailbreak",
}

AXIS_NAME = "axis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one logistic regression per layer, save coefficient tables, and render multi-page PDF heatmaps."
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
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--metric_type",
        type=str,
        default="last_prompt_proj",
        choices=[
            "last_prompt_proj",
            "mean_response_proj",
            "delta_proj",
            "last_prompt_cos",
            "mean_response_cos",
            "delta_cos",
        ],
        help="Which metric family to use from each axis/trait entry.",
    )
    parser.add_argument(
        "--feature_source",
        type=str,
        default="traits_plus_axis",
        choices=["traits_only", "traits_plus_axis", "axis_only"],
        help="Whether to include only traits, traits plus the assistant axis, or just the axis.",
    )
    parser.add_argument(
        "--positive_class",
        type=str,
        default="successful_jailbreak",
        choices=["successful_jailbreak"],
        help="Positive class for binary classification.",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of CV folds for per-layer performance estimates.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=5000,
        help="Max iterations for logistic regression.",
    )
    parser.add_argument(
        "--c_value",
        type=float,
        default=1.0,
        help="Inverse regularization strength.",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help="Whether to use class-balanced loss.",
    )
    parser.add_argument(
        "--traits_per_page",
        type=int,
        default=40,
        help="Number of traits per PDF heatmap page.",
    )
    parser.add_argument(
        "--top_k_per_layer",
        type=int,
        default=10,
        help="How many strongest positive/negative features to save per layer in the rankings CSV.",
    )
    parser.add_argument(
        "--drop_axis_from_heatmap",
        action="store_true",
        help="If set, exclude the assistant axis row from the trait heatmap/table output.",
    )
    parser.add_argument(
        "--debug_print_schema",
        action="store_true",
        help="Print schema info inferred from the JSONL.",
    )
    return parser.parse_args()


def metric_key_from_metric_type(metric_type: str) -> str:
    mapping = {
        "last_prompt_proj": "last_prompt_proj_all_layers",
        "mean_response_proj": "mean_response_proj_all_layers",
        "delta_proj": "delta_proj_all_layers",
        "last_prompt_cos": "last_prompt_cos_all_layers",
        "mean_response_cos": "mean_response_cos_all_layers",
        "delta_cos": "delta_cos_all_layers",
    }
    return mapping[metric_type]


def load_rows(summary_jsonl: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(summary_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = obj.get("prompt_label")
            if label not in VALID_LABELS:
                raise ValueError(f"Unexpected prompt_label at line {line_idx + 1}: {label!r}")
            rows.append(obj)
    if not rows:
        raise ValueError("No rows found in input JSONL.")
    return rows


def debug_print_schema(rows: List[Dict]) -> None:
    first = rows[0]
    print("=" * 80)
    print("DEBUG SCHEMA")
    print("=" * 80)
    print("Top-level keys:", list(first.keys()))
    print("Axis keys:", list(first.get("axis", {}).keys()))
    print("Trait names in first row:", list(first.get("traits", {}).keys()))
    print("=" * 80)


def validate_32_float_array(values: List[float], where: str) -> np.ndarray:
    if not isinstance(values, list):
        raise TypeError(f"{where} must be a list.")
    if len(values) != 32:
        raise ValueError(f"{where} must have length 32, got {len(values)}.")
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{where} contains non-finite values.")
    return arr


def get_trait_names(rows: List[Dict]) -> List[str]:
    names = set()
    for row in rows:
        for name in row.get("traits", {}).keys():
            names.add(name)
    return sorted(names)


def extract_entity_vector(row: Dict, entity_name: str, metric_key: str) -> np.ndarray:
    if entity_name == AXIS_NAME:
        axis_section = row.get("axis", {})
        if metric_key not in axis_section:
            raise KeyError(f"Missing axis.{metric_key}")
        return validate_32_float_array(axis_section[metric_key], f"axis.{metric_key}")

    trait_section = row.get("traits", {}).get(entity_name)
    if trait_section is None or metric_key not in trait_section:
        return np.zeros(32, dtype=np.float64)
    return validate_32_float_array(trait_section[metric_key], f"{entity_name}.{metric_key}")


def get_feature_names(rows: List[Dict], feature_source: str) -> List[str]:
    trait_names = get_trait_names(rows)
    if feature_source == "traits_only":
        return trait_names
    if feature_source == "traits_plus_axis":
        return [AXIS_NAME] + trait_names
    if feature_source == "axis_only":
        return [AXIS_NAME]
    raise ValueError(f"Unknown feature_source={feature_source!r}")


def label_to_target(label: str, positive_class: str) -> int:
    return 1 if label == positive_class else 0


def build_layer_dataset(
    rows: List[Dict],
    feature_names: List[str],
    metric_key: str,
    layer_idx: int,
    positive_class: str,
) -> Tuple[np.ndarray, np.ndarray]:
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []

    for row in rows:
        feats = []
        for feature_name in feature_names:
            vec = extract_entity_vector(row, feature_name, metric_key)
            feats.append(vec[layer_idx])
        X_rows.append(np.asarray(feats, dtype=np.float64))
        y_rows.append(label_to_target(row["prompt_label"], positive_class))

    X = np.stack(X_rows, axis=0).astype(np.float64)
    y = np.asarray(y_rows, dtype=np.int64)

    if not np.all(np.isfinite(X)):
        raise ValueError(f"Non-finite values found in X for layer {layer_idx}.")
    return X, y


def build_pipeline(
    c_value: float,
    max_iter: int,
    random_state: int,
    class_weight: str,
) -> Pipeline:
    lr_class_weight = None if class_weight == "none" else "balanced"
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    solver="liblinear",
                    max_iter=max_iter,
                    random_state=random_state,
                    class_weight=lr_class_weight,
                ),
            ),
        ]
    )


def evaluate_layer_cv(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    cv_folds: int,
    random_state: int,
) -> Dict[str, float]:
    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    roc_aucs: List[float] = []
    f1s: List[float] = []
    aps: List[float] = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]

        roc_aucs.append(float(roc_auc_score(y_test, y_score)))
        f1s.append(float(f1_score(y_test, y_pred, zero_division=0)))
        aps.append(float(average_precision_score(y_test, y_score)))

    return {
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs, ddof=1) if len(roc_aucs) > 1 else 0.0),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0),
        "average_precision_mean": float(np.mean(aps)),
        "average_precision_std": float(np.std(aps, ddof=1) if len(aps) > 1 else 0.0),
    }


def fit_layer_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
) -> Tuple[np.ndarray, float]:
    pipeline.fit(X, y)
    clf: LogisticRegression = pipeline.named_steps["clf"]
    coef = clf.coef_.ravel().astype(np.float64)
    intercept = float(clf.intercept_.ravel()[0])
    return coef, intercept


def make_metrics_plot(metrics_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(11, 6))
    plt.plot(metrics_df["layer"], metrics_df["roc_auc_mean"], label="ROC-AUC")
    plt.plot(metrics_df["layer"], metrics_df["f1_mean"], label="F1")
    plt.plot(metrics_df["layer"], metrics_df["average_precision_mean"], label="Average Precision")
    plt.xlabel("Layer")
    plt.ylabel("CV Metric")
    plt.title("Per-Layer Logistic Regression Performance")
    plt.xticks(np.arange(0, 32, 2))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def make_top_rankings_table(
    coef_df: pd.DataFrame,
    top_k_per_layer: int,
) -> pd.DataFrame:
    rows = []
    for layer_idx in coef_df.columns:
        col = coef_df[layer_idx].sort_values(ascending=False)

        top_pos = col.head(top_k_per_layer)
        top_neg = col.tail(top_k_per_layer)

        for feature, coef in top_pos.items():
            rows.append(
                {
                    "layer": int(layer_idx),
                    "feature": feature,
                    "coefficient": float(coef),
                    "odds_ratio": float(math.exp(coef)),
                    "direction": "toward_successful_jailbreak",
                    "rank_within_direction": int(np.where(top_pos.index == feature)[0][0] + 1),
                }
            )

        for feature, coef in top_neg.items():
            rows.append(
                {
                    "layer": int(layer_idx),
                    "feature": feature,
                    "coefficient": float(coef),
                    "odds_ratio": float(math.exp(coef)),
                    "direction": "away_from_successful_jailbreak",
                    "rank_within_direction": int(np.where(top_neg.index == feature)[0][0] + 1),
                }
            )

    return pd.DataFrame(rows)


def render_heatmap_pages_to_pdf(
    coef_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    out_pdf_path: Path,
    traits_per_page: int,
    title_prefix: str,
) -> None:
    max_abs = float(np.abs(coef_df.to_numpy()).max())
    if max_abs == 0.0:
        max_abs = 1.0

    n_traits = coef_df.shape[0]
    n_pages = math.ceil(n_traits / traits_per_page)

    with PdfPages(out_pdf_path) as pdf:
        # cover page with metric curves
        fig = plt.figure(figsize=(14, 8))
        plt.plot(metrics_df["layer"], metrics_df["roc_auc_mean"], label="ROC-AUC")
        plt.plot(metrics_df["layer"], metrics_df["f1_mean"], label="F1")
        plt.plot(metrics_df["layer"], metrics_df["average_precision_mean"], label="Average Precision")
        plt.xlabel("Layer")
        plt.ylabel("CV Metric")
        plt.title(f"{title_prefix}: Per-Layer Logistic Regression Performance")
        plt.xticks(np.arange(0, 32, 2))
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for page_idx in range(n_pages):
            start = page_idx * traits_per_page
            end = min((page_idx + 1) * traits_per_page, n_traits)
            chunk_df = coef_df.iloc[start:end]

            fig_height = max(8, 0.33 * chunk_df.shape[0] + 2.5)
            fig, ax = plt.subplots(figsize=(16, fig_height))

            im = ax.imshow(
                chunk_df.to_numpy(),
                aspect="auto",
                cmap="coolwarm",
                vmin=-max_abs,
                vmax=max_abs,
            )

            ax.set_xticks(np.arange(chunk_df.shape[1]))
            ax.set_xticklabels([str(c) for c in chunk_df.columns], rotation=90)
            ax.set_yticks(np.arange(chunk_df.shape[0]))
            ax.set_yticklabels(chunk_df.index.tolist())
            ax.set_xlabel("Layer")
            ax.set_ylabel("Trait")
            ax.set_title(
                f"{title_prefix}: Signed Logistic Coefficients by Layer\n"
                f"Traits {start + 1}-{end} of {n_traits}"
            )

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Coefficient (red=toward jailbreak, blue=away)")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # last page with absolute-value heatmap overview
        fig_height = max(8, 0.18 * coef_df.shape[0] + 2.5)
        fig, ax = plt.subplots(figsize=(16, fig_height))
        abs_arr = np.abs(coef_df.to_numpy())
        im = ax.imshow(abs_arr, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(coef_df.shape[1]))
        ax.set_xticklabels([str(c) for c in coef_df.columns], rotation=90)
        ax.set_yticks(np.arange(coef_df.shape[0]))
        ax.set_yticklabels(coef_df.index.tolist())
        ax.set_xlabel("Layer")
        ax.set_ylabel("Trait")
        ax.set_title(f"{title_prefix}: Absolute Coefficient Magnitudes by Layer")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("|Coefficient|")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_key = metric_key_from_metric_type(args.metric_type)
    rows = load_rows(args.summary_jsonl)

    if args.debug_print_schema:
        debug_print_schema(rows)

    feature_names = get_feature_names(rows, args.feature_source)

    metrics_rows = []
    coef_matrix = []
    intercepts = []

    for layer_idx in range(32):
        X, y = build_layer_dataset(
            rows=rows,
            feature_names=feature_names,
            metric_key=metric_key,
            layer_idx=layer_idx,
            positive_class=args.positive_class,
        )

        pipeline = build_pipeline(
            c_value=args.c_value,
            max_iter=args.max_iter,
            random_state=args.random_state,
            class_weight=args.class_weight,
        )

        metrics = evaluate_layer_cv(
            X=X,
            y=y,
            pipeline=pipeline,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )
        coef, intercept = fit_layer_coefficients(
            X=X,
            y=y,
            pipeline=pipeline,
        )

        metrics_rows.append(
            {
                "layer": layer_idx,
                **metrics,
            }
        )
        coef_matrix.append(coef)
        intercepts.append(intercept)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("layer").reset_index(drop=True)

    coef_arr = np.stack(coef_matrix, axis=1)  # [n_features, 32]
    coef_df = pd.DataFrame(coef_arr, index=feature_names, columns=list(range(32)))

    if args.drop_axis_from_heatmap and AXIS_NAME in coef_df.index:
        coef_df_for_heatmap = coef_df.drop(index=AXIS_NAME)
    else:
        coef_df_for_heatmap = coef_df.copy()

    intercept_df = pd.DataFrame(
        {
            "layer": list(range(32)),
            "intercept": intercepts,
        }
    )

    rankings_df = make_top_rankings_table(
        coef_df=coef_df,
        top_k_per_layer=args.top_k_per_layer,
    )

    metrics_df.to_csv(out_dir / "per_layer_metrics.csv", index=False)
    coef_df.to_csv(out_dir / "trait_coefficients_by_layer.csv")
    intercept_df.to_csv(out_dir / "intercepts_by_layer.csv", index=False)
    rankings_df.to_csv(out_dir / "top_ranked_features_by_layer.csv", index=False)

    make_metrics_plot(
        metrics_df=metrics_df,
        out_path=out_dir / "per_layer_logreg_metrics.png",
    )

    render_heatmap_pages_to_pdf(
        coef_df=coef_df_for_heatmap,
        metrics_df=metrics_df,
        out_pdf_path=out_dir / "trait_importance_vs_layer_heatmap.pdf",
        traits_per_page=args.traits_per_page,
        title_prefix=f"{args.metric_type} / {args.feature_source}",
    )

    run_config = {
        "summary_jsonl": args.summary_jsonl,
        "metric_type": args.metric_type,
        "metric_key": metric_key,
        "feature_source": args.feature_source,
        "positive_class": args.positive_class,
        "cv_folds": args.cv_folds,
        "random_state": args.random_state,
        "max_iter": args.max_iter,
        "c_value": args.c_value,
        "class_weight": args.class_weight,
        "traits_per_page": args.traits_per_page,
        "top_k_per_layer": args.top_k_per_layer,
        "drop_axis_from_heatmap": args.drop_axis_from_heatmap,
        "n_examples": len(rows),
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("Done.")
    print(f"Outputs written to: {out_dir}")
    print(f"Main files:")
    print(f"  {out_dir / 'trait_importance_vs_layer_heatmap.pdf'}")
    print(f"  {out_dir / 'trait_coefficients_by_layer.csv'}")
    print(f"  {out_dir / 'top_ranked_features_by_layer.csv'}")
    print(f"  {out_dir / 'per_layer_metrics.csv'}")


if __name__ == "__main__":
    main()