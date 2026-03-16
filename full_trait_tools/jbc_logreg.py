# interview note: this version adds --attack_method_filter so you can run the logistic regression on only one regime like JBC

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
        description="Run interpretable logistic regression on assistant-axis / trait projections."
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
        "--attack_method_filter",
        type=str,
        default=None,
        help="Optional comma-separated list of attack_method values to keep, e.g. JBC or JBC,GCG",
    )
    parser.add_argument(
        "--feature_source",
        type=str,
        default="traits_plus_axis",
        choices=[
            "axis_only",
            "traits_only",
            "traits_plus_axis",
        ],
        help="Which features to include in the regression.",
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
        help="Which projection/cosine family to use.",
    )
    parser.add_argument(
        "--layer_mode",
        type=str,
        default="single",
        choices=[
            "single",
            "mean_range",
            "all_layers_concat",
        ],
        help="How to extract features from layers.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=26,
        help="Layer to use when --layer_mode single.",
    )
    parser.add_argument(
        "--layer_start",
        type=int,
        default=24,
        help="Start layer (inclusive) when --layer_mode mean_range.",
    )
    parser.add_argument(
        "--layer_end",
        type=int,
        default=29,
        help="End layer (inclusive) when --layer_mode mean_range.",
    )
    parser.add_argument(
        "--positive_class",
        type=str,
        default="successful_jailbreak",
        choices=["successful_jailbreak"],
        help="Positive target class.",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of stratified CV folds.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for CV and model.",
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
        help="Inverse regularization strength for logistic regression.",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help="Whether to use class-balanced loss.",
    )
    parser.add_argument(
        "--save_per_layer_scan",
        action="store_true",
        help="Also run a scan over all single layers and save CV metrics by layer.",
    )
    parser.add_argument(
        "--debug_print_schema",
        action="store_true",
        help="Print schema information inferred from the first row and trait coverage stats.",
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


def label_to_target(label: str, positive_class: str) -> int:
    return 1 if label == positive_class else 0


def load_rows(summary_jsonl: str, attack_method_filter: str | None = None) -> List[Dict]:
    rows: List[Dict] = []

    allowed_methods = None
    if attack_method_filter:
        allowed_methods = {
            x.strip() for x in attack_method_filter.split(",") if x.strip()
        }

    with open(summary_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt_label = obj.get("prompt_label")
            if prompt_label not in VALID_LABELS:
                raise ValueError(
                    f"Unexpected prompt_label at line {line_idx + 1}: {prompt_label!r}"
                )

            if allowed_methods is not None:
                if obj.get("attack_method") not in allowed_methods:
                    continue

            rows.append(obj)

    if not rows:
        raise ValueError("No rows found in summary_jsonl after filtering.")
    return rows


def validate_32_float_array(values: List[float], where: str) -> np.ndarray:
    if not isinstance(values, list):
        raise TypeError(f"{where} must be a list.")
    if len(values) != 32:
        raise ValueError(f"{where} must have length 32, got {len(values)}.")
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{where} contains non-finite values.")
    return arr


def get_trait_presence_counts(rows: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        for trait_name in row.get("traits", {}).keys():
            counts[trait_name] = counts.get(trait_name, 0) + 1
    return counts


def get_available_trait_names(rows: List[Dict]) -> List[str]:
    counts = get_trait_presence_counts(rows)
    return sorted(counts.keys())


def get_trait_names_present_in_all_rows(rows: List[Dict]) -> List[str]:
    counts = get_trait_presence_counts(rows)
    n_rows = len(rows)
    return sorted([name for name, count in counts.items() if count == n_rows])


def debug_print_schema(rows: List[Dict]) -> None:
    first = rows[0]
    print("=" * 80)
    print("DEBUG SCHEMA")
    print("=" * 80)
    print("Top-level keys:", list(first.keys()))
    print("Axis keys:", list(first.get("axis", {}).keys()))
    print("Trait names in first row:", list(first.get("traits", {}).keys()))

    first_traits = first.get("traits", {})
    if first_traits:
        first_trait_name = sorted(first_traits.keys())[0]
        print(
            f"Metric keys for first trait '{first_trait_name}':",
            list(first_traits[first_trait_name].keys()),
        )

    counts = get_trait_presence_counts(rows)
    print("\nTrait coverage across rows:")
    for trait_name in sorted(counts.keys()):
        print(f"  {trait_name}: {counts[trait_name]}/{len(rows)} rows")

    all_rows_traits = get_trait_names_present_in_all_rows(rows)
    print("\nTraits present in ALL rows:", all_rows_traits)
    print("=" * 80)


def extract_entity_vector(
    obj: Dict,
    entity_name: str,
    metric_key: str,
) -> np.ndarray:
    if entity_name == AXIS_NAME:
        section = obj.get("axis", {})
        if metric_key not in section:
            raise KeyError(f"Missing axis.{metric_key}")
        return validate_32_float_array(section[metric_key], f"axis.{metric_key}")

    traits = obj.get("traits", {})
    section = traits.get(entity_name)

    if section is None or metric_key not in section:
        return np.zeros(32, dtype=np.float64)

    return validate_32_float_array(section[metric_key], f"{entity_name}.{metric_key}")


def get_entity_names(feature_source: str, rows: List[Dict]) -> List[str]:
    available_traits = get_available_trait_names(rows)

    if feature_source == "axis_only":
        return [AXIS_NAME]
    if feature_source == "traits_only":
        return available_traits
    if feature_source == "traits_plus_axis":
        return [AXIS_NAME] + available_traits

    raise ValueError(f"Unknown feature_source={feature_source!r}")


def reduce_layers(
    arr_32: np.ndarray,
    layer_mode: str,
    layer: int,
    layer_start: int,
    layer_end: int,
) -> np.ndarray:
    if layer_mode == "single":
        if not (0 <= layer < 32):
            raise ValueError("--layer must be in [0, 31]")
        return np.asarray([arr_32[layer]], dtype=np.float64)

    if layer_mode == "mean_range":
        if not (0 <= layer_start <= layer_end < 32):
            raise ValueError("--layer_start/--layer_end must define a valid inclusive range within [0, 31]")
        return np.asarray([arr_32[layer_start: layer_end + 1].mean()], dtype=np.float64)

    if layer_mode == "all_layers_concat":
        return arr_32.copy()

    raise ValueError(f"Unknown layer_mode={layer_mode!r}")


def build_feature_matrix(
    rows: List[Dict],
    feature_source: str,
    metric_key: str,
    layer_mode: str,
    layer: int,
    layer_start: int,
    layer_end: int,
    positive_class: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict], List[str]]:
    entity_names = get_entity_names(feature_source, rows)

    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    metadata_rows: List[Dict] = []
    feature_names: List[str] = []

    for row in rows:
        row_features: List[np.ndarray] = []
        for entity_name in entity_names:
            vec_32 = extract_entity_vector(row, entity_name, metric_key)
            reduced = reduce_layers(
                arr_32=vec_32,
                layer_mode=layer_mode,
                layer=layer,
                layer_start=layer_start,
                layer_end=layer_end,
            )
            row_features.append(reduced)

        X_rows.append(np.concatenate(row_features, axis=0))
        y_rows.append(label_to_target(row["prompt_label"], positive_class))
        metadata_rows.append(
            {
                "prompt_idx": row.get("prompt_idx"),
                "prompt_label": row.get("prompt_label"),
                "attack_method": row.get("attack_method"),
            }
        )

    if layer_mode in {"single", "mean_range"}:
        feature_names = entity_names
    else:
        for entity_name in entity_names:
            for layer_idx in range(32):
                feature_names.append(f"{entity_name}_layer_{layer_idx}")

    X = np.stack(X_rows, axis=0).astype(np.float64)
    y = np.asarray(y_rows, dtype=np.int64)

    if not np.all(np.isfinite(X)):
        raise ValueError("Feature matrix contains non-finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Target vector contains non-finite values.")

    return X, y, feature_names, metadata_rows, entity_names


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
                    penalty="l2",
                    C=c_value,
                    solver="liblinear",
                    max_iter=max_iter,
                    random_state=random_state,
                    class_weight=lr_class_weight,
                ),
            ),
        ]
    )


def evaluate_cv(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    cv_folds: int,
    random_state: int,
) -> Dict:
    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    fold_results: List[Dict] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_score_all: List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]

        fold_result = {
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_score)),
            "average_precision": float(average_precision_score(y_test, y_score)),
        }
        fold_results.append(fold_result)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        y_score_all.extend(y_score.tolist())

    y_true = np.asarray(y_true_all, dtype=np.int64)
    y_pred = np.asarray(y_pred_all, dtype=np.int64)
    y_score = np.asarray(y_score_all, dtype=np.float64)

    aggregate = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "positive_rate_true": float(y_true.mean()),
        "positive_rate_pred": float(y_pred.mean()),
    }

    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    summary = {}
    for metric in metric_names:
        vals = np.asarray([fold[metric] for fold in fold_results], dtype=np.float64)
        summary[metric] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    return {
        "fold_results": fold_results,
        "aggregate_out_of_fold": aggregate,
        "cv_metric_summary": summary,
    }


def fit_full_model(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    feature_names: List[str],
) -> Dict:
    pipeline.fit(X, y)

    scaler: StandardScaler = pipeline.named_steps["scaler"]
    clf: LogisticRegression = pipeline.named_steps["clf"]

    coef = clf.coef_.ravel().astype(np.float64)
    intercept = float(clf.intercept_.ravel()[0])

    feature_rows = []
    for name, value in zip(feature_names, coef):
        odds_ratio = float(math.exp(value))
        feature_rows.append(
            {
                "feature": name,
                "coefficient": float(value),
                "abs_coefficient": float(abs(value)),
                "odds_ratio": odds_ratio,
                "direction": "pushes_toward_successful_jailbreak" if value > 0 else "pushes_away_from_successful_jailbreak",
            }
        )

    feature_rows_sorted = sorted(
        feature_rows,
        key=lambda x: abs(x["coefficient"]),
        reverse=True,
    )

    return {
        "intercept": intercept,
        "feature_coefficients_sorted": feature_rows_sorted,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_coefficients_csv(feature_rows: List[Dict], path: Path) -> None:
    header = [
        "feature",
        "coefficient",
        "abs_coefficient",
        "odds_ratio",
        "direction",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in feature_rows:
            vals = [
                row["feature"],
                f"{row['coefficient']:.10f}",
                f"{row['abs_coefficient']:.10f}",
                f"{row['odds_ratio']:.10f}",
                row["direction"],
            ]
            f.write(",".join(vals) + "\n")


def make_coefficient_plot(
    feature_rows: List[Dict],
    out_path: Path,
    title: str,
    top_k: int = 20,
) -> None:
    top_rows = feature_rows[:top_k]
    names = [row["feature"] for row in top_rows][::-1]
    values = [row["coefficient"] for row in top_rows][::-1]

    plt.figure(figsize=(10, max(6, 0.4 * len(top_rows))))
    plt.barh(names, values)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Logistic Regression Coefficient")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def make_per_layer_scan(
    rows: List[Dict],
    args: argparse.Namespace,
    metric_key: str,
    out_dir: Path,
) -> None:
    layer_metrics: List[Dict] = []

    for layer in range(32):
        X, y, feature_names, _, entity_names = build_feature_matrix(
            rows=rows,
            feature_source=args.feature_source,
            metric_key=metric_key,
            layer_mode="single",
            layer=layer,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            positive_class=args.positive_class,
        )
        pipeline = build_pipeline(
            c_value=args.c_value,
            max_iter=args.max_iter,
            random_state=args.random_state,
            class_weight=args.class_weight,
        )
        cv_res = evaluate_cv(
            X=X,
            y=y,
            pipeline=pipeline,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )
        layer_metrics.append(
            {
                "layer": layer,
                "n_features": int(X.shape[1]),
                "entity_names": entity_names,
                "roc_auc_mean": cv_res["cv_metric_summary"]["roc_auc"]["mean"],
                "roc_auc_std": cv_res["cv_metric_summary"]["roc_auc"]["std"],
                "f1_mean": cv_res["cv_metric_summary"]["f1"]["mean"],
                "f1_std": cv_res["cv_metric_summary"]["f1"]["std"],
                "average_precision_mean": cv_res["cv_metric_summary"]["average_precision"]["mean"],
                "average_precision_std": cv_res["cv_metric_summary"]["average_precision"]["std"],
            }
        )

    save_json({"per_layer_scan": layer_metrics}, out_dir / "per_layer_scan_metrics.json")

    layers = [row["layer"] for row in layer_metrics]
    roc_auc_means = [row["roc_auc_mean"] for row in layer_metrics]
    f1_means = [row["f1_mean"] for row in layer_metrics]
    ap_means = [row["average_precision_mean"] for row in layer_metrics]

    plt.figure(figsize=(11, 6))
    plt.plot(layers, roc_auc_means, label="ROC-AUC")
    plt.plot(layers, f1_means, label="F1")
    plt.plot(layers, ap_means, label="Average Precision")
    plt.xlabel("Layer")
    plt.ylabel("CV Metric")
    plt.title("Per-Layer Logistic Regression Performance")
    plt.xticks(np.arange(0, 32, 2))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "per_layer_scan_metrics.png", dpi=220, bbox_inches="tight")
    plt.close()


def write_human_readable_report(
    args: argparse.Namespace,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    entity_names: List[str],
    rows: List[Dict],
    cv_results: Dict,
    full_fit: Dict,
    out_path: Path,
) -> None:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    coeffs = full_fit["feature_coefficients_sorted"]

    top_positive = [row for row in coeffs if row["coefficient"] > 0][:10]
    top_negative = [row for row in coeffs if row["coefficient"] < 0][:10]

    trait_counts = get_trait_presence_counts(rows)
    attack_method_counts: Dict[str, int] = {}
    for row in rows:
        method = str(row.get("attack_method"))
        attack_method_counts[method] = attack_method_counts.get(method, 0) + 1

    lines: List[str] = []
    lines.append("LOGISTIC REGRESSION INTERPRETABILITY REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Configuration")
    lines.append("-" * 80)
    lines.append(f"summary_jsonl: {args.summary_jsonl}")
    lines.append(f"attack_method_filter: {args.attack_method_filter}")
    lines.append(f"feature_source: {args.feature_source}")
    lines.append(f"metric_type: {args.metric_type}")
    lines.append(f"layer_mode: {args.layer_mode}")
    if args.layer_mode == "single":
        lines.append(f"layer: {args.layer}")
    elif args.layer_mode == "mean_range":
        lines.append(f"layer_range: [{args.layer_start}, {args.layer_end}]")
    lines.append(f"positive_class: {args.positive_class}")
    lines.append(f"class_weight: {args.class_weight}")
    lines.append(f"cv_folds: {args.cv_folds}")
    lines.append("")
    lines.append("Dataset")
    lines.append("-" * 80)
    lines.append(f"n_examples: {len(y)}")
    lines.append(f"n_positive: {positives}")
    lines.append(f"n_negative: {negatives}")
    lines.append(f"positive_rate: {positives / len(y):.6f}")
    lines.append(f"n_features: {X.shape[1]}")
    lines.append(f"entity_names_used: {entity_names}")
    lines.append(f"feature_names: {feature_names}")
    lines.append("")
    lines.append("Attack Method Counts")
    lines.append("-" * 80)
    for method in sorted(attack_method_counts.keys()):
        lines.append(f"{method:28s} {attack_method_counts[method]}")
    lines.append("")
    lines.append("Trait Coverage in Input JSONL")
    lines.append("-" * 80)
    for trait_name in sorted(trait_counts.keys()):
        lines.append(f"{trait_name:28s} {trait_counts[trait_name]}/{len(rows)} rows")
    lines.append("")
    lines.append("Cross-Validation Summary")
    lines.append("-" * 80)
    for metric_name, metric_dict in cv_results["cv_metric_summary"].items():
        lines.append(
            f"{metric_name:18s} mean={metric_dict['mean']:.6f}  std={metric_dict['std']:.6f}  "
            f"min={metric_dict['min']:.6f}  max={metric_dict['max']:.6f}"
        )
    lines.append("")
    lines.append("Out-of-Fold Aggregate")
    lines.append("-" * 80)
    agg = cv_results["aggregate_out_of_fold"]
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
        lines.append(f"{key:18s} {agg[key]:.6f}")
    lines.append(f"confusion_matrix: {agg['confusion_matrix']}")
    lines.append("")
    lines.append("Top Features Pushing Toward Successful Jailbreak")
    lines.append("-" * 80)
    if top_positive:
        for row in top_positive:
            lines.append(
                f"{row['feature']:28s} coef={row['coefficient']:+.6f}  odds_ratio={row['odds_ratio']:.6f}"
            )
    else:
        lines.append("No positive coefficients found.")
    lines.append("")
    lines.append("Top Features Pushing Away From Successful Jailbreak")
    lines.append("-" * 80)
    if top_negative:
        for row in top_negative:
            lines.append(
                f"{row['feature']:28s} coef={row['coefficient']:+.6f}  odds_ratio={row['odds_ratio']:.6f}"
            )
    else:
        lines.append("No negative coefficients found.")
    lines.append("")
    lines.append("Interpretation Guide")
    lines.append("-" * 80)
    lines.append("Positive coefficient  -> higher value of this feature increases successful jailbreak log-odds.")
    lines.append("Negative coefficient  -> higher value of this feature decreases successful jailbreak log-odds.")
    lines.append("Because features are standardized, coefficient magnitudes are directly comparable.")
    lines.append("Odds ratio > 1        -> pushes toward successful jailbreak.")
    lines.append("Odds ratio < 1        -> pushes away from successful jailbreak.")
    lines.append("Missing trait entries in a row are encoded as zeros in this script.")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_key = metric_key_from_metric_type(args.metric_type)
    rows = load_rows(args.summary_jsonl, args.attack_method_filter)

    if args.debug_print_schema:
        debug_print_schema(rows)

    X, y, feature_names, metadata_rows, entity_names = build_feature_matrix(
        rows=rows,
        feature_source=args.feature_source,
        metric_key=metric_key,
        layer_mode=args.layer_mode,
        layer=args.layer,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        positive_class=args.positive_class,
    )

    pipeline = build_pipeline(
        c_value=args.c_value,
        max_iter=args.max_iter,
        random_state=args.random_state,
        class_weight=args.class_weight,
    )

    cv_results = evaluate_cv(
        X=X,
        y=y,
        pipeline=pipeline,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )

    full_fit = fit_full_model(
        X=X,
        y=y,
        pipeline=pipeline,
        feature_names=feature_names,
    )

    trait_presence_counts = get_trait_presence_counts(rows)

    config = {
        "summary_jsonl": args.summary_jsonl,
        "attack_method_filter": args.attack_method_filter,
        "feature_source": args.feature_source,
        "metric_type": args.metric_type,
        "metric_key": metric_key,
        "layer_mode": args.layer_mode,
        "layer": args.layer,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "positive_class": args.positive_class,
        "cv_folds": args.cv_folds,
        "random_state": args.random_state,
        "max_iter": args.max_iter,
        "c_value": args.c_value,
        "class_weight": args.class_weight,
        "n_examples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "entity_names_used": entity_names,
        "feature_names": feature_names,
        "trait_presence_counts": trait_presence_counts,
        "traits_present_in_all_rows": get_trait_names_present_in_all_rows(rows),
        "traits_present_in_any_row": get_available_trait_names(rows),
    }

    save_json(config, out_dir / "run_config.json")
    save_json(cv_results, out_dir / "cv_results.json")
    save_json(full_fit, out_dir / "full_model_coefficients.json")
    save_coefficients_csv(
        full_fit["feature_coefficients_sorted"],
        out_dir / "full_model_coefficients.csv",
    )

    make_coefficient_plot(
        feature_rows=full_fit["feature_coefficients_sorted"],
        out_path=out_dir / "top_coefficients.png",
        title="Top Logistic Regression Coefficients by Absolute Magnitude",
        top_k=min(20, len(full_fit["feature_coefficients_sorted"])),
    )

    write_human_readable_report(
        args=args,
        X=X,
        y=y,
        feature_names=feature_names,
        entity_names=entity_names,
        rows=rows,
        cv_results=cv_results,
        full_fit=full_fit,
        out_path=out_dir / "interpretability_report.txt",
    )

    if args.save_per_layer_scan:
        make_per_layer_scan(
            rows=rows,
            args=args,
            metric_key=metric_key,
            out_dir=out_dir,
        )

    print("Done.")
    print(f"Outputs written to: {out_dir}")
    print(f"n_examples={len(y)}  n_features={X.shape[1]}  n_positive={int(y.sum())}  n_negative={int(len(y)-y.sum())}")
    print(f"entity_names_used={entity_names}")
    print("Best place to start reading:")
    print(f"  1) {out_dir / 'interpretability_report.txt'}")
    print(f"  2) {out_dir / 'top_coefficients.png'}")
    print(f"  3) {out_dir / 'full_model_coefficients.csv'}")


if __name__ == "__main__":
    main()