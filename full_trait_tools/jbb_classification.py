# interview note: updated logistic regression for the new jailbreak projection JSONL; supports new pre_response/answer_mean/delta metrics, zero-fills missing traits, and can do both binary jailbreak-success classification and multiclass attack-method classification including benign

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
    "unjudged_jailbreak",
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
        "--task_mode",
        type=str,
        default="binary_success_vs_rest",
        choices=[
            "binary_success_vs_rest",
            "multiclass_attack_method",
        ],
        help="Binary successful-jailbreak-vs-rest or multiclass attack-method classification (including benign).",
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
        default="delta_proj",
        choices=[
            "pre_response_proj",
            "answer_mean_proj",
            "delta_proj",
            "pre_response_cos",
            "answer_mean_cos",
            "delta_cos",
            # backward-compatible old names
            "last_prompt_proj",
            "mean_response_proj",
            "last_prompt_cos",
            "mean_response_cos",
        ],
        help="Which projection/cosine family to use.",
    )
    parser.add_argument(
        "--layer_mode",
        type=str,
        default="mean_range",
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
        help="Positive target class for binary mode.",
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
        # new format
        "pre_response_proj": "pre_response_proj_all_layers",
        "answer_mean_proj": "answer_mean_proj_all_layers",
        "delta_proj": "delta_proj_all_layers",
        "pre_response_cos": "pre_response_cos_all_layers",
        "answer_mean_cos": "answer_mean_cos_all_layers",
        "delta_cos": "delta_cos_all_layers",
        # old compatibility
        "last_prompt_proj": "last_prompt_proj_all_layers",
        "mean_response_proj": "mean_response_proj_all_layers",
        "last_prompt_cos": "last_prompt_cos_all_layers",
        "mean_response_cos": "mean_response_cos_all_layers",
    }
    return mapping[metric_type]


def label_to_binary_target(label: str, positive_class: str) -> int:
    return 1 if label == positive_class else 0


def attack_method_to_multiclass_target(row: Dict) -> str:
    method = row.get("attack_method")
    if method is None:
        raise ValueError("Missing attack_method for multiclass task.")
    return str(method)


def load_rows(summary_jsonl: str) -> List[Dict]:
    rows: List[Dict] = []
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
            rows.append(obj)
    if not rows:
        raise ValueError("No rows found in summary_jsonl.")
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

    # Robust fallback: if a trait is missing in a row, return zeros instead of crashing.
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
        return np.asarray([arr_32[layer_start : layer_end + 1].mean()], dtype=np.float64)

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
    task_mode: str,
    positive_class: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict], List[str], Dict[int, str]]:
    entity_names = get_entity_names(feature_source, rows)

    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    metadata_rows: List[Dict] = []
    feature_names: List[str] = []

    if task_mode == "multiclass_attack_method":
        class_names = sorted({attack_method_to_multiclass_target(row) for row in rows})
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        idx_to_class = {i: name for name, i in class_to_idx.items()}
    else:
        idx_to_class = {0: "rest", 1: positive_class}

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

        if task_mode == "binary_success_vs_rest":
            y_rows.append(label_to_binary_target(row["prompt_label"], positive_class))
        elif task_mode == "multiclass_attack_method":
            y_rows.append(class_to_idx[attack_method_to_multiclass_target(row)])
        else:
            raise ValueError(f"Unknown task_mode={task_mode!r}")

        metadata_rows.append(
            {
                "pt_file": row.get("pt_file"),
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

    return X, y, feature_names, metadata_rows, entity_names, idx_to_class


def build_pipeline(
    c_value: float,
    max_iter: int,
    random_state: int,
    class_weight: str,
    task_mode: str,
) -> Pipeline:
    lr_class_weight = None if class_weight == "none" else "balanced"
    multi_class = "ovr" if task_mode == "multiclass_attack_method" else "auto"

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
                    multi_class=multi_class,
                ),
            ),
        ]
    )


def evaluate_cv_binary(
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


def evaluate_cv_multiclass(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    cv_folds: int,
    random_state: int,
    idx_to_class: Dict[int, str],
) -> Dict:
    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    fold_results: List[Dict] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        fold_result = {
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }
        fold_results.append(fold_result)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

    y_true = np.asarray(y_true_all, dtype=np.int64)
    y_pred = np.asarray(y_pred_all, dtype=np.int64)

    labels = sorted(idx_to_class.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    aggregate = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [idx_to_class[i] for i in labels],
    }

    metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
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


def fit_full_model_binary(
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


def fit_full_model_multiclass(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    feature_names: List[str],
    idx_to_class: Dict[int, str],
) -> Dict:
    pipeline.fit(X, y)

    scaler: StandardScaler = pipeline.named_steps["scaler"]
    clf: LogisticRegression = pipeline.named_steps["clf"]

    coef = clf.coef_.astype(np.float64)
    intercept = clf.intercept_.astype(np.float64)

    class_rows = {}
    for class_idx, class_name in idx_to_class.items():
        feature_rows = []
        class_coef = coef[class_idx]
        for name, value in zip(feature_names, class_coef):
            feature_rows.append(
                {
                    "feature": name,
                    "coefficient": float(value),
                    "abs_coefficient": float(abs(value)),
                    "direction": f"pushes_toward_{class_name}" if value > 0 else f"pushes_away_from_{class_name}",
                }
            )
        class_rows[class_name] = sorted(feature_rows, key=lambda x: x["abs_coefficient"], reverse=True)

    return {
        "intercepts": {idx_to_class[i]: float(intercept[i]) for i in range(len(intercept))},
        "feature_coefficients_by_class": class_rows,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_coefficients_csv_binary(feature_rows: List[Dict], path: Path) -> None:
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


def save_coefficients_csv_multiclass(feature_rows_by_class: Dict[str, List[Dict]], path: Path) -> None:
    header = [
        "class_name",
        "feature",
        "coefficient",
        "abs_coefficient",
        "direction",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for class_name, rows in feature_rows_by_class.items():
            for row in rows:
                vals = [
                    class_name,
                    row["feature"],
                    f"{row['coefficient']:.10f}",
                    f"{row['abs_coefficient']:.10f}",
                    row["direction"],
                ]
                f.write(",".join(vals) + "\n")


def make_coefficient_plot_binary(
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


def make_coefficient_plot_multiclass(
    feature_rows_by_class: Dict[str, List[Dict]],
    out_dir: Path,
    top_k: int = 15,
) -> None:
    for class_name, rows in feature_rows_by_class.items():
        top_rows = rows[:top_k]
        names = [row["feature"] for row in top_rows][::-1]
        values = [row["coefficient"] for row in top_rows][::-1]

        plt.figure(figsize=(10, max(6, 0.4 * len(top_rows))))
        plt.barh(names, values)
        plt.axvline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("One-vs-Rest Logistic Regression Coefficient")
        plt.title(f"Top Coefficients for Class: {class_name}")
        plt.tight_layout()
        safe_name = class_name.replace("/", "_").replace(" ", "_")
        plt.savefig(out_dir / f"top_coefficients_{safe_name}.png", dpi=220, bbox_inches="tight")
        plt.close()


def make_per_layer_scan(
    rows: List[Dict],
    args: argparse.Namespace,
    metric_key: str,
    out_dir: Path,
) -> None:
    layer_metrics: List[Dict] = []

    for layer in range(32):
        X, y, feature_names, _, entity_names, idx_to_class = build_feature_matrix(
            rows=rows,
            feature_source=args.feature_source,
            metric_key=metric_key,
            layer_mode="single",
            layer=layer,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            task_mode=args.task_mode,
            positive_class=args.positive_class,
        )
        pipeline = build_pipeline(
            c_value=args.c_value,
            max_iter=args.max_iter,
            random_state=args.random_state,
            class_weight=args.class_weight,
            task_mode=args.task_mode,
        )

        if args.task_mode == "binary_success_vs_rest":
            cv_res = evaluate_cv_binary(
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
                    "primary_metric_mean": cv_res["cv_metric_summary"]["roc_auc"]["mean"],
                    "primary_metric_std": cv_res["cv_metric_summary"]["roc_auc"]["std"],
                    "secondary_metric_mean": cv_res["cv_metric_summary"]["f1"]["mean"],
                    "secondary_metric_std": cv_res["cv_metric_summary"]["f1"]["std"],
                    "primary_metric_name": "roc_auc",
                    "secondary_metric_name": "f1",
                }
            )
        else:
            cv_res = evaluate_cv_multiclass(
                X=X,
                y=y,
                pipeline=pipeline,
                cv_folds=args.cv_folds,
                random_state=args.random_state,
                idx_to_class=idx_to_class,
            )
            layer_metrics.append(
                {
                    "layer": layer,
                    "n_features": int(X.shape[1]),
                    "entity_names": entity_names,
                    "primary_metric_mean": cv_res["cv_metric_summary"]["macro_f1"]["mean"],
                    "primary_metric_std": cv_res["cv_metric_summary"]["macro_f1"]["std"],
                    "secondary_metric_mean": cv_res["cv_metric_summary"]["accuracy"]["mean"],
                    "secondary_metric_std": cv_res["cv_metric_summary"]["accuracy"]["std"],
                    "primary_metric_name": "macro_f1",
                    "secondary_metric_name": "accuracy",
                }
            )

    save_json({"per_layer_scan": layer_metrics}, out_dir / "per_layer_scan_metrics.json")

    layers = [row["layer"] for row in layer_metrics]
    primary_means = [row["primary_metric_mean"] for row in layer_metrics]
    secondary_means = [row["secondary_metric_mean"] for row in layer_metrics]
    primary_name = layer_metrics[0]["primary_metric_name"]
    secondary_name = layer_metrics[0]["secondary_metric_name"]

    plt.figure(figsize=(11, 6))
    plt.plot(layers, primary_means, label=primary_name)
    plt.plot(layers, secondary_means, label=secondary_name)
    plt.xlabel("Layer")
    plt.ylabel("CV Metric")
    plt.title("Per-Layer Logistic Regression Performance")
    plt.xticks(np.arange(0, 32, 2))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "per_layer_scan_metrics.png", dpi=220, bbox_inches="tight")
    plt.close()


def write_human_readable_report_binary(
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

    lines: List[str] = []
    lines.append("BINARY LOGISTIC REGRESSION INTERPRETABILITY REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Configuration")
    lines.append("-" * 80)
    lines.append(f"summary_jsonl: {args.summary_jsonl}")
    lines.append(f"task_mode: {args.task_mode}")
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
    lines.append("Missing trait entries in a row are encoded as zeros in this script.")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_human_readable_report_multiclass(
    args: argparse.Namespace,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    entity_names: List[str],
    rows: List[Dict],
    cv_results: Dict,
    full_fit: Dict,
    idx_to_class: Dict[int, str],
    out_path: Path,
) -> None:
    trait_counts = get_trait_presence_counts(rows)
    unique, counts = np.unique(y, return_counts=True)
    class_counts = {idx_to_class[int(k)]: int(v) for k, v in zip(unique, counts)}

    lines: List[str] = []
    lines.append("MULTICLASS LOGISTIC REGRESSION INTERPRETABILITY REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Configuration")
    lines.append("-" * 80)
    lines.append(f"summary_jsonl: {args.summary_jsonl}")
    lines.append(f"task_mode: {args.task_mode}")
    lines.append(f"feature_source: {args.feature_source}")
    lines.append(f"metric_type: {args.metric_type}")
    lines.append(f"layer_mode: {args.layer_mode}")
    if args.layer_mode == "single":
        lines.append(f"layer: {args.layer}")
    elif args.layer_mode == "mean_range":
        lines.append(f"layer_range: [{args.layer_start}, {args.layer_end}]")
    lines.append(f"class_weight: {args.class_weight}")
    lines.append(f"cv_folds: {args.cv_folds}")
    lines.append("")
    lines.append("Dataset")
    lines.append("-" * 80)
    lines.append(f"n_examples: {len(y)}")
    lines.append(f"n_features: {X.shape[1]}")
    lines.append(f"entity_names_used: {entity_names}")
    lines.append(f"classes: {list(class_counts.keys())}")
    lines.append(f"class_counts: {class_counts}")
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
    for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]:
        lines.append(f"{key:18s} {agg[key]:.6f}")
    lines.append(f"confusion_matrix_labels: {agg['confusion_matrix_labels']}")
    lines.append(f"confusion_matrix: {agg['confusion_matrix']}")
    lines.append("")
    lines.append("Top Features By Class")
    lines.append("-" * 80)
    for class_name, rows_for_class in full_fit["feature_coefficients_by_class"].items():
        lines.append(f"[{class_name}]")
        for row in rows_for_class[:10]:
            lines.append(f"  {row['feature']:28s} coef={row['coefficient']:+.6f}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_key = metric_key_from_metric_type(args.metric_type)
    rows = load_rows(args.summary_jsonl)

    if args.debug_print_schema:
        debug_print_schema(rows)

    X, y, feature_names, metadata_rows, entity_names, idx_to_class = build_feature_matrix(
        rows=rows,
        feature_source=args.feature_source,
        metric_key=metric_key,
        layer_mode=args.layer_mode,
        layer=args.layer,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        task_mode=args.task_mode,
        positive_class=args.positive_class,
    )

    pipeline = build_pipeline(
        c_value=args.c_value,
        max_iter=args.max_iter,
        random_state=args.random_state,
        class_weight=args.class_weight,
        task_mode=args.task_mode,
    )

    if args.task_mode == "binary_success_vs_rest":
        cv_results = evaluate_cv_binary(
            X=X,
            y=y,
            pipeline=pipeline,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )
        full_fit = fit_full_model_binary(
            X=X,
            y=y,
            pipeline=pipeline,
            feature_names=feature_names,
        )
    else:
        cv_results = evaluate_cv_multiclass(
            X=X,
            y=y,
            pipeline=pipeline,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            idx_to_class=idx_to_class,
        )
        full_fit = fit_full_model_multiclass(
            X=X,
            y=y,
            pipeline=pipeline,
            feature_names=feature_names,
            idx_to_class=idx_to_class,
        )

    trait_presence_counts = get_trait_presence_counts(rows)

    config = {
        "summary_jsonl": args.summary_jsonl,
        "task_mode": args.task_mode,
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
        "entity_names_used": entity_names,
        "feature_names": feature_names,
        "trait_presence_counts": trait_presence_counts,
        "traits_present_in_all_rows": get_trait_names_present_in_all_rows(rows),
        "traits_present_in_any_row": get_available_trait_names(rows),
        "class_index_mapping": idx_to_class,
    }

    save_json(config, out_dir / "run_config.json")
    save_json(cv_results, out_dir / "cv_results.json")
    save_json(full_fit, out_dir / "full_model_coefficients.json")

    if args.task_mode == "binary_success_vs_rest":
        save_coefficients_csv_binary(
            full_fit["feature_coefficients_sorted"],
            out_dir / "full_model_coefficients.csv",
        )
        make_coefficient_plot_binary(
            feature_rows=full_fit["feature_coefficients_sorted"],
            out_path=out_dir / "top_coefficients.png",
            title="Top Logistic Regression Coefficients by Absolute Magnitude",
            top_k=min(20, len(full_fit["feature_coefficients_sorted"])),
        )
        write_human_readable_report_binary(
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
    else:
        save_coefficients_csv_multiclass(
            full_fit["feature_coefficients_by_class"],
            out_dir / "full_model_coefficients.csv",
        )
        make_coefficient_plot_multiclass(
            feature_rows_by_class=full_fit["feature_coefficients_by_class"],
            out_dir=out_dir,
            top_k=15,
        )
        write_human_readable_report_multiclass(
            args=args,
            X=X,
            y=y,
            feature_names=feature_names,
            entity_names=entity_names,
            rows=rows,
            cv_results=cv_results,
            full_fit=full_fit,
            idx_to_class=idx_to_class,
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
    print(f"n_examples={len(y)}  n_features={X.shape[1]}")
    print(f"entity_names_used={entity_names}")
    print("Best place to start reading:")
    print(f"  1) {out_dir / 'interpretability_report.txt'}")
    print(f"  2) {out_dir / 'cv_results.json'}")
    print(f"  3) {out_dir / 'full_model_coefficients.csv'}")


if __name__ == "__main__":
    main()