# interview note: this baseline tests how predictive the attack method alone is by one-hot encoding attack_type and fitting a logistic regression to predict jailbroken vs not

#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


VALID_POSITIVE_LABELS = {"successful_jailbreak"}
VALID_NEGATIVE_LABELS = {"benign", "unsuccessful_jailbreak"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Logistic regression on one-hot attack type to predict whether a prompt was jailbroken."
    )
    parser.add_argument(
        "--summary_jsonl",
        type=str,
        required=True,
        help="Path to projection_summary.jsonl or any JSONL with attack_method and prompt_label / jailbroken_our_judge fields.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of stratified CV folds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save metrics and coefficients.",
    )
    parser.add_argument(
        "--use_prompt_label",
        action="store_true",
        help="Use prompt_label for labels. Otherwise infer from jailbroken_our_judge + is_jailbreak.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_num}: {e}") from e
    return rows


def infer_binary_label(rec: dict, use_prompt_label: bool) -> int | None:
    if use_prompt_label:
        label = rec.get("prompt_label")
        if label in VALID_POSITIVE_LABELS:
            return 1
        if label in VALID_NEGATIVE_LABELS:
            return 0
        return None

    is_jailbreak = rec.get("is_jailbreak")
    judge = rec.get("jailbroken_our_judge")

    if is_jailbreak is False:
        return 0
    if judge is True:
        return 1
    if judge is False:
        return 0
    return None


def build_xy(rows: list[dict], use_prompt_label: bool):
    x_attack = []
    y = []
    kept_rows = []

    for rec in rows:
        attack_method = rec.get("attack_method")
        label = infer_binary_label(rec, use_prompt_label=use_prompt_label)

        if attack_method is None or label is None:
            continue

        x_attack.append([str(attack_method)])
        y.append(int(label))
        kept_rows.append(rec)

    if not x_attack:
        raise ValueError("No usable rows found after filtering.")

    x_attack = np.array(x_attack, dtype=object)
    y = np.array(y, dtype=np.int64)
    return x_attack, y, kept_rows


def safe_metric(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    summary_path = Path(args.summary_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(summary_path)
    x_attack, y, kept_rows = build_xy(rows, use_prompt_label=args.use_prompt_label)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "attack_type",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [0],
            )
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=args.seed,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    cv = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    y_pred = cross_val_predict(
        pipeline,
        x_attack,
        y,
        cv=cv,
        method="predict",
    )
    y_prob = cross_val_predict(
        pipeline,
        x_attack,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    metrics = {
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "attack_method_counts": {},
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": safe_metric(roc_auc_score, y, y_prob),
        "average_precision": safe_metric(average_precision_score, y, y_prob),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(y, y_pred, zero_division=0),
    }

    for attack in x_attack[:, 0]:
        metrics["attack_method_counts"][attack] = metrics["attack_method_counts"].get(attack, 0) + 1

    pipeline.fit(x_attack, y)

    encoder = pipeline.named_steps["preprocess"].named_transformers_["attack_type"]
    feature_names = encoder.get_feature_names_out(["attack_type"])
    coefs = pipeline.named_steps["clf"].coef_[0]
    intercept = float(pipeline.named_steps["clf"].intercept_[0])

    coef_rows = []
    for name, coef in sorted(zip(feature_names, coefs), key=lambda t: abs(t[1]), reverse=True):
        coef_rows.append(
            {
                "feature": str(name),
                "coefficient": float(coef),
                "odds_multiplier": float(np.exp(coef)),
            }
        )

    predictions_rows = []
    for rec, pred, prob in zip(kept_rows, y_pred, y_prob):
        predictions_rows.append(
            {
                "pt_file": rec.get("pt_file"),
                "prompt_idx": rec.get("prompt_idx"),
                "attack_method": rec.get("attack_method"),
                "prompt_label": rec.get("prompt_label"),
                "is_jailbreak": rec.get("is_jailbreak"),
                "jailbroken_our_judge": rec.get("jailbroken_our_judge"),
                "predicted_jailbroken": int(pred),
                "predicted_prob_jailbroken": float(prob),
            }
        )

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "coefficients.json").write_text(
        json.dumps(
            {
                "intercept": intercept,
                "coefficients": coef_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions_rows:
            f.write(json.dumps(row) + "\n")

    print("=" * 80)
    print("Attack-type-only logistic regression")
    print("=" * 80)
    print(f"Samples:        {metrics['n_samples']}")
    print(f"Positive:       {metrics['n_positive']}")
    print(f"Negative:       {metrics['n_negative']}")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1:             {metrics['f1']:.4f}")
    print(f"ROC-AUC:        {metrics['roc_auc']}")
    print(f"Avg Precision:  {metrics['average_precision']}")
    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nTop coefficients:")
    for row in coef_rows[:10]:
        print(
            f"  {row['feature']:<35} "
            f"coef={row['coefficient']:+.4f} "
            f"odds_x={row['odds_multiplier']:.4f}"
        )
    print(f"\nSaved metrics to:      {out_dir / 'metrics.json'}")
    print(f"Saved coefficients to: {out_dir / 'coefficients.json'}")
    print(f"Saved predictions to:  {out_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()