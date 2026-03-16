#!/usr/bin/env python3

# interview note: simple baseline — predict jailbreak success using only attack_method one-hot features

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


VALID_POSITIVE_LABELS = {"successful_jailbreak"}
VALID_NEGATIVE_LABELS = {"benign", "unsuccessful_jailbreak"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary_jsonl", required=True)
    p.add_argument("--n_splits", type=int, default=5)
    return p.parse_args()


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def infer_label(rec):
    label = rec.get("prompt_label")

    if label in VALID_POSITIVE_LABELS:
        return 1
    if label in VALID_NEGATIVE_LABELS:
        return 0

    return None


def build_xy(rows):

    X = []
    y = []

    for r in rows:
        attack = r.get("attack_method")
        label = infer_label(r)

        if attack is None or label is None:
            continue

        X.append([attack])
        y.append(label)

    return np.array(X), np.array(y)


def main():

    args = parse_args()

    rows = load_jsonl(args.summary_jsonl)

    X, y = build_xy(rows)

    print("Samples:", len(y))
    print("Positive:", y.sum())
    print("Negative:", (y == 0).sum())

    preprocess = ColumnTransformer(
        [
            (
                "attack",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [0],
            )
        ]
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    pipe = Pipeline([
        ("prep", preprocess),
        ("clf", model)
    ])

    cv = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=42
    )

    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:,1]

    print("\nMetrics")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred))
    print("F1:", f1_score(y, y_pred))
    print("ROC-AUC:", roc_auc_score(y, y_prob))

    print("\nConfusion Matrix")
    print(confusion_matrix(y, y_pred))

    pipe.fit(X, y)

    encoder = pipe.named_steps["prep"].named_transformers_["attack"]
    feature_names = encoder.get_feature_names_out(["attack"])

    coefs = pipe.named_steps["clf"].coef_[0]

    print("\nAttack Method Coefficients")

    pairs = list(zip(feature_names, coefs))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, coef in pairs:
        odds = np.exp(coef)
        print(f"{name:35s} coef={coef:+.3f} odds_mult={odds:.3f}")


if __name__ == "__main__":
    main()