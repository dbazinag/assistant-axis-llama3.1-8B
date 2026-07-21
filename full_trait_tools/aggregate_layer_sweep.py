#!/usr/bin/env python3
"""
aggregate_layer_sweep.py

Reads full_trait_output/all_traits_sweep_v2_layer{L}/results.json for each layer
produced by run_all_layers_sweep.sh and builds one summary table: per layer, the
best-by-human-test-AUC model and the best-by-average-transfer-AUC model, with
their val/human_test/transfer AUCs.
"""

import argparse
import csv
import json
from pathlib import Path


def load_results(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def families_in(summary: dict) -> list[str]:
    any_model = next(iter(summary.values()))
    return [k for k in any_model.keys() if k not in ("val_auc", "human_test")]


def row_for(name: str, summary: dict, families: list[str]) -> dict:
    m = summary[name]
    row = {
        "model": name,
        "val_auc": m["val_auc"]["mean"],
        "human_test_auc": m["human_test"]["auc"]["mean"],
    }
    for f in families:
        row[f] = m.get(f, {}).get("auc", {}).get("mean")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_glob", default="full_trait_output/all_traits_sweep_v2_layer{layer}/results.json")
    parser.add_argument("--min_layer", type=int, default=0)
    parser.add_argument("--max_layer", type=int, default=31)
    parser.add_argument("--output_csv", default="full_trait_output/all_layers_sweep_summary.csv")
    args = parser.parse_args()

    all_rows = []
    families_seen: list[str] = []

    for layer in range(args.min_layer, args.max_layer + 1):
        path = Path(args.sweep_glob.format(layer=layer))
        data = load_results(path)
        if data is None:
            print(f"layer {layer}: missing ({path})")
            continue

        summary = data["summary"]
        families = families_in(summary)
        for f in families:
            if f not in families_seen:
                families_seen.append(f)

        def avg_transfer(name: str) -> float:
            vals = [summary[name].get(f, {}).get("auc", {}).get("mean") for f in families]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else float("-inf")

        best_by_test = max(summary, key=lambda n: summary[n]["human_test"]["auc"]["mean"])
        best_by_transfer = max(summary, key=avg_transfer)

        row = {"layer": layer}
        for k, v in row_for(best_by_test, summary, families).items():
            row[f"best_test_{k}"] = v
        for k, v in row_for(best_by_transfer, summary, families).items():
            row[f"best_transfer_{k}"] = v
        all_rows.append(row)

    if not all_rows:
        print("No layer results found.")
        return

    fieldnames = ["layer"]
    fieldnames += [f"best_test_{k}" for k in ("model", "val_auc", "human_test_auc", *families_seen)]
    fieldnames += [f"best_transfer_{k}" for k in ("model", "val_auc", "human_test_auc", *families_seen)]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Wrote {len(all_rows)} layer rows to {out_path}")
    print()
    header = f"{'layer':>5} {'test_auc':>9} {'avg_xfer':>9} " + " ".join(f"{f:>9}" for f in families_seen)
    print(header)
    for row in all_rows:
        xfer_vals = [row.get(f"best_transfer_{f}") for f in families_seen]
        xfer_vals = [v for v in xfer_vals if v is not None]
        avg_xfer = sum(xfer_vals) / len(xfer_vals) if xfer_vals else float("nan")
        fam_str = " ".join(f"{row.get(f'best_transfer_{f}', float('nan')):>9.3f}" for f in families_seen)
        print(f"{row['layer']:>5} {row['best_test_human_test_auc']:>9.3f} {avg_xfer:>9.3f} {fam_str}")


if __name__ == "__main__":
    main()
