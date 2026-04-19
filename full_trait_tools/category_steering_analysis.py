#!/usr/bin/env python3
"""
category_steering_analysis.py

Reads steering_robustness_v2_results.json and classified_responses.jsonl,
joins on pair_id to get semantic categories, then reports jailbreak rates
broken down by semantic category for the most important conditions.

Shows top 5 most steerable and top 5 least steerable categories
for w_protective and PC1_protective at alpha=0.25.

Usage:
  uv run full_trait_tools/category_steering_analysis.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_rates(entries: List[dict]) -> dict:
    n_jb    = sum(1 for e in entries if e["label"] == "jailbroken")
    n_ref   = sum(1 for e in entries if e["label"] == "refused")
    n_deg   = sum(1 for e in entries if e["label"] == "degenerate")
    n_valid = n_jb + n_ref
    return {
        "n_jailbroken": n_jb, "n_refused": n_ref,
        "n_degenerate": n_deg, "n_valid": n_valid,
        "jb_rate": n_jb / n_valid if n_valid > 0 else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str,
        default="full_trait_output/steering_robustness_v2/steering_robustness_v2_results.json")
    parser.add_argument("--classified_path", type=str,
        default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()

    # Load results
    data = json.load(open(args.results_path))
    alpha_str = str(args.alpha)

    # Load classified responses to get semantic categories
    classified = load_jsonl(Path(args.classified_path))
    pair_to_category = {r["pair_id"]: r.get("semantic_category", "unknown")
                        for r in classified}

    # Conditions to analyze
    conditions_of_interest = [
        ("baseline",              data["protective"]),
        ("w_protective (-w)",     data["protective"]),
        ("PC1_protective (+PC1)", data["protective"]),
        ("cautious (+)",          data["protective"]),
    ]

    sep = "=" * 80

    for cond_name, results in conditions_of_interest:
        entries = results.get(cond_name, {}).get(alpha_str, [])
        if not entries:
            print(f"\nNo data for {cond_name} at alpha={args.alpha}")
            continue

        # Group by category
        by_cat: Dict[str, List[dict]] = defaultdict(list)
        for e in entries:
            pid = e.get("pair_id")
            cat = pair_to_category.get(pid, "unknown")
            by_cat[cat].append(e)

        # Compute rates per category
        cat_rates = []
        for cat, cat_entries in by_cat.items():
            r = compute_rates(cat_entries)
            if r["n_valid"] >= 3:  # only report categories with enough samples
                cat_rates.append((cat, r))

        if not cat_rates:
            print(f"\nNot enough data per category for {cond_name}")
            continue

        cat_rates.sort(key=lambda x: x[1]["jb_rate"])

        print(f"\n{sep}")
        print(f"  CONDITION: {cond_name}  (alpha={args.alpha})")
        print(sep)

        # Overall rate
        overall = compute_rates(entries)
        print(f"\n  Overall: {overall['n_jailbroken']}/{overall['n_valid']} = "
              f"{100*overall['jb_rate']:.0f}% jailbroken")

        # Most steerable (lowest JB rate = steering worked best)
        print(f"\n  Most steerable categories (lowest JB rate):")
        print(f"  {'Category':35s}  {'JB rate':>10}  {'n valid':>8}  {'n deg':>6}")
        print("  " + "─" * 65)
        for cat, r in cat_rates[:args.top_n]:
            print(f"  {cat:35s}  "
                  f"{r['n_jailbroken']}/{r['n_valid']}={100*r['jb_rate']:.0f}%  "
                  f"{r['n_valid']:>8}  {r['n_degenerate']:>6}")

        # Least steerable (highest JB rate = steering barely worked)
        print(f"\n  Least steerable categories (highest JB rate):")
        print(f"  {'Category':35s}  {'JB rate':>10}  {'n valid':>8}  {'n deg':>6}")
        print("  " + "─" * 65)
        for cat, r in reversed(cat_rates[-args.top_n:]):
            print(f"  {cat:35s}  "
                  f"{r['n_jailbroken']}/{r['n_valid']}={100*r['jb_rate']:.0f}%  "
                  f"{r['n_valid']:>8}  {r['n_degenerate']:>6}")

    # ── Comparison table: baseline vs w_protective per category ───────────────
    print(f"\n\n{sep}")
    print(f"  CATEGORY COMPARISON: baseline vs w_protective at alpha={args.alpha}")
    print(sep)

    baseline_entries = data["protective"].get("baseline", {}).get(alpha_str, [])
    w_entries        = data["protective"].get("w_protective (-w)", {}).get(alpha_str, [])

    if baseline_entries and w_entries:
        # Build per-category dicts
        bl_by_cat: Dict[str, List[dict]] = defaultdict(list)
        for e in baseline_entries:
            cat = pair_to_category.get(e.get("pair_id"), "unknown")
            bl_by_cat[cat].append(e)

        w_by_cat: Dict[str, List[dict]] = defaultdict(list)
        for e in w_entries:
            cat = pair_to_category.get(e.get("pair_id"), "unknown")
            w_by_cat[cat].append(e)

        all_cats = sorted(set(bl_by_cat.keys()) | set(w_by_cat.keys()))

        print(f"\n  {'Category':35s}  {'Baseline':>10}  {'w_prot':>10}  {'Delta':>8}  {'n':>4}")
        print("  " + "─" * 75)

        deltas = []
        for cat in all_cats:
            bl_r = compute_rates(bl_by_cat.get(cat, []))
            w_r  = compute_rates(w_by_cat.get(cat, []))
            if bl_r["n_valid"] < 2 or w_r["n_valid"] < 2:
                continue
            delta = w_r["jb_rate"] - bl_r["jb_rate"]
            deltas.append((cat, bl_r, w_r, delta))

        deltas.sort(key=lambda x: x[3])  # most protective first

        for cat, bl_r, w_r, delta in deltas:
            bl_str = f"{bl_r['n_jailbroken']}/{bl_r['n_valid']}={100*bl_r['jb_rate']:.0f}%"
            w_str  = f"{w_r['n_jailbroken']}/{w_r['n_valid']}={100*w_r['jb_rate']:.0f}%"
            print(f"  {cat:35s}  {bl_str:>10}  {w_str:>10}  {delta:>+8.2f}  "
                  f"{w_r['n_valid']:>4}")

    print(f"\n{sep}")
    print("Done.")


if __name__ == "__main__":
    main()
