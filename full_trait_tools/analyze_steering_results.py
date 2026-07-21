#!/usr/bin/env python3
"""
analyze_steering_results.py

Reads existing steering_robustness_v2_results.json and prints:
  1. Enhanced summary table with fractions (jailbroken/valid) and
     degenerate counts per condition per alpha
  2. Flip examples for PC1, w, and cautious:
     - One example where steering caused jailbroken → refused (protective flip)
     - One example where steering caused refused → jailbroken (jailbreak flip)
     For each: print baseline response + steered response side by side

Uses alpha=0.25 (highest) for flip examples to maximize signal.

Usage:
  uv run full_trait_tools/analyze_steering_results.py
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_rates(entries: List[dict]) -> dict:
    n_jb  = sum(1 for e in entries if e["label"] == "jailbroken")
    n_ref = sum(1 for e in entries if e["label"] == "refused")
    n_deg = sum(1 for e in entries if e["label"] == "degenerate")
    n_err = sum(1 for e in entries if e["label"] == "error")
    n_valid = n_jb + n_ref
    return {
        "n_jailbroken": n_jb, "n_refused": n_ref,
        "n_degenerate": n_deg, "n_error": n_err,
        "n_valid": n_valid,
        "jb_rate": n_jb / n_valid if n_valid > 0 else float("nan"),
    }


def print_enhanced_table(
    results: dict,
    condition_names: List[str],
    alphas: List[float],
    baseline_rates: Dict[float, float],
    title: str,
) -> None:
    sep = "=" * 105
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    # Header
    alpha_header = "".join(f"  {'a='+str(a):>20}" for a in alphas)
    print(f"\n  {'Condition':42s}{alpha_header}")
    print("  " + "─" * 103)

    for cond_name in condition_names:
        if cond_name not in results:
            continue
        row = f"  {cond_name:42s}"
        for alpha in alphas:
            alpha_str = str(alpha)
            entries = results[cond_name].get(alpha_str, [])
            if not entries:
                row += f"  {'—':>20}"
                continue
            rates = compute_rates(entries)
            jb_rate = rates["jb_rate"]
            n_valid = rates["n_valid"]
            n_jb    = rates["n_jailbroken"]
            n_deg   = rates["n_degenerate"]
            baseline = baseline_rates.get(alpha, float("nan"))

            if not np.isnan(jb_rate) and not np.isnan(baseline):
                delta = jb_rate - baseline
                cell = f"{n_jb}/{n_valid}={100*jb_rate:.0f}%({delta:+.2f}) deg={n_deg}"
            else:
                cell = "nan"
            row += f"  {cell:>20}"
        print(row)

    print(f"\n  Format: jailbroken/valid=JB%(Δ) deg=degenerate_count")
    print(f"  Negative Δ = protective. Degenerate excluded from JB% calculation.")
    print(sep)


def find_flip_example(
    protective_results: dict,
    induction_results: dict,
    cond_name: str,
    alpha: float,
    flip_type: str,  # "protective" or "induction"
    example_idx: int = 0,
) -> Optional[dict]:
    """
    Find a pair where steering caused a label flip at the given alpha.
    flip_type="protective": baseline=jailbroken, steered=refused
    flip_type="induction": baseline=refused, steered=jailbroken
    """
    alpha_str = str(alpha)

    if flip_type == "protective":
        baseline_entries = protective_results.get("baseline", {}).get(alpha_str, [])
        steered_entries  = protective_results.get(cond_name, {}).get(alpha_str, [])
        want_baseline    = "jailbroken"
        want_steered     = "refused"
    else:
        baseline_entries = induction_results.get("baseline", {}).get(alpha_str, [])
        steered_entries  = induction_results.get(cond_name, {}).get(alpha_str, [])
        want_baseline    = "refused"
        want_steered     = "jailbroken"

    if not baseline_entries or not steered_entries:
        return None

    # Index baseline by pair_id
    baseline_by_pid = {e["pair_id"]: e for e in baseline_entries}

    match_count = 0
    for steered in steered_entries:
        pid = steered["pair_id"]
        baseline = baseline_by_pid.get(pid)
        if baseline is None:
            continue
        if baseline["label"] == want_baseline and steered["label"] == want_steered:
            if match_count < example_idx:
                match_count += 1
                continue
            return {
                "pair_id":          pid,
                "behavior":         steered["behavior"],
                "baseline_label":   baseline["label"],
                "baseline_response": baseline["response"],
                "steered_label":    steered["label"],
                "steered_response": steered["response"],
                "alpha":            steered["alpha"],
            }
    return None


def print_flip_example(example: dict, title: str) -> None:
    sep = "─" * 90
    print(f"\n{sep}")
    print(f"  {title}")
    print(f"  Behavior: {example['behavior']}")
    print(f"  Pair ID:  {example['pair_id']} | Alpha: {example['alpha']:+.2f}")
    print(sep)

    print(f"\n  BASELINE [{example['baseline_label'].upper()}]:")
    response = example["baseline_response"]
    # Print first 500 chars
    for line in response[:500].split("\n"):
        print(f"    {line}")
    if len(response) > 500:
        print(f"    ... [{len(response)} chars total]")

    print(f"\n  STEERED [{example['steered_label'].upper()}]:")
    response = example["steered_response"]
    for line in response[:500].split("\n"):
        print(f"    {line}")
    if len(response) > 500:
        print(f"    ... [{len(response)} chars total]")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str,
        default="full_trait_output/steering_robustness_v2/steering_robustness_v2_results.json")
    parser.add_argument("--flip_example_idx", type=int, default=0, help="Which flip example to show (0=first)")
    parser.add_argument("--flip_alpha", type=float, default=0.25,
        help="Alpha level to use for flip examples")
    args = parser.parse_args()

    print(f"Loading results from {args.results_path}...")
    data = load_results(Path(args.results_path))

    alphas     = data["alphas"]
    protective = data["protective"]
    induction  = data.get("induction", {})

    # Compute baseline rates
    baseline_prot_rates = {}
    for alpha in alphas:
        entries = protective.get("baseline", {}).get(str(alpha), [])
        if entries:
            rates = compute_rates(entries)
            baseline_prot_rates[alpha] = rates["jb_rate"]

    baseline_ind_rates = {}
    for alpha in alphas:
        entries = induction.get("baseline", {}).get(str(alpha), [])
        if entries:
            rates = compute_rates(entries)
            baseline_ind_rates[alpha] = rates["jb_rate"]

    # ── Part 1: Enhanced summary tables ───────────────────────────────────────
    prot_conditions = list(protective.keys())
    print_enhanced_table(
        protective, prot_conditions, alphas, baseline_prot_rates,
        f"PROTECTIVE STEERING — jailbroken pairs  (n={data.get('n_jb', '?')})"
    )

    if induction:
        ind_conditions = list(induction.keys())
        print_enhanced_table(
            induction, ind_conditions, alphas, baseline_ind_rates,
            f"JAILBREAK INDUCTION — refused pairs  (n={data.get('n_refused', '?')})"
        )

    # ── Part 2: Flip examples ─────────────────────────────────────────────────
    flip_alpha = args.flip_alpha
    # Map condition names in results to trait labels
    traits_to_show = [
        ("PC1_protective (+PC1)",      "w_jailbreak (+w)",         "PC1"),
        ("w_protective (-w)",          "w_jailbreak (+w)",         "w"),
        ("cautious (+)",               "w_jailbreak (+w)",         "cautious"),
    ]
    # Note: for induction we use the induction result conditions
    # w_jailbreak is in induction results, PC1 and cautious only in protective

    sep = "=" * 90
    print(f"\n\n{sep}")
    print(f"  FLIP EXAMPLES  (alpha={flip_alpha})")
    print(f"  For each trait: one protective flip (jailbroken→refused)")
    print(f"                  one induction flip  (refused→jailbroken)")
    print(sep)

    # Conditions available in protective results
    prot_cond_names = list(protective.keys())
    ind_cond_names  = list(induction.keys()) if induction else []

    traits_config = [
        {
            "label":      "PC1",
            "prot_cond":  "PC1_protective (+PC1)",
            "ind_cond":   "w_jailbreak (+w)",   # best induction proxy we have
        },
        {
            "label":      "w",
            "prot_cond":  "w_protective (-w)",
            "ind_cond":   "w_jailbreak (+w)",
        },
        {
            "label":      "cautious",
            "prot_cond":  "cautious (+)",
            "ind_cond":   "w_jailbreak (+w)",   # cautious not tested in induction
        },
    ]

    for cfg in traits_config:
        label     = cfg["label"]
        prot_cond = cfg["prot_cond"]
        ind_cond  = cfg["ind_cond"]

        print(f"\n{'█'*90}")
        print(f"  TRAIT: {label}")
        print(f"{'█'*90}")

        # --- Baseline for this trait's protective pairs ---
        alpha_str = str(flip_alpha)
        baseline_entries = protective.get("baseline", {}).get(alpha_str, [])
        if baseline_entries:
            bl_rates = compute_rates(baseline_entries)
            print(f"\n  Baseline (jailbroken pairs): "
                  f"{bl_rates['n_jailbroken']}/{bl_rates['n_valid']} = "
                  f"{100*bl_rates['jb_rate']:.0f}% jailbroken "
                  f"(deg={bl_rates['n_degenerate']})")

        # --- Protective flip: jailbroken → refused ---
        prot_example = find_flip_example(
            protective, induction, prot_cond, flip_alpha, "protective", args.flip_example_idx
        )
        if prot_example:
            print_flip_example(
                prot_example,
                f"PROTECTIVE FLIP [{label}]: jailbroken → refused"
            )
        else:
            print(f"\n  No protective flip found for {label} at alpha={flip_alpha}")
            # Try lower alphas
            for alt_alpha in sorted(alphas, reverse=True):
                if alt_alpha == flip_alpha:
                    continue
                ex = find_flip_example(
                    protective, induction, prot_cond, alt_alpha, "protective", args.flip_example_idx
                )
                if ex:
                    print(f"  Found at alpha={alt_alpha} instead:")
                    print_flip_example(
                        ex, f"PROTECTIVE FLIP [{label}]: jailbroken → refused"
                    )
                    break

        # --- Induction flip: refused → jailbroken ---
        if induction and ind_cond in induction:
            ind_example = find_flip_example(
                protective, induction, ind_cond, flip_alpha, "induction", args.flip_example_idx
            )
            if ind_example:
                print_flip_example(
                    ind_example,
                    f"INDUCTION FLIP [{ind_cond}]: refused → jailbroken"
                )
            else:
                print(f"\n  No induction flip found for {ind_cond} at alpha={flip_alpha}")
                for alt_alpha in sorted(alphas, reverse=True):
                    if alt_alpha == flip_alpha:
                        continue
                    ex = find_flip_example(
                        protective, induction, ind_cond, alt_alpha, "induction", args.flip_example_idx
                    )
                    if ex:
                        print(f"  Found at alpha={alt_alpha} instead:")
                        print_flip_example(
                            ex, f"INDUCTION FLIP [{ind_cond}]: refused → jailbroken"
                        )
                        break
        else:
            print(f"\n  Induction results not available for {label}")

    print(f"\n{'='*90}")
    print("Done.")


if __name__ == "__main__":
    main()
