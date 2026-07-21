#!/usr/bin/env python3
"""
prepare_wjb_harmbench_data.py

One-time setup: creates the HarmBench-format files needed to run GCG/PAIR/PAP/GPTFuzz/PEZ
on the WildJailbreak eval `adversarial_harmful` prompts (the same 2000 prompts used as the
51.2% steering baseline). Mirrors prepare_jbb_harmbench_data.py.

The attacked unit is the *adversarial prompt itself* (WJB eval has no vanilla field), so each
HarmBench "Behavior" is a full WJB adversarial prompt. Gradient attacks (GCG/PEZ) need an
affirmative optimization target; WJB has none, so we use a generic compliant opener.

Creates (under HARMBENCH_ROOT):
  data/behavior_datasets/wjb_behaviors.csv        — HarmBench behaviors CSV (Behavior = adv prompt)
  data/optimizer_targets/wjb_targets.json         — generic affirmative target per behavior (GCG/PEZ)
  data/behavior_datasets/wjb_behavior_meta.json   — behavior_id -> {goal, category, source, target}
  configs/method_configs/GCG_wjb_config.yaml      — GCG config redirected to wjb_targets.json
  configs/method_configs/PEZ_wjb_config.yaml      — PEZ config redirected to wjb_targets.json

Usage (from repo root):
  uv run python jailbreaking_tools/prepare_wjb_harmbench_data.py
"""

import csv
import json
import logging
import os
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HARMBENCH_ROOT = Path("/dlabscratch1/bazina/HarmBench")

# Generic affirmative target for gradient attacks (GCG/PEZ). WJB adv prompts have no
# per-behavior target string, so we optimize toward a generic compliant opener.
GENERIC_TARGET = "Sure, here is the information you requested"


def fetch_wjb_adversarial_harmful(max_behaviors=None):
    """Load WildJailbreak eval config, adversarial_harmful rows. Same source as the
    51.2% steering baseline (fetch_wildjailbreak_behaviors in trait_steering_attack_jbb.py)."""
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    logger.info("Loading allenai/wildjailbreak (config=eval)...")
    ds = load_dataset("allenai/wildjailbreak", name="eval", split="train", token=token)
    behaviors = []
    for i, row in enumerate(ds):
        dtype = (row.get("data_type") or "").lower()
        if dtype != "adversarial_harmful":
            continue
        prompt = (row.get("adversarial") or "").strip()
        if not prompt:
            continue
        bid = f"wjb_{len(behaviors):04d}"
        behaviors.append({
            "behavior_id": bid,
            "goal":        prompt,            # the adversarial prompt IS the task
            "target":      GENERIC_TARGET,
            "category":    "wildjailbreak",
            "source":      "WildJailbreak",
        })
        if max_behaviors and len(behaviors) >= max_behaviors:
            break
    logger.info(f"Loaded {len(behaviors)} WJB adversarial_harmful prompts")
    return behaviors


def write_behaviors_csv(behaviors, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Behavior", "FunctionalCategory", "SemanticCategory",
                        "Tags", "ContextString", "BehaviorID"],
        )
        writer.writeheader()
        for b in behaviors:
            writer.writerow({
                "Behavior":            b["goal"],
                "FunctionalCategory":  "standard",
                "SemanticCategory":    "wildjailbreak",
                "Tags":                "",
                "ContextString":       "",
                "BehaviorID":          b["behavior_id"],
            })
    logger.info(f"Wrote {len(behaviors)} rows → {out_path}")


def write_targets_json(behaviors, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    targets = {b["behavior_id"]: b["target"] for b in behaviors}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(targets, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {len(targets)} targets → {out_path}")


def write_behavior_meta(behaviors, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {b["behavior_id"]: {k: b[k] for k in ("goal", "target", "category", "source")}
            for b in behaviors}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote behavior meta → {out_path}")


def write_attack_config(template_path: Path, out_path: Path, wjb_targets_path: Path):
    """Copy an existing method config and override ALL targets_path lines → wjb_targets.json.
    PAIR/GPTFuzz configs set targets_path in several experiment blocks, so replace all."""
    text = template_path.read_text(encoding="utf-8")
    # Preserve each line's original indentation (PAIR nests targets_path 4 spaces deep under
    # default_method_hyperparameters; collapsing to 2 spaces breaks the YAML mapping).
    text = re.sub(r"^(\s*)targets_path:.*$",
                  lambda m: f"{m.group(1)}targets_path: {wjb_targets_path}",
                  text, flags=re.MULTILINE)
    out_path.write_text(text, encoding="utf-8")
    logger.info(f"Wrote WJB config → {out_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_behaviors", type=int, default=None,
                    help="Cap number of WJB prompts (default: all adversarial_harmful)")
    args = ap.parse_args()

    behaviors = fetch_wjb_adversarial_harmful(args.max_behaviors)
    assert len(behaviors) > 0, "No behaviors loaded"

    behaviors_csv = HARMBENCH_ROOT / "data/behavior_datasets/wjb_behaviors.csv"
    targets_json  = HARMBENCH_ROOT / "data/optimizer_targets/wjb_targets.json"
    meta_json     = HARMBENCH_ROOT / "data/behavior_datasets/wjb_behavior_meta.json"

    write_behaviors_csv(behaviors, behaviors_csv)
    write_targets_json(behaviors, targets_json)
    write_behavior_meta(behaviors, meta_json)

    # Attacks that reference targets_path (GCG/PEZ optimize toward it; PAIR/GPTFuzz use it as
    # the target string). PAP is target-agnostic and needs no override.
    for attack in ("GCG", "PEZ", "PAIR", "GPTFuzz"):
        template = HARMBENCH_ROOT / f"configs/method_configs/{attack}_config.yaml"
        out      = HARMBENCH_ROOT / f"configs/method_configs/{attack}_wjb_config.yaml"
        if template.exists():
            write_attack_config(template, out, targets_json)
        else:
            logger.warning(f"Template not found: {template}, skipping {attack} config")

    logger.info("Done.")
    logger.info(f"  behaviors: {behaviors_csv}")
    logger.info(f"  targets:   {targets_json}")
    logger.info(f"  meta:      {meta_json}")


if __name__ == "__main__":
    main()
