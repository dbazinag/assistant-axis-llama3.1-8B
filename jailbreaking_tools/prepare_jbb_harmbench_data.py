#!/usr/bin/env python3
"""
prepare_jbb_harmbench_data.py

One-time setup: creates the HarmBench-format files needed to run GCG/PAIR/PAP/GPTFuzz/PEZ
on the 100 JailbreakBench behaviors.

Creates:
  {HARMBENCH_ROOT}/data/behavior_datasets/jbb_behaviors.csv
      HarmBench-format behaviors CSV with JBB's 100 goals.

  {HARMBENCH_ROOT}/data/optimizer_targets/jbb_targets.json
      Optimization target strings (from JBB's Target field) keyed by behavior_id.
      Used by gradient-based attacks (GCG, PEZ).

  {HARMBENCH_ROOT}/configs/method_configs/GCG_jbb_config.yaml
  {HARMBENCH_ROOT}/configs/method_configs/PEZ_jbb_config.yaml
      Minimal overrides of the original configs that redirect targets_path → jbb_targets.json.

  {HARMBENCH_ROOT}/data/behavior_datasets/jbb_behavior_meta.json
      Maps behavior_id → {goal, category, source, target} for downstream eval scripts.

Usage (from repo root):
  uv run python jailbreaking_tools/prepare_jbb_harmbench_data.py
"""

import csv
import json
import logging
import re
import shutil
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HARMBENCH_ROOT = Path("/dlabscratch1/bazina/HarmBench")


def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()


def fetch_jbb_behaviors():
    logger.info("Loading JBB behaviors from HuggingFace...")
    try:
        from datasets import load_dataset
        raw = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        split = next(iter(raw.values()))
        behaviors = []
        for row in split:
            behaviors.append({
                "behavior_id": slugify(row["Behavior"]),
                "behavior":    row["Behavior"],
                "goal":        row["Goal"],
                "target":      row["Target"],
                "category":    row["Category"],
                "source":      row["Source"],
            })
        logger.info(f"Loaded {len(behaviors)} JBB behaviors")
        return behaviors
    except Exception as e:
        logger.warning(f"HuggingFace load failed ({e}), trying CSV fallback...")
        return _fetch_csv_fallback()


def _fetch_csv_fallback():
    import csv, io
    url = ("https://raw.githubusercontent.com/JailbreakBench/jailbreakbench"
           "/main/src/jailbreakbench/data/behaviors.csv")
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    behaviors = []
    for row in reader:
        beh_name = row.get("Behavior", row.get("behavior_id", ""))
        behaviors.append({
            "behavior_id": slugify(beh_name),
            "behavior":    beh_name,
            "goal":        row.get("Goal", row.get("goal", "")),
            "target":      row.get("Target", row.get("target", "")),
            "category":    row.get("Category", row.get("category", "")),
            "source":      row.get("Source", row.get("source", "")),
        })
    logger.info(f"Loaded {len(behaviors)} JBB behaviors via CSV fallback")
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
            # Sanitise category for SemanticCategory column
            sem_cat = re.sub(r"[^a-zA-Z0-9_/]", "_", b["category"]).lower()
            writer.writerow({
                "Behavior":            b["goal"],
                "FunctionalCategory":  "standard",
                "SemanticCategory":    sem_cat,
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
    meta = {b["behavior_id"]: {k: b[k] for k in ("goal", "target", "category", "source", "behavior")}
            for b in behaviors}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote behavior meta → {out_path}")


def write_attack_config(template_path: Path, out_path: Path, jbb_targets_path: Path):
    """Copy an existing method config and override targets_path → jbb_targets.json."""
    text = template_path.read_text(encoding="utf-8")
    # Replace the targets_path line in default_method_hyperparameters
    new_targets = f"  targets_path: {jbb_targets_path}"
    text = re.sub(r"^\s*targets_path:.*$", new_targets, text, flags=re.MULTILINE)
    out_path.write_text(text, encoding="utf-8")
    logger.info(f"Wrote JBB config → {out_path}")


def main():
    behaviors = fetch_jbb_behaviors()
    assert len(behaviors) > 0, "No behaviors loaded"

    behaviors_csv  = HARMBENCH_ROOT / "data/behavior_datasets/jbb_behaviors.csv"
    targets_json   = HARMBENCH_ROOT / "data/optimizer_targets/jbb_targets.json"
    meta_json      = HARMBENCH_ROOT / "data/behavior_datasets/jbb_behavior_meta.json"

    write_behaviors_csv(behaviors, behaviors_csv)
    write_targets_json(behaviors, targets_json)
    write_behavior_meta(behaviors, meta_json)

    # Create JBB-specific attack configs for gradient-based attacks
    for attack in ("GCG", "PEZ"):
        template = HARMBENCH_ROOT / f"configs/method_configs/{attack}_config.yaml"
        out      = HARMBENCH_ROOT / f"configs/method_configs/{attack}_jbb_config.yaml"
        if template.exists():
            write_attack_config(template, out, targets_json)
        else:
            logger.warning(f"Template not found: {template}, skipping {attack} config")

    logger.info("Done. Files written to HarmBench data directories.")
    logger.info(f"  behaviors: {behaviors_csv}")
    logger.info(f"  targets:   {targets_json}")
    logger.info(f"  meta:      {meta_json}")


if __name__ == "__main__":
    main()
