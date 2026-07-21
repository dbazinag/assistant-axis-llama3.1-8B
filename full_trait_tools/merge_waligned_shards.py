"""Merge sharded trait_steering_attack_jbb.py w_aligned runs (results.jsonl per shard)
into one combined results.jsonl + summary.json (per-pair, per-behavior, by_category).
Usage: python merge_waligned_shards.py <output_dir>/w_aligned  (parent containing shard*/ dirs)
"""
import argparse, json, sys
from pathlib import Path
from collections import defaultdict


def compute_metrics(results, jailbroken_key="final_jailbroken"):
    if not results:
        return {"n": 0, "per_pair_asr": 0.0, "per_behavior_asr": 0.0}
    per_pair = sum(1 for r in results if r[jailbroken_key]) / len(results)
    behavior_jb = {}
    for r in results:
        bid = r["behavior_id"]
        behavior_jb[bid] = behavior_jb.get(bid, False) or r[jailbroken_key]
    per_behavior = sum(behavior_jb.values()) / len(behavior_jb) if behavior_jb else 0.0
    return {"n": len(results), "n_behaviors": len(behavior_jb),
            "per_pair_asr": per_pair, "per_behavior_asr": per_behavior}


def compute_category_metrics(results, jailbroken_key="final_jailbroken"):
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    return {cat: compute_metrics(rows, jailbroken_key) for cat, rows in sorted(by_cat.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode_dir", help="e.g. full_trait_output/harmbench_waligned_tpl_a7/w_aligned")
    args = ap.parse_args()

    mode_dir = Path(args.mode_dir)
    shard_dirs = sorted(mode_dir.glob("shard*"), key=lambda p: int(p.name.replace("shard", "")))
    if not shard_dirs:
        sys.exit(f"No shard* dirs found under {mode_dir}")

    all_rows = []
    for sd in shard_dirs:
        rp = sd / "results.jsonl"
        if not rp.exists():
            print(f"  [warn] missing {rp}")
            continue
        with open(rp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_rows.append(json.loads(line))
    print(f"Merged {len(all_rows)} pairs from {len(shard_dirs)} shards")

    merged_path = mode_dir / "results.jsonl"
    with open(merged_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "mode": "w_aligned",
        "n_shards": len(shard_dirs),
        "gpt41mini": {
            **compute_metrics(all_rows, "final_jailbroken"),
            "by_category": compute_category_metrics(all_rows, "final_jailbroken"),
        },
    }
    with open(mode_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {merged_path} and summary.json")
    print(f"Overall per_pair_asr = {summary['gpt41mini']['per_pair_asr']:.4f}")


if __name__ == "__main__":
    main()
