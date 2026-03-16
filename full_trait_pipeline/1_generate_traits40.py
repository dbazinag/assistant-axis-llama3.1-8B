#!/usr/bin/env python3
# Interview note: verifies the generated trait outputs are complete and consistent with the per-trait 40-question files.

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import jsonlines


def load_trait(trait_file: Path) -> Dict:
    with open(trait_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def verify_trait_output_file(output_file: Path, trait_data: Dict) -> Tuple[bool, str]:
    rows = []
    with jsonlines.open(output_file, "r") as reader:
        for row in reader:
            rows.append(row)

    expected_count = 10 * 40
    if len(rows) != expected_count:
        return False, f"Expected {expected_count} rows, found {len(rows)}"

    seen = set()
    for row in rows:
        key = (row["polarity"], row["prompt_index"], row["question_index"])
        if key in seen:
            return False, f"Duplicate key found: {key}"
        seen.add(key)

        qidx = row["question_index"]
        expected_question = trait_data["questions"][qidx]
        if row["question"] != expected_question:
            return False, f"Question mismatch at qidx={qidx}"

        pidx = row["prompt_index"]
        expected_prompt = trait_data["instruction"][pidx]["pos"] if row["polarity"] == "positive" else trait_data["instruction"][pidx]["neg"]
        if row["system_prompt"] != expected_prompt:
            return False, f"System prompt mismatch at {row['polarity']} pidx={pidx}"

    for pidx in range(5):
        for polarity in ("positive", "negative"):
            for qidx in range(40):
                if (polarity, pidx, qidx) not in seen:
                    return False, f"Missing key: {(polarity, pidx, qidx)}"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description="Verify generated trait response files.")
    parser.add_argument("--traits_dir", type=str, default="data/traits/instructions")
    parser.add_argument("--responses_dir", type=str, default="full_trait_output/traits40_generation/responses")
    args = parser.parse_args()

    traits_dir = Path(args.traits_dir)
    responses_dir = Path(args.responses_dir)

    ok_count = 0
    fail_count = 0

    for trait_file in sorted(traits_dir.glob("*.json")):
        trait_name = trait_file.stem
        output_file = responses_dir / f"{trait_name}.jsonl"

        if not output_file.exists():
            print(f"[MISSING] {trait_name}: no output file")
            fail_count += 1
            continue

        trait_data = load_trait(trait_file)
        ok, msg = verify_trait_output_file(output_file, trait_data)
        if ok:
            print(f"[OK] {trait_name}")
            ok_count += 1
        else:
            print(f"[FAIL] {trait_name}: {msg}")
            fail_count += 1

    print()
    print(f"ok={ok_count} fail={fail_count}")


if __name__ == "__main__":
    main()