#!/usr/bin/env python3
# Interview note: verifies the trait generation outputs match the expected 5×2×40 structure.

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import jsonlines


EXPECTED_INSTRUCTION_PAIRS = 5
EXPECTED_QUESTIONS = 40
EXPECTED_ROWS = EXPECTED_INSTRUCTION_PAIRS * 2 * EXPECTED_QUESTIONS


def load_trait(trait_file: Path) -> Dict:
    with open(trait_file, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_trait_output(output_file: Path, trait_data: Dict) -> Tuple[bool, str]:

    rows = []

    with jsonlines.open(output_file, "r") as reader:
        for row in reader:
            rows.append(row)

    if len(rows) != EXPECTED_ROWS:
        return False, f"expected {EXPECTED_ROWS} rows but found {len(rows)}"

    seen = set()

    for row in rows:

        polarity = row["polarity"]
        prompt_index = row["prompt_index"]
        question_index = row["question_index"]

        key = (polarity, prompt_index, question_index)

        if key in seen:
            return False, f"duplicate entry detected: {key}"

        seen.add(key)

        # verify question text
        expected_question = trait_data["questions"][question_index]

        if row["question"] != expected_question:
            return False, f"question mismatch at index {question_index}"

        # verify system prompt
        if polarity == "positive":
            expected_prompt = trait_data["instruction"][prompt_index]["pos"]
        else:
            expected_prompt = trait_data["instruction"][prompt_index]["neg"]

        if row["system_prompt"] != expected_prompt:
            return False, f"system prompt mismatch for {key}"

    # verify full coverage
    for p in range(EXPECTED_INSTRUCTION_PAIRS):
        for polarity in ["positive", "negative"]:
            for q in range(EXPECTED_QUESTIONS):
                if (polarity, p, q) not in seen:
                    return False, f"missing entry {(polarity, p, q)}"

    return True, "ok"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--traits_dir",
        type=str,
        default="data/traits/instructions",
    )

    parser.add_argument(
        "--responses_dir",
        type=str,
        default="full_trait_output/traits40_generation/responses",
    )

    args = parser.parse_args()

    traits_dir = Path(args.traits_dir)
    responses_dir = Path(args.responses_dir)

    ok_count = 0
    fail_count = 0
    missing_count = 0

    for trait_file in sorted(traits_dir.glob("*.json")):

        trait_name = trait_file.stem
        response_file = responses_dir / f"{trait_name}.jsonl"

        if not response_file.exists():
            print(f"[MISSING] {trait_name}")
            missing_count += 1
            continue

        trait_data = load_trait(trait_file)

        ok, message = verify_trait_output(response_file, trait_data)

        if ok:
            print(f"[OK] {trait_name}")
            ok_count += 1
        else:
            print(f"[FAIL] {trait_name}: {message}")
            fail_count += 1

    print()
    print("SUMMARY")
    print("-------")
    print("OK:", ok_count)
    print("FAILED:", fail_count)
    print("MISSING:", missing_count)


if __name__ == "__main__":
    main()