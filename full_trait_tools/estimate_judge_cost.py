#!/usr/bin/env python3
"""
estimate_judge_cost.py

Estimates GPT-4.1-mini judging cost for the traits40 pipeline
without making any API calls.

Usage:
  uv run python estimate_judge_cost.py
  uv run python estimate_judge_cost.py --responses_dir full_trait_output/traits40_generation/responses
"""

import argparse
import json
from pathlib import Path

import tiktoken

# GPT-4.1-mini pricing (per 1M tokens, as of April 2026)
PRICE_INPUT_PER_M  = 0.40
PRICE_OUTPUT_PER_M = 1.60
JUDGE_MODEL        = "gpt-4.1-mini"
MAX_OUTPUT_TOKENS  = 10  # judge outputs a number 0-100

def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))

def load_eval_prompt(trait_file: Path) -> str:
    with open(trait_file, encoding="utf-8") as f:
        return json.load(f).get("eval_prompt", "")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses_dir", default="full_trait_output/traits40_generation/responses")
    parser.add_argument("--traits_dir",    default="data/traits/instructions")
    parser.add_argument("--scores_dir",    default="full_trait_output/traits40_judge/scores",
                        help="If provided, skip already-scored responses")
    parser.add_argument("--count_already_scored", action="store_true",
                        help="Also report cost of already-completed runs")
    args = parser.parse_args()

    responses_dir = Path(args.responses_dir)
    traits_dir    = Path(args.traits_dir)
    scores_dir    = Path(args.scores_dir)

    try:
        enc = tiktoken.encoding_for_model("gpt-4o-mini")  # same tokenizer family
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    total_input_tokens  = 0
    total_output_tokens = 0
    total_prompts       = 0
    already_scored      = 0
    traits_processed    = 0
    traits_skipped      = 0

    response_files = sorted(responses_dir.glob("*.jsonl"))
    if not response_files:
        print(f"No response files found in {responses_dir}")
        return

    import jsonlines

    for response_file in response_files:
        trait = response_file.stem
        trait_file = traits_dir / f"{trait}.json"
        if not trait_file.exists():
            traits_skipped += 1
            continue

        eval_prompt_template = load_eval_prompt(trait_file)
        if not eval_prompt_template:
            traits_skipped += 1
            continue

        # Load existing scores to skip already-done
        existing_keys = set()
        score_file = scores_dir / f"{trait}.json"
        if score_file.exists():
            try:
                existing_keys = set(json.load(open(score_file)).keys())
                already_scored += len(existing_keys)
            except Exception:
                pass

        traits_processed += 1

        with jsonlines.open(response_file) as reader:
            for resp in reader:
                key = resp["label"]
                if key in existing_keys and not args.count_already_scored:
                    continue

                assistant_response = resp.get("assistant_response", "")
                if not assistant_response:
                    assistant_response = next(
                        (m["content"] for m in resp.get("conversation", [])
                         if m["role"] == "assistant"), ""
                    )

                prompt = eval_prompt_template.format(
                    question=resp["question"],
                    answer=assistant_response,
                )

                n_in = count_tokens(prompt, enc)
                total_input_tokens  += n_in
                total_output_tokens += MAX_OUTPUT_TOKENS
                total_prompts       += 1

    cost_input  = total_input_tokens  / 1_000_000 * PRICE_INPUT_PER_M
    cost_output = total_output_tokens / 1_000_000 * PRICE_OUTPUT_PER_M
    cost_total  = cost_input + cost_output

    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  JUDGE COST ESTIMATE — {JUDGE_MODEL}")
    print(sep)
    print(f"  Traits processed        : {traits_processed}")
    print(f"  Traits skipped          : {traits_skipped} (no trait file)")
    print(f"  Prompts to send         : {total_prompts:,}")
    print(f"  Already scored (skipped): {already_scored:,}")
    print(f"")
    print(f"  Input tokens            : {total_input_tokens:,}")
    print(f"  Output tokens (est.)    : {total_output_tokens:,}")
    print(f"")
    print(f"  Input cost  (${PRICE_INPUT_PER_M}/1M)  : ${cost_input:.2f}")
    print(f"  Output cost (${PRICE_OUTPUT_PER_M}/1M) : ${cost_output:.2f}")
    print(f"  ─────────────────────────────────────")
    print(f"  Total estimated cost    : ${cost_total:.2f}")
    print(f"")
    print(f"  Est. time @ 100 req/s   : ~{total_prompts / 100 / 60:.1f} minutes")
    print(sep)
    print(f"\nNote: output tokens capped at {MAX_OUTPUT_TOKENS} per call (judge outputs a single number).")
    print(f"Actual cost may vary slightly with API overhead and retries.\n")


if __name__ == "__main__":
    main()
