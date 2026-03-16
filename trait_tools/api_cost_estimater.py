#!/usr/bin/env python3
"""
Estimate OpenAI API cost for judging trait responses.

It reconstructs every judge prompt exactly like the judge script,
counts tokens with tiktoken, and estimates total API cost.

Usage:
    python estimate_judge_cost.py \
        --responses_dir outputs/llama-3.1-8b/trait_responses \
        --traits_dir data/prompts/traits \
        --judge_model gpt-4.1-mini
"""

import argparse
import json
from pathlib import Path

import jsonlines
import tiktoken


# ----- MODEL PRICING (USD per 1M tokens) -----
PRICING = {
    "gpt-4.1-mini": {
        "input": 0.15,
        "output": 0.60
    },
    "gpt-4.1": {
        "input": 5.00,
        "output": 15.00
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60
    }
}


def load_trait_eval_prompt(trait_file: Path):
    with open(trait_file) as f:
        data = json.load(f)
    return data.get("eval_prompt", "")


def load_responses(responses_file: Path):
    responses = []
    with jsonlines.open(responses_file) as reader:
        for r in reader:
            responses.append(r)
    return responses


def count_tokens(text, enc):
    return len(enc.encode(text))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--responses_dir", required=True)
    parser.add_argument("--traits_dir", required=True)
    parser.add_argument("--judge_model", default="gpt-4.1-mini")
    parser.add_argument("--expected_output_tokens", type=int, default=5)

    args = parser.parse_args()

    responses_dir = Path(args.responses_dir)
    traits_dir = Path(args.traits_dir)

    enc = tiktoken.encoding_for_model(args.judge_model)

    total_prompts = 0
    total_input_tokens = 0
    total_output_tokens = 0

    response_files = sorted(responses_dir.glob("*.jsonl"))

    for rf in response_files:

        trait = rf.stem
        trait_file = traits_dir / f"{trait}.json"

        if not trait_file.exists():
            continue

        eval_prompt_template = load_trait_eval_prompt(trait_file)

        responses = load_responses(rf)

        for resp in responses:

            assistant_response = ""
            for msg in resp["conversation"]:
                if msg["role"] == "assistant":
                    assistant_response = msg["content"]
                    break

            judge_prompt = eval_prompt_template.format(
                question=resp["question"],
                answer=assistant_response
            )

            tokens = count_tokens(judge_prompt, enc)

            total_prompts += 1
            total_input_tokens += tokens
            total_output_tokens += args.expected_output_tokens

    pricing = PRICING[args.judge_model]

    input_cost = total_input_tokens / 1_000_000 * pricing["input"]
    output_cost = total_output_tokens / 1_000_000 * pricing["output"]

    total_cost = input_cost + output_cost

    print("\n==== COST ESTIMATE ====\n")

    print(f"Prompts: {total_prompts}")
    print(f"Input tokens: {total_input_tokens:,}")
    print(f"Output tokens (est): {total_output_tokens:,}")

    print("\n---- Estimated Cost ----\n")

    print(f"Input cost:  ${input_cost:.4f}")
    print(f"Output cost: ${output_cost:.4f}")
    print(f"Total cost:  ${total_cost:.4f}")

    avg = total_input_tokens / max(total_prompts, 1)
    print(f"\nAvg tokens per prompt: {avg:.1f}")


if __name__ == "__main__":
    main()