#!/usr/bin/env python3
# Interview note: your previous estimator returned 0 because it looked for per-trait JSON files with eval_prompt fields, but your repo stores all trait metadata in one shared JSONL file.

import argparse
import json
from pathlib import Path

import jsonlines
import tiktoken


# update these if you use a different judge model / pricing
PRICING_PER_1M = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def load_trait_eval_prompts(extraction_questions_file: Path) -> dict[str, str]:
    """
    Expected: one JSON object per line, with at least:
      - "trait": trait name OR "name": trait name
      - "eval_prompt": template string containing {question} and {answer}
    """
    trait_to_eval_prompt = {}

    with jsonlines.open(extraction_questions_file, "r") as reader:
        for row in reader:
            trait = row.get("trait") or row.get("name")
            eval_prompt = row.get("eval_prompt", "")

            if trait and eval_prompt:
                trait_to_eval_prompt[trait] = eval_prompt

    return trait_to_eval_prompt


def load_responses(responses_file: Path) -> list[dict]:
    rows = []
    with jsonlines.open(responses_file, "r") as reader:
        for row in reader:
            rows.append(row)
    return rows


def get_first_assistant_message(conversation: list[dict]) -> str:
    for msg in conversation:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def main():
    parser = argparse.ArgumentParser(description="Estimate judge token usage and API cost.")
    parser.add_argument("--responses_dir", type=str, required=True)
    parser.add_argument("--extraction_questions_file", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--expected_output_tokens", type=int, default=5)
    parser.add_argument("--show_missing_traits", action="store_true")
    args = parser.parse_args()

    responses_dir = Path(args.responses_dir).expanduser()
    extraction_questions_file = Path(args.extraction_questions_file).expanduser()

    if not responses_dir.exists():
        raise FileNotFoundError(f"responses_dir does not exist: {responses_dir}")
    if not extraction_questions_file.exists():
        raise FileNotFoundError(
            f"extraction_questions_file does not exist: {extraction_questions_file}"
        )

    trait_to_eval_prompt = load_trait_eval_prompts(extraction_questions_file)
    if not trait_to_eval_prompt:
        raise RuntimeError(
            "Loaded 0 eval prompts from extraction_questions.jsonl. "
            "Check the field names. Expected one of ['trait', 'name'] plus 'eval_prompt'."
        )

    try:
        enc = tiktoken.encoding_for_model(args.judge_model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    response_files = sorted(responses_dir.glob("*.jsonl"))
    if not response_files:
        raise RuntimeError(f"No .jsonl files found in {responses_dir}")

    pricing = PRICING_PER_1M.get(args.judge_model)
    if pricing is None:
        raise RuntimeError(
            f"No pricing configured for model '{args.judge_model}'. "
            f"Add it to PRICING_PER_1M."
        )

    total_prompts = 0
    total_input_tokens = 0
    total_output_tokens = 0
    skipped_traits = []

    per_trait_rows = []

    for response_file in response_files:
        trait = response_file.stem
        eval_prompt_template = trait_to_eval_prompt.get(trait)

        if not eval_prompt_template:
            skipped_traits.append(trait)
            continue

        responses = load_responses(response_file)

        trait_prompt_count = 0
        trait_input_tokens = 0
        trait_output_tokens = 0

        for resp in responses:
            question = resp.get("question", "")
            assistant_response = get_first_assistant_message(resp.get("conversation", []))

            judge_prompt = eval_prompt_template.format(
                question=question,
                answer=assistant_response,
            )

            in_tokens = len(enc.encode(judge_prompt))
            out_tokens = args.expected_output_tokens

            trait_prompt_count += 1
            trait_input_tokens += in_tokens
            trait_output_tokens += out_tokens

        total_prompts += trait_prompt_count
        total_input_tokens += trait_input_tokens
        total_output_tokens += trait_output_tokens

        trait_input_cost = (trait_input_tokens / 1_000_000) * pricing["input"]
        trait_output_cost = (trait_output_tokens / 1_000_000) * pricing["output"]

        per_trait_rows.append(
            {
                "trait": trait,
                "prompts": trait_prompt_count,
                "input_tokens": trait_input_tokens,
                "output_tokens_est": trait_output_tokens,
                "input_cost_usd": trait_input_cost,
                "output_cost_usd": trait_output_cost,
                "total_cost_usd": trait_input_cost + trait_output_cost,
            }
        )

    input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
    output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    print("\n==== JUDGE COST ESTIMATE ====\n")
    print(f"responses_dir:              {responses_dir}")
    print(f"extraction_questions_file:  {extraction_questions_file}")
    print(f"judge_model:                {args.judge_model}")
    print(f"traits with eval prompts:   {len(trait_to_eval_prompt)}")
    print(f"response files found:       {len(response_files)}")
    print(f"traits processed:           {len(per_trait_rows)}")
    print(f"traits skipped:             {len(skipped_traits)}")
    print(f"total prompts:              {total_prompts:,}")
    print(f"total input tokens:         {total_input_tokens:,}")
    print(f"total output tokens (est):  {total_output_tokens:,}")
    print(f"avg input tokens/prompt:    {total_input_tokens / max(total_prompts, 1):.2f}")
    print()
    print(f"input cost (USD):           ${input_cost:.4f}")
    print(f"output cost (USD):          ${output_cost:.4f}")
    print(f"total cost (USD):           ${total_cost:.4f}")

    print("\n==== TOP 20 MOST EXPENSIVE TRAITS ====\n")
    per_trait_rows.sort(key=lambda x: x["total_cost_usd"], reverse=True)
    for row in per_trait_rows[:20]:
        print(
            f"{row['trait']:<20}  prompts={row['prompts']:<5}  "
            f"in_tok={row['input_tokens']:<8}  "
            f"cost=${row['total_cost_usd']:.4f}"
        )

    if args.show_missing_traits and skipped_traits:
        print("\n==== SKIPPED TRAITS (NO eval_prompt MATCH) ====\n")
        for trait in skipped_traits[:200]:
            print(trait)


if __name__ == "__main__":
    main()