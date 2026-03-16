#!/usr/bin/env python3
# Interview note: this version matches your actual schema:
# - one shared extraction_questions.jsonl with only {id, question}
# - one response file per trait, where each row already contains question, trait, polarity, label, and conversation

import argparse
from pathlib import Path

import jsonlines
import tiktoken


PRICING_PER_1M = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with jsonlines.open(path, "r") as reader:
        for row in reader:
            rows.append(row)
    return rows


def get_first_assistant_message(conversation: list[dict]) -> str:
    for msg in conversation:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def get_user_question(resp: dict) -> str:
    if isinstance(resp.get("question"), str):
        return resp["question"]

    for msg in resp.get("conversation", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Estimate judge API token usage/cost for trait responses."
    )
    parser.add_argument("--responses_dir", type=str, required=True)
    parser.add_argument("--extraction_questions_file", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini")
    parser.add_argument(
        "--expected_output_tokens",
        type=int,
        default=5,
        help="Expected tokens in judge output, e.g. '73' or 'REFUSAL'.",
    )
    parser.add_argument(
        "--judge_wrapper_tokens",
        type=int,
        default=120,
        help="Approx fixed token overhead for the hidden eval prompt/instructions.",
    )
    parser.add_argument(
        "--show_top",
        type=int,
        default=20,
        help="How many most-expensive traits to print.",
    )
    args = parser.parse_args()

    responses_dir = Path(args.responses_dir).expanduser()
    extraction_questions_file = Path(args.extraction_questions_file).expanduser()

    if not responses_dir.exists():
        raise FileNotFoundError(f"responses_dir does not exist: {responses_dir}")
    if not extraction_questions_file.exists():
        raise FileNotFoundError(
            f"extraction_questions_file does not exist: {extraction_questions_file}"
        )
    if args.judge_model not in PRICING_PER_1M:
        raise RuntimeError(
            f"Missing pricing for model '{args.judge_model}'. "
            f"Add it to PRICING_PER_1M."
        )

    try:
        enc = tiktoken.encoding_for_model(args.judge_model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    pricing = PRICING_PER_1M[args.judge_model]

    # only used as a sanity check, since the actual question text is already inside each response row
    extraction_rows = load_jsonl(extraction_questions_file)
    if not extraction_rows:
        raise RuntimeError("extraction_questions.jsonl is empty")
    question_ids = {row["id"] for row in extraction_rows if "id" in row}

    response_files = sorted(responses_dir.glob("*.jsonl"))
    if not response_files:
        raise RuntimeError(f"No .jsonl files found in {responses_dir}")

    total_prompts = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_matched_question_ids = 0
    total_missing_question_ids = 0

    per_trait = []

    for response_file in response_files:
        rows = load_jsonl(response_file)
        if not rows:
            continue

        trait_name = rows[0].get("trait", response_file.stem)

        trait_prompts = 0
        trait_input_tokens = 0
        trait_output_tokens = 0
        trait_matched_question_ids = 0
        trait_missing_question_ids = 0

        for resp in rows:
            question = get_user_question(resp)
            answer = get_first_assistant_message(resp.get("conversation", []))
            qidx = resp.get("question_index")

            if qidx in question_ids:
                trait_matched_question_ids += 1
            else:
                trait_missing_question_ids += 1

            # approximate actual judge request:
            # [trait-specific eval instructions] + question + answer
            # since the eval prompt text is not in extraction_questions.jsonl,
            # we model its fixed overhead with judge_wrapper_tokens
            visible_payload = f"Question:\n{question}\n\nAnswer:\n{answer}\n"
            input_tokens = len(enc.encode(visible_payload)) + args.judge_wrapper_tokens
            output_tokens = args.expected_output_tokens

            trait_prompts += 1
            trait_input_tokens += input_tokens
            trait_output_tokens += output_tokens

        trait_input_cost = (trait_input_tokens / 1_000_000) * pricing["input"]
        trait_output_cost = (trait_output_tokens / 1_000_000) * pricing["output"]
        trait_total_cost = trait_input_cost + trait_output_cost

        per_trait.append(
            {
                "trait": trait_name,
                "prompts": trait_prompts,
                "input_tokens": trait_input_tokens,
                "output_tokens": trait_output_tokens,
                "input_cost_usd": trait_input_cost,
                "output_cost_usd": trait_output_cost,
                "total_cost_usd": trait_total_cost,
                "matched_question_ids": trait_matched_question_ids,
                "missing_question_ids": trait_missing_question_ids,
            }
        )

        total_prompts += trait_prompts
        total_input_tokens += trait_input_tokens
        total_output_tokens += trait_output_tokens
        total_matched_question_ids += trait_matched_question_ids
        total_missing_question_ids += trait_missing_question_ids

    total_input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
    total_output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
    total_cost = total_input_cost + total_output_cost

    print("\n==== JUDGE API COST ESTIMATE ====\n")
    print(f"responses_dir:               {responses_dir}")
    print(f"extraction_questions_file:   {extraction_questions_file}")
    print(f"judge_model:                 {args.judge_model}")
    print(f"response files found:        {len(response_files)}")
    print(f"questions in bank:           {len(question_ids)}")
    print(f"matched question_index rows: {total_matched_question_ids:,}")
    print(f"missing question_index rows: {total_missing_question_ids:,}")
    print(f"total prompts:               {total_prompts:,}")
    print(f"total input tokens (est):    {total_input_tokens:,}")
    print(f"total output tokens (est):   {total_output_tokens:,}")
    print(f"avg input tokens/prompt:     {total_input_tokens / max(total_prompts, 1):.2f}")
    print()
    print(f"input cost (USD):            ${total_input_cost:.4f}")
    print(f"output cost (USD):           ${total_output_cost:.4f}")
    print(f"total cost (USD):            ${total_cost:.4f}")

    print(f"\n==== TOP {args.show_top} MOST EXPENSIVE TRAITS ====\n")
    per_trait.sort(key=lambda x: x["total_cost_usd"], reverse=True)
    for row in per_trait[: args.show_top]:
        print(
            f"{row['trait']:<20} "
            f"prompts={row['prompts']:<5} "
            f"in_tok={row['input_tokens']:<9} "
            f"cost=${row['total_cost_usd']:.4f}"
        )


if __name__ == "__main__":
    main()