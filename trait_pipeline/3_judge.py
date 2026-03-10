#!/usr/bin/env python3
"""
Score trait responses using a judge LLM.

Mirrors scripts/3_judge.py exactly. The only differences:
  - Reads eval_prompt from trait JSON files (same field name, same format).
  - Parses scores on a 0-100 scale instead of 0-3.
  - Treats "REFUSAL" responses as score=0 (same logic, different scale).

Usage:
    uv run trait_pipeline/3_judge.py \
        --responses_dir outputs/llama-3.1-8b/trait_responses \
        --traits_dir data/prompts/traits \
        --output_dir outputs/llama-3.1-8b/trait_scores \
        --judge_model gpt-4.1-mini
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.judge import RateLimiter, call_judge_batch
import openai

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def load_trait_eval_prompt(trait_file: Path) -> str:
    """Load eval_prompt from trait JSON file. Same as load_role_eval_prompt in 3_judge.py."""
    with open(trait_file, 'r') as f:
        data = json.load(f)
    return data.get("eval_prompt", "")


def load_responses(responses_file: Path) -> List[dict]:
    responses = []
    with jsonlines.open(responses_file, 'r') as reader:
        for entry in reader:
            responses.append(entry)
    return responses


def parse_judge_score(response_text: str) -> Optional[int]:
    """
    Parse a 0-100 score from judge response.

    The trait eval_prompt asks for:
      - "REFUSAL" if the model refused → we treat as 0
      - An integer 0-100

    Mirrors parse_judge_score from assistant_axis/judge.py but for 0-100 scale.
    """
    text = response_text.strip()
    if text.upper().startswith("REFUSAL"):
        return 0
    try:
        score = int(text.split()[0])
        if 0 <= score <= 100:
            return score
    except (ValueError, IndexError):
        pass
    return None


async def process_trait(
    trait: str,
    responses: List[dict],
    eval_prompt_template: str,
    client: openai.AsyncOpenAI,
    rate_limiter: RateLimiter,
    judge_model: str,
    max_tokens: int,
    batch_size: int,
    existing_scores: Dict[str, int],
) -> dict:
    """Process a single trait. Mirrors process_role from 3_judge.py."""
    prompts = []
    keys = []

    for resp in responses:
        key = resp["label"]
        if key in existing_scores:
            continue

        assistant_response = ""
        for msg in resp["conversation"]:
            if msg["role"] == "assistant":
                assistant_response = msg["content"]
                break

        judge_prompt = eval_prompt_template.format(
            question=resp["question"],
            answer=assistant_response
        )
        prompts.append(judge_prompt)
        keys.append(key)

    if not prompts:
        return {}

    logger.info(f"Scoring {len(prompts)} new responses for {trait}...")
    responses_text = await call_judge_batch(
        client=client,
        prompts=prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size,
    )

    scores = {}
    for key, response_text in zip(keys, responses_text):
        if response_text:
            score = parse_judge_score(response_text)
            if score is not None:
                scores[key] = score

    return scores


async def main_async():
    parser = argparse.ArgumentParser(description="Score trait responses with judge LLM")
    parser.add_argument("--responses_dir", type=str, required=True)
    parser.add_argument("--traits_dir", type=str, default="../data/extraction_questions/traits")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--max_tokens", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--requests_per_second", type=int, default=100)
    parser.add_argument("--traits", nargs="+", help="Specific traits to process")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    responses_dir = Path(args.responses_dir)
    traits_dir = Path(args.traits_dir)

    response_files = sorted(responses_dir.glob("*.jsonl"))
    if args.traits:
        response_files = [f for f in response_files if f.stem in args.traits]

    logger.info(f"Processing {len(response_files)} traits")

    if args.dry_run:
        logger.info("Dry run mode - no API calls will be made")
        total_prompts = 0
        sample_shown = False

        for response_file in response_files:
            trait = response_file.stem
            output_file = output_dir / f"{trait}.json"

            existing_scores = {}
            if output_file.exists():
                try:
                    with open(output_file, 'r') as f:
                        existing_scores = json.load(f)
                except Exception:
                    pass

            trait_file = traits_dir / f"{trait}.json"
            if not trait_file.exists():
                logger.info(f"  {trait}: no trait file, skipping")
                continue

            eval_prompt_template = load_trait_eval_prompt(trait_file)
            if not eval_prompt_template:
                logger.info(f"  {trait}: no eval_prompt, skipping")
                continue

            responses = load_responses(response_file)
            prompts_for_trait = sum(1 for r in responses if r["label"] not in existing_scores)

            if prompts_for_trait > 0:
                total_prompts += prompts_for_trait
                logger.info(f"  {trait}: {prompts_for_trait} prompts")

                if not sample_shown:
                    resp = next(r for r in responses if r["label"] not in existing_scores)
                    assistant_response = next(
                        (m["content"] for m in resp["conversation"] if m["role"] == "assistant"), ""
                    )
                    sample_prompt = eval_prompt_template.format(
                        question=resp["question"], answer=assistant_response
                    )
                    logger.info("\n" + "=" * 60)
                    logger.info("SAMPLE JUDGE PROMPT:")
                    logger.info("=" * 60)
                    logger.info(sample_prompt)
                    logger.info("=" * 60 + "\n")
                    sample_shown = True

        logger.info(f"\nTotal prompts to send: {total_prompts}")
        return

    client = openai.AsyncOpenAI()
    rate_limiter = RateLimiter(args.requests_per_second)

    successful = 0
    skipped = 0
    failed = 0
    errors = []

    for response_file in tqdm(response_files, desc="Scoring traits"):
        trait = response_file.stem
        output_file = output_dir / f"{trait}.json"

        existing_scores = {}
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    existing_scores = json.load(f)
            except Exception:
                pass

        trait_file = traits_dir / f"{trait}.json"
        if not trait_file.exists():
            logger.info(f"Skipping {trait}: no trait file")
            skipped += 1
            continue

        eval_prompt_template = load_trait_eval_prompt(trait_file)
        if not eval_prompt_template:
            logger.info(f"Skipping {trait}: no eval_prompt")
            skipped += 1
            continue

        responses = load_responses(response_file)
        if not responses:
            errors.append(f"{trait}: no responses")
            failed += 1
            continue

        all_scored = all(r["label"] in existing_scores for r in responses)
        if all_scored:
            logger.info(f"Skipping {trait}: all {len(responses)} responses already scored")
            skipped += 1
            continue

        try:
            new_scores = await process_trait(
                trait=trait,
                responses=responses,
                eval_prompt_template=eval_prompt_template,
                client=client,
                rate_limiter=rate_limiter,
                judge_model=args.judge_model,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                existing_scores=existing_scores,
            )

            all_scores = {**existing_scores, **new_scores}
            with open(output_file, 'w') as f:
                json.dump(all_scores, f, indent=2)

            logger.info(f"Saved {len(all_scores)} scores for {trait} ({len(new_scores)} new)")
            successful += 1

        except Exception as e:
            errors.append(f"{trait}: {e}")
            failed += 1

    logger.info(f"\nSummary: {successful} successful, {skipped} skipped, {failed} failed")
    if errors:
        for err in errors[:10]:
            logger.info(f"  - {err}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
