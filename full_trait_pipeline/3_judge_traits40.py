#!/usr/bin/env python3
# Interview note: scores traits40 responses with the per-trait eval_prompt, saves simple scores plus separate diagnostics.

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.judge import RateLimiter, call_judge_batch
import openai


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_trait_eval_prompt(trait_file: Path) -> str:
    with open(trait_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("eval_prompt", "")


def load_responses(responses_file: Path) -> List[dict]:
    responses = []
    with jsonlines.open(responses_file, "r") as reader:
        for entry in reader:
            responses.append(entry)
    return responses


def parse_judge_score(response_text: str) -> Optional[int]:
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
) -> Dict[str, Dict]:
    prompts = []
    keys = []

    for resp in responses:
        key = resp["label"]
        if key in existing_scores:
            continue

        assistant_response = resp.get("assistant_response", "")
        if not assistant_response:
            assistant_response = next(
                (m["content"] for m in resp["conversation"] if m["role"] == "assistant"),
                "",
            )

        judge_prompt = eval_prompt_template.format(
            question=resp["question"],
            answer=assistant_response,
        )
        prompts.append(judge_prompt)
        keys.append(key)

    if not prompts:
        return {
            "scores": {},
            "raw_judge_responses": {},
            "parse_failures": {},
            "stats": {
                "requested": 0,
                "parsed_successfully": 0,
                "parse_failures": 0,
                "refusals_as_zero": 0,
            },
        }

    logger.info(f"Scoring {len(prompts)} new responses for {trait}...")

    responses_text = await call_judge_batch(
        client=client,
        prompts=prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size,
    )

    scores: Dict[str, int] = {}
    raw_judge_responses: Dict[str, str] = {}
    parse_failures: Dict[str, Dict[str, str]] = {}
    refusal_count = 0

    for key, response_text in zip(keys, responses_text):
        if response_text is None:
            parse_failures[key] = {
                "reason": "empty_response",
                "raw_response": "",
            }
            continue

        raw_judge_responses[key] = response_text
        score = parse_judge_score(response_text)

        if score is not None:
            scores[key] = score
            if response_text.strip().upper().startswith("REFUSAL"):
                refusal_count += 1
        else:
            parse_failures[key] = {
                "reason": "parse_failed",
                "raw_response": response_text,
            }

    return {
        "scores": scores,
        "raw_judge_responses": raw_judge_responses,
        "parse_failures": parse_failures,
        "stats": {
            "requested": len(prompts),
            "parsed_successfully": len(scores),
            "parse_failures": len(parse_failures),
            "refusals_as_zero": refusal_count,
        },
    }


async def main_async():
    parser = argparse.ArgumentParser(description="Score traits40 responses with a judge LLM")
    parser.add_argument(
        "--responses_dir",
        type=str,
        default="full_trait_output/traits40_generation/responses",
    )
    parser.add_argument(
        "--traits_dir",
        type=str,
        default="data/traits/instructions",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="full_trait_output/traits40_judge",
    )
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

    output_root = Path(args.output_root)
    scores_dir = output_root / "scores"
    diagnostics_dir = output_root / "diagnostics"
    manifests_dir = output_root / "manifests"
    logs_dir = output_root / "logs"

    if not args.dry_run:
        scores_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        write_json(
            manifests_dir / "run_config.json",
            {
                "created_at_utc": utc_now_iso(),
                "responses_dir": str(Path(args.responses_dir).resolve()),
                "traits_dir": str(Path(args.traits_dir).resolve()),
                "output_root": str(output_root.resolve()),
                "judge_model": args.judge_model,
                "max_tokens": args.max_tokens,
                "batch_size": args.batch_size,
                "requests_per_second": args.requests_per_second,
                "selected_traits": args.traits,
                "git_commit": safe_git_commit(),
            },
        )

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
            output_file = scores_dir / f"{trait}.json"

            existing_scores = {}
            if output_file.exists():
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
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
                    assistant_response = resp.get("assistant_response", "")
                    if not assistant_response:
                        assistant_response = next(
                            (m["content"] for m in resp["conversation"] if m["role"] == "assistant"),
                            "",
                        )
                    sample_prompt = eval_prompt_template.format(
                        question=resp["question"],
                        answer=assistant_response,
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

    global_stats = {
        "traits_attempted": 0,
        "traits_successful": 0,
        "traits_skipped": 0,
        "traits_failed": 0,
        "requested_prompts": 0,
        "parsed_successfully": 0,
        "parse_failures": 0,
        "refusals_as_zero": 0,
    }

    for response_file in tqdm(response_files, desc="Scoring traits"):
        trait = response_file.stem
        global_stats["traits_attempted"] += 1

        output_file = scores_dir / f"{trait}.json"
        diagnostic_file = diagnostics_dir / f"{trait}.json"

        existing_scores = {}
        if output_file.exists():
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    existing_scores = json.load(f)
            except Exception:
                pass

        trait_file = traits_dir / f"{trait}.json"
        if not trait_file.exists():
            logger.info(f"Skipping {trait}: no trait file")
            skipped += 1
            global_stats["traits_skipped"] += 1
            continue

        eval_prompt_template = load_trait_eval_prompt(trait_file)
        if not eval_prompt_template:
            logger.info(f"Skipping {trait}: no eval_prompt")
            skipped += 1
            global_stats["traits_skipped"] += 1
            continue

        responses = load_responses(response_file)
        if not responses:
            errors.append(f"{trait}: no responses")
            failed += 1
            global_stats["traits_failed"] += 1
            continue

        all_scored = all(r["label"] in existing_scores for r in responses)
        if all_scored:
            logger.info(f"Skipping {trait}: all {len(responses)} responses already scored")
            skipped += 1
            global_stats["traits_skipped"] += 1
            continue

        try:
            result = await process_trait(
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

            new_scores = result["scores"]
            all_scores = {**existing_scores, **new_scores}

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_scores, f, indent=2, ensure_ascii=False)

            trait_diag = {
                "trait": trait,
                "created_at_utc": utc_now_iso(),
                "judge_model": args.judge_model,
                "response_file": str(response_file.resolve()),
                "trait_file": str(trait_file.resolve()),
                "stats": {
                    "total_responses_in_file": len(responses),
                    "existing_scores_before_run": len(existing_scores),
                    "new_scores_added": len(new_scores),
                    "total_scores_after_run": len(all_scores),
                    **result["stats"],
                },
                "raw_judge_responses": result["raw_judge_responses"],
                "parse_failures": result["parse_failures"],
            }
            write_json(diagnostic_file, trait_diag)

            logger.info(f"Saved {len(all_scores)} scores for {trait} ({len(new_scores)} new)")
            successful += 1
            global_stats["traits_successful"] += 1
            global_stats["requested_prompts"] += result["stats"]["requested"]
            global_stats["parsed_successfully"] += result["stats"]["parsed_successfully"]
            global_stats["parse_failures"] += result["stats"]["parse_failures"]
            global_stats["refusals_as_zero"] += result["stats"]["refusals_as_zero"]

        except Exception as e:
            errors.append(f"{trait}: {e}")
            failed += 1
            global_stats["traits_failed"] += 1

    global_stats["traits_skipped"] = skipped
    write_json(
        manifests_dir / "summary.json",
        {
            "completed_at_utc": utc_now_iso(),
            "summary": global_stats,
            "sample_errors": errors[:20],
        },
    )

    logger.info(f"\nSummary: {successful} successful, {skipped} skipped, {failed} failed")
    if errors:
        for err in errors[:10]:
            logger.info(f"  - {err}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()