#!/usr/bin/env python3
"""
Generate model responses for all traits using vLLM batch inference.

Mirrors scripts/1_generate.py exactly. The only differences:
  - Trait JSONs have "instruction" as a list of {pos, neg} pairs instead of a single string.
  - We iterate over all 10 instructions (5 pos + 5 neg) per trait, calling
    generate_role_responses() once per instruction (treating each as a mini-role).
  - All responses are merged into a single output JSONL per trait, with each
    record labeled "positive" or "negative" and its prompt_index.

Usage:
    uv run trait_pipeline/1_generate.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --traits_dir data/prompts/traits \
        --questions_file data/extraction_questions.jsonl \
        --output_dir outputs/llama-3.1-8b/trait_responses \
        --question_count 50
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import jsonlines
import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.generation import RoleResponseGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_trait(trait_file: Path) -> Optional[dict]:
    """Load a trait JSON file."""
    try:
        with open(trait_file, 'r') as f:
            data = json.load(f)
        if 'instruction' not in data:
            logger.warning(f"No 'instruction' key in {trait_file}")
            return None
        return data
    except Exception as e:
        logger.error(f"Error loading {trait_file}: {e}")
        return None


def generate_and_save_trait(
    generator: RoleResponseGenerator,
    trait_name: str,
    trait_data: dict,
    output_dir: Path,
    skip_existing: bool = True,
) -> bool:
    """
    Generate all responses for one trait and save to a single JSONL.

    For each of the 5 instruction pairs we call generate_role_responses() twice
    (once for pos, once for neg), exactly as the role pipeline calls it for a role.
    All 10 * question_count responses are merged into one file.
    """
    output_file = output_dir / f"{trait_name}.jsonl"
    if skip_existing and output_file.exists():
        logger.info(f"Skipping '{trait_name}' (already exists)")
        return True

    instruction_pairs = trait_data['instruction']  # list of {pos: ..., neg: ...}
    all_responses = []

    for p_idx, pair in enumerate(instruction_pairs):
        for polarity, system_prompt in [('positive', pair['pos']), ('negative', pair['neg'])]:
            # Reuse generate_role_responses by passing a role_data dict with a single 'instruction'.
            # This is exactly how 1_generate.py uses it for roles.
            role_data = {'instruction': system_prompt}
            responses = generator.generate_role_responses(
                f"{trait_name}_{polarity}_{p_idx}", role_data
            )
            if not responses:
                logger.warning(f"No responses for {trait_name} {polarity} p{p_idx}")
                continue

            # Re-label each response with trait metadata
            for resp in responses:
                resp['trait'] = trait_name
                resp['polarity'] = polarity
                resp['prompt_index'] = p_idx
                # question_index and question are already set by generate_role_responses
                resp['label'] = f"{polarity}_p{p_idx}_q{resp['question_index']}"
            all_responses.extend(responses)

    if not all_responses:
        logger.warning(f"No responses generated for trait '{trait_name}'")
        return False

    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(all_responses)

    logger.info(f"Saved {len(all_responses)} responses for '{trait_name}'")
    return True


def process_traits_on_worker(worker_id: int, gpu_ids: List[int], trait_names: List[str], args):
    """Process a subset of traits on a worker. Mirrors process_roles_on_worker."""
    gpu_ids_str = ','.join(map(str, gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str

    worker_logger = logging.getLogger(f"Worker-{worker_id}")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f'%(asctime)s - Worker-{worker_id}[GPUs:{gpu_ids_str}] - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    worker_logger.addHandler(handler)
    worker_logger.setLevel(logging.INFO)

    worker_logger.info(f"Worker {worker_id} starting: GPUs {gpu_ids}, {len(trait_names)} traits")

    try:
        generator = RoleResponseGenerator(
            model_name=args.model,
            roles_dir=args.traits_dir,
            output_dir=args.output_dir,
            questions_file=args.questions_file,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            question_count=args.question_count,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )
        generator.generator.load()

        traits_dir = Path(args.traits_dir)
        output_dir = Path(args.output_dir)
        completed = 0
        failed = 0

        from tqdm import tqdm
        for trait_name in tqdm(trait_names, desc=f"Worker-{worker_id}", position=worker_id):
            trait_file = traits_dir / f"{trait_name}.json"
            trait_data = load_trait(trait_file)
            if trait_data is None:
                failed += 1
                continue
            try:
                success = generate_and_save_trait(generator, trait_name, trait_data, output_dir)
                if success:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                worker_logger.error(f"Exception for {trait_name}: {e}")

        worker_logger.info(f"Worker {worker_id} done: {completed} OK, {failed} failed")

    except Exception as e:
        worker_logger.error(f"Fatal error on Worker {worker_id}: {e}")
        import traceback
        worker_logger.error(traceback.format_exc())


def run_multi_worker(args) -> int:
    """Run multi-worker processing. Mirrors run_multi_worker from 1_generate.py."""
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = [int(x.strip()) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    total_gpus = len(gpu_ids)
    tensor_parallel_size = args.tensor_parallel_size

    if total_gpus == 0:
        logger.error("No GPUs available.")
        return 1
    if tensor_parallel_size > total_gpus:
        logger.error(f"tensor_parallel_size ({tensor_parallel_size}) > available GPUs ({total_gpus})")
        return 1

    num_workers = total_gpus // tensor_parallel_size
    if total_gpus % tensor_parallel_size != 0:
        logger.warning(
            f"GPUs ({total_gpus}) not evenly divisible by tensor_parallel_size ({tensor_parallel_size}). "
            f"Using {num_workers} workers."
        )

    logger.info(f"Available GPUs: {gpu_ids}, tensor_parallel_size: {tensor_parallel_size}, workers: {num_workers}")

    traits_dir = Path(args.traits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trait_names = []
    for f in sorted(traits_dir.glob("*.json")):
        name = f.stem
        if args.traits and name not in args.traits:
            continue
        if (output_dir / f"{name}.jsonl").exists():
            logger.info(f"Skipping '{name}' (already exists)")
            continue
        trait_names.append(name)

    if not trait_names:
        logger.info("No traits to process")
        return 0

    logger.info(f"Processing {len(trait_names)} traits across {num_workers} workers")

    gpu_chunks = [gpu_ids[i*tensor_parallel_size:(i+1)*tensor_parallel_size] for i in range(num_workers)]
    trait_chunks = [[] for _ in range(num_workers)]
    for i, name in enumerate(trait_names):
        trait_chunks[i % num_workers].append(name)
    for i in range(num_workers):
        logger.info(f"Worker {i} (GPUs {gpu_chunks[i]}): {len(trait_chunks[i])} traits")

    mp.set_start_method('spawn', force=True)
    processes = []
    for worker_id in range(num_workers):
        if trait_chunks[worker_id]:
            p = mp.Process(
                target=process_traits_on_worker,
                args=(worker_id, gpu_chunks[worker_id], trait_chunks[worker_id], args)
            )
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    logger.info("Multi-worker processing completed!")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Generate trait responses using vLLM batch inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--traits_dir', type=str, default="../data/extraction_questions/traits")
    parser.add_argument('--questions_file', type=str, default="../data/extraction_questions.jsonl",
                        help='Shared extraction questions (same file used for roles)')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--question_count', type=int, default=50,
                        help='Questions per instruction (default: 50, matching role pipeline)')
    parser.add_argument('--max_model_len', type=int, default=2048)
    parser.add_argument('--tensor_parallel_size', type=int, default=None)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--traits', nargs='+', help='Specific traits to process')

    args = parser.parse_args()

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        available_gpus = [int(x.strip()) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()]
        total_gpus = len(available_gpus)
    else:
        total_gpus = torch.cuda.device_count()

    tensor_parallel_size = args.tensor_parallel_size if args.tensor_parallel_size else max(1, total_gpus)
    args.tensor_parallel_size = tensor_parallel_size

    use_multi_worker = total_gpus > 1 and total_gpus > tensor_parallel_size

    if use_multi_worker:
        logger.info(f"Multi-worker mode: {total_gpus} GPUs, tensor_parallel_size={tensor_parallel_size}")
        exit_code = run_multi_worker(args)
        if exit_code != 0:
            sys.exit(exit_code)
    else:
        logger.info(f"Single-worker mode: {tensor_parallel_size} GPU(s)")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        traits_dir = Path(args.traits_dir)

        generator = RoleResponseGenerator(
            model_name=args.model,
            roles_dir=args.traits_dir,
            output_dir=args.output_dir,
            questions_file=args.questions_file,
            max_model_len=args.max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            question_count=args.question_count,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )
        generator.generator.load()

        trait_names = []
        for f in sorted(traits_dir.glob("*.json")):
            name = f.stem
            if args.traits and name not in args.traits:
                continue
            if (output_dir / f"{name}.jsonl").exists():
                logger.info(f"Skipping '{name}' (already exists)")
                continue
            trait_names.append(name)

        from tqdm import tqdm
        for trait_name in tqdm(trait_names, desc="Processing traits"):
            trait_file = traits_dir / f"{trait_name}.json"
            trait_data = load_trait(trait_file)
            if trait_data is None:
                continue
            try:
                generate_and_save_trait(generator, trait_name, trait_data, output_dir)
            except Exception as e:
                logger.error(f"Exception for {trait_name}: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
