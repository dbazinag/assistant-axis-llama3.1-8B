#!/usr/bin/env python3
"""
Extract activations from trait response JSONL files.

Identical to scripts/2_activations.py. No changes needed — this script reads
JSONL response files and extracts mean activations; it is agnostic to whether
the responses came from roles or traits.

The only difference from the role version is the default --responses_dir and
--output_dir paths.

Usage:
    uv run trait_pipeline/2_activations.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --responses_dir outputs/llama-3.1-8b/trait_responses \
        --output_dir outputs/llama-3.1-8b/trait_activations
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor, SpanMapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_responses(responses_file: Path) -> List[dict]:
    responses = []
    with jsonlines.open(responses_file, 'r') as reader:
        for entry in reader:
            responses.append(entry)
    return responses


def extract_activations_batch(
    pm: ProbingModel,
    conversations: List[List[Dict[str, str]]],
    layers: List[int],
    batch_size: int = 16,
    max_length: int = 2048,
    enable_thinking: bool = False,
) -> List[Optional[torch.Tensor]]:
    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
    extractor = ActivationExtractor(pm, encoder)
    span_mapper = SpanMapper(pm.tokenizer)

    chat_kwargs = {}
    if 'qwen' in pm.model_name.lower():
        chat_kwargs['enable_thinking'] = enable_thinking

    print(f"DEBUG: chat_kwargs = {chat_kwargs}")

    all_activations = []
    num_conversations = len(conversations)

    for batch_start in range(0, num_conversations, batch_size):
        batch_end = min(batch_start + batch_size, num_conversations)
        batch_conversations = conversations[batch_start:batch_end]

        batch_activations, batch_metadata = extractor.batch_conversations(
            batch_conversations,
            layer=layers,
            max_length=max_length,
            **chat_kwargs,
        )

        _, batch_spans, span_metadata = encoder.build_batch_turn_spans(batch_conversations, **chat_kwargs)

        if batch_start == 0:
            for span in batch_spans[:4]:
                if span['role'] == 'assistant':
                    print(f"  DEBUG span: conv={span['conversation_id']} start={span['start']} end={span['end']} n_tokens={span['n_tokens']}")

        conv_activations_list = span_mapper.map_spans(batch_activations, batch_spans, batch_metadata)

        for conv_acts in conv_activations_list:
            if conv_acts.numel() == 0:
                all_activations.append(None)
                continue

            if conv_acts.shape[0] >= 2:
                assistant_act = conv_acts[1::2]
                if assistant_act.shape[0] > 0:
                    mean_act = assistant_act.mean(dim=0).cpu()
                    all_activations.append(mean_act)
                else:
                    all_activations.append(None)
            else:
                all_activations.append(None)

        del batch_activations
        if (batch_start // batch_size) % 5 == 0:
            torch.cuda.empty_cache()

    return all_activations


def process_role(pm, role_file, output_dir, layers, batch_size, max_length, enable_thinking=False):
    role = role_file.stem
    output_file = output_dir / f"{role}.pt"

    responses = load_responses(role_file)
    if not responses:
        return False

    conversations = []
    metadata = []
    for resp in responses:
        conversations.append(resp["conversation"])
        metadata.append({
            "prompt_index": resp["prompt_index"],
            "question_index": resp["question_index"],
            "label": resp["label"],
        })

    logger.info(f"Processing {role}: {len(conversations)} conversations")

    activations_list = extract_activations_batch(
        pm=pm,
        conversations=conversations,
        layers=layers,
        batch_size=batch_size,
        max_length=max_length,
        enable_thinking=enable_thinking,
    )

    activations_dict = {}
    for i, (act, meta) in enumerate(zip(activations_list, metadata)):
        if act is not None:
            key = f"{meta['label']}"
            activations_dict[key] = act

    if activations_dict:
        torch.save(activations_dict, output_file)
        logger.info(f"Saved {len(activations_dict)} activations for {role}")

    gc.collect()
    torch.cuda.empty_cache()
    return True


def process_roles_on_worker(worker_id, gpu_ids, role_files, args):
    gpu_ids_str = ','.join(map(str, gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str

    worker_logger = logging.getLogger(f"Worker-{worker_id}")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f'%(asctime)s - Worker-{worker_id}[GPUs:{gpu_ids_str}] - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    worker_logger.addHandler(handler)
    worker_logger.setLevel(logging.INFO)

    output_dir = Path(args.output_dir)

    try:
        pm = ProbingModel(args.model)

        n_layers = len(pm.get_layers())
        if args.layers == "all":
            layers = list(range(n_layers))
        else:
            layers = [int(x.strip()) for x in args.layers.split(",")]

        completed = 0
        failed = 0

        for role_file in tqdm(role_files, desc=f"Worker-{worker_id}", position=worker_id):
            try:
                success = process_role(pm, role_file, output_dir, layers, args.batch_size, args.max_length, args.thinking)
                if success:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                worker_logger.error(f"Exception processing {role_file.stem}: {e}")

        worker_logger.info(f"Worker {worker_id} done: {completed} OK, {failed} failed")

    except Exception as e:
        worker_logger.error(f"Fatal error on Worker {worker_id}: {e}")


def run_multi_worker(args):
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

    responses_dir = Path(args.responses_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    role_files = []
    for f in sorted(responses_dir.glob("*.jsonl")):
        if args.roles and f.stem not in args.roles:
            continue
        output_file = output_dir / f"{f.stem}.pt"
        if output_file.exists():
            logger.info(f"Skipping {f.stem} (already exists)")
            continue
        role_files.append(f)

    if not role_files:
        logger.info("No files to process")
        return 0

    gpu_chunks = [gpu_ids[i*tensor_parallel_size:(i+1)*tensor_parallel_size] for i in range(num_workers)]
    role_chunks = [[] for _ in range(num_workers)]
    for i, f in enumerate(role_files):
        role_chunks[i % num_workers].append(f)

    mp.set_start_method('spawn', force=True)
    processes = []
    for worker_id in range(num_workers):
        if role_chunks[worker_id]:
            p = mp.Process(target=process_roles_on_worker, args=(worker_id, gpu_chunks[worker_id], role_chunks[worker_id], args))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    return 0


def main():
    parser = argparse.ArgumentParser(description="Extract activations from trait responses")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--responses_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--roles", nargs="+", help="Specific traits to process")
    parser.add_argument("--thinking", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False)
    args = parser.parse_args()

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        available_gpus = [int(x.strip()) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()]
        total_gpus = len(available_gpus)
    else:
        total_gpus = torch.cuda.device_count()

    tensor_parallel_size = args.tensor_parallel_size if args.tensor_parallel_size else total_gpus
    use_multi_worker = total_gpus > 1 and tensor_parallel_size > 0 and total_gpus > tensor_parallel_size

    if use_multi_worker:
        args.tensor_parallel_size = tensor_parallel_size
        exit_code = run_multi_worker(args)
        if exit_code != 0:
            sys.exit(exit_code)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        responses_dir = Path(args.responses_dir)

        pm = ProbingModel(args.model)
        n_layers = len(pm.get_layers())

        if args.layers == "all":
            layers = list(range(n_layers))
        else:
            layers = [int(x.strip()) for x in args.layers.split(",")]

        response_files = sorted(responses_dir.glob("*.jsonl"))
        if args.roles:
            response_files = [f for f in response_files if f.stem in args.roles]

        role_files = []
        for f in response_files:
            output_file = output_dir / f"{f.stem}.pt"
            if output_file.exists():
                logger.info(f"Skipping {f.stem} (already exists)")
                continue
            role_files.append(f)

        for role_file in tqdm(role_files, desc="Processing"):
            process_role(pm, role_file, output_dir, layers, args.batch_size, args.max_length, args.thinking)

    logger.info("Done!")


if __name__ == "__main__":
    main()
