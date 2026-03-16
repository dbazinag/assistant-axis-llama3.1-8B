#!/usr/bin/env python3
# Interview note: extracts all-layer activations for multiple token positions from the new traits40 generation outputs.

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jsonlines
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.internals import ProbingModel


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


POSITION_NAMES = [
    "user_last_token",
    "pre_generation_last_token",
    "assistant_header_mean",
    "assistant_header_span",
    "answer_mean",
    "user_mean",
]


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


def write_error_log(log_path: Path, trait_name: str, error_text: str, traceback_text: Optional[str] = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{utc_now_iso()}] trait={trait_name}\n{error_text}\n")
        if traceback_text:
            f.write(traceback_text)
            if not traceback_text.endswith("\n"):
                f.write("\n")
        f.write("\n")


def traceback_string() -> str:
    import traceback
    return traceback.format_exc()


def load_responses(responses_file: Path) -> List[dict]:
    rows = []
    with jsonlines.open(responses_file, "r") as reader:
        for entry in reader:
            rows.append(entry)
    return rows


def ensure_tokenizer_ready(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_chat_text(tokenizer, conversation: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
    )


def get_valid_offset(tokenizer, attention_mask_row: torch.Tensor) -> int:
    valid_len = int(attention_mask_row.sum().item())
    total_len = int(attention_mask_row.shape[0])
    if tokenizer.padding_side == "left":
        return total_len - valid_len
    return 0


def get_answer_span_from_row(tokenizer, row: dict) -> Optional[List[int]]:
    meta = row["chat_template_metadata"]
    answer = row.get("assistant_response", "")
    answer_token_ids = tokenizer.encode(answer, add_special_tokens=False)

    if len(answer_token_ids) == 0:
        return None

    start = int(meta["prompt_token_count"])
    end = start + len(answer_token_ids) - 1
    return list(range(start, end + 1))


def extract_positions_for_batch(
    pm: ProbingModel,
    batch_rows: List[dict],
    layers: List[int],
    max_length: int,
) -> Dict[str, List[Optional[torch.Tensor]]]:
    tokenizer = pm.tokenizer
    model = pm.model

    ensure_tokenizer_ready(tokenizer)

    texts = [build_chat_text(tokenizer, row["conversation"]) for row in batch_rows]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    device = get_model_device(model)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
    selected_hidden_states = [hidden_states[layer_idx + 1] for layer_idx in layers]

    batch_outputs: Dict[str, List[Optional[torch.Tensor]]] = {
        name: [] for name in POSITION_NAMES
    }

    for batch_idx, row in enumerate(batch_rows):
        attn_row = encoded["attention_mask"][batch_idx]
        valid_len = int(attn_row.sum().item())
        offset = get_valid_offset(tokenizer, attn_row)

        meta = row["chat_template_metadata"]

        user_last_idx = int(meta["user_last_token_index"])
        pregen_last_idx = int(meta["full_prompt_last_token_index"])
        assistant_header_indices = list(meta["assistant_header_token_indices"])
        user_indices = list(meta["user_content_token_indices"])
        answer_indices = get_answer_span_from_row(tokenizer, row)

        def shift_and_validate(indices: Optional[List[int]]) -> Optional[List[int]]:
            if indices is None or len(indices) == 0:
                return None
            shifted = [idx + offset for idx in indices]
            for idx in shifted:
                if idx < 0 or idx >= offset + valid_len:
                    return None
            return shifted

        user_last_shifted = shift_and_validate([user_last_idx])
        pregen_last_shifted = shift_and_validate([pregen_last_idx])
        assistant_header_shifted = shift_and_validate(assistant_header_indices)
        user_shifted = shift_and_validate(user_indices)
        answer_shifted = shift_and_validate(answer_indices)

        if user_last_shifted is None:
            batch_outputs["user_last_token"].append(None)
        else:
            idx = user_last_shifted[0]
            stacked = torch.stack(
                [hs[batch_idx, idx, :].detach().cpu() for hs in selected_hidden_states],
                dim=0,
            )
            batch_outputs["user_last_token"].append(stacked)

        if pregen_last_shifted is None:
            batch_outputs["pre_generation_last_token"].append(None)
        else:
            idx = pregen_last_shifted[0]
            stacked = torch.stack(
                [hs[batch_idx, idx, :].detach().cpu() for hs in selected_hidden_states],
                dim=0,
            )
            batch_outputs["pre_generation_last_token"].append(stacked)

        if assistant_header_shifted is None:
            batch_outputs["assistant_header_mean"].append(None)
            batch_outputs["assistant_header_span"].append(None)
        else:
            span = assistant_header_shifted
            mean_stacked = torch.stack(
                [hs[batch_idx, span, :].mean(dim=0).detach().cpu() for hs in selected_hidden_states],
                dim=0,
            )
            span_stacked = torch.stack(
                [hs[batch_idx, span, :].detach().cpu() for hs in selected_hidden_states],
                dim=0,
            )
            batch_outputs["assistant_header_mean"].append(mean_stacked)
            batch_outputs["assistant_header_span"].append(span_stacked)

        if answer_shifted is None:
            batch_outputs["answer_mean"].append(None)
        else:
            span = answer_shifted
            mean_stacked = torch.stack(
                [hs[batch_idx, span, :].mean(dim=0).detach().cpu() for hs in selected_hidden_states],
                dim=0,
            )
            batch_outputs["answer_mean"].append(mean_stacked)

        if user_shifted is None:
            batch_outputs["user_mean"].append(None)
        else:
            span = user_shifted
            mean_stacked = torch.stack(
                [hs[batch_idx, span, :].mean(dim=0).detach().cpu() for hs in selected_hidden_states],
                dim=0,
            )
            batch_outputs["user_mean"].append(mean_stacked)

    del outputs
    del hidden_states
    del selected_hidden_states
    del encoded
    torch.cuda.empty_cache()

    return batch_outputs


def process_trait_file(
    pm: ProbingModel,
    responses_file: Path,
    output_root: Path,
    layers: List[int],
    batch_size: int,
    max_length: int,
) -> bool:
    trait_name = responses_file.stem
    rows = load_responses(responses_file)

    if not rows:
        return False

    all_outputs: Dict[str, Dict[str, torch.Tensor]] = {name: {} for name in POSITION_NAMES}

    logger.info(f"Processing {trait_name}: {len(rows)} conversations")

    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        batch_results = extract_positions_for_batch(
            pm=pm,
            batch_rows=batch_rows,
            layers=layers,
            max_length=max_length,
        )

        for position_name in POSITION_NAMES:
            for row, tensor_value in zip(batch_rows, batch_results[position_name]):
                if tensor_value is not None:
                    all_outputs[position_name][row["label"]] = tensor_value

        if (batch_start // batch_size) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    for position_name in POSITION_NAMES:
        position_dir = output_root / position_name
        position_dir.mkdir(parents=True, exist_ok=True)
        output_file = position_dir / f"{trait_name}.pt"
        torch.save(all_outputs[position_name], output_file)
        logger.info(f"Saved {len(all_outputs[position_name])} entries to {output_file}")

    gc.collect()
    torch.cuda.empty_cache()
    return True


def process_traits_on_worker(worker_id: int, gpu_ids: List[int], response_files: List[Path], args) -> None:
    gpu_ids_str = ",".join(map(str, gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

    worker_logger = logging.getLogger(f"Worker-{worker_id}")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f"%(asctime)s - Worker-{worker_id}[GPUs:{gpu_ids_str}] - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    worker_logger.handlers = []
    worker_logger.addHandler(handler)
    worker_logger.setLevel(logging.INFO)

    logs_dir = Path(args.output_root) / "logs"
    error_log = logs_dir / f"worker_{worker_id}_errors.log"

    try:
        pm = ProbingModel(args.model)
        pm.model.eval()

        n_layers = len(pm.get_layers())
        if args.layers == "all":
            layers = list(range(n_layers))
        else:
            layers = [int(x.strip()) for x in args.layers.split(",")]

        completed = 0
        failed = 0

        for response_file in tqdm(response_files, desc=f"Worker-{worker_id}", position=worker_id):
            try:
                success = process_trait_file(
                    pm=pm,
                    responses_file=response_file,
                    output_root=Path(args.output_root),
                    layers=layers,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )
                if success:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                error_text = f"{type(e).__name__}: {str(e)}"
                write_error_log(error_log, response_file.stem, error_text, traceback_string())
                worker_logger.error(f"Exception processing {response_file.stem}: {error_text}")

        worker_logger.info(f"Worker {worker_id} done: {completed} OK, {failed} failed")

    except Exception as e:
        error_text = f"Fatal worker error: {type(e).__name__}: {str(e)}"
        write_error_log(error_log, "__worker__", error_text, traceback_string())
        worker_logger.error(error_text)


def run_multi_worker(args) -> int:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_ids = [int(x.strip()) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    total_gpus = len(gpu_ids)
    tensor_parallel_size = args.tensor_parallel_size

    if total_gpus == 0:
        logger.error("No GPUs available.")
        return 1
    if tensor_parallel_size > total_gpus:
        logger.error(
            f"tensor_parallel_size ({tensor_parallel_size}) > available GPUs ({total_gpus})"
        )
        return 1

    num_workers = total_gpus // tensor_parallel_size

    responses_dir = Path(args.responses_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    response_files = []
    for f in sorted(responses_dir.glob("*.jsonl")):
        if args.traits and f.stem not in args.traits:
            continue

        already_done = True
        for position_name in POSITION_NAMES:
            if not (output_root / position_name / f"{f.stem}.pt").exists():
                already_done = False
                break

        if already_done:
            logger.info(f"Skipping {f.stem} (all position outputs already exist)")
            continue

        response_files.append(f)

    if not response_files:
        logger.info("No files to process")
        return 0

    gpu_chunks = [
        gpu_ids[i * tensor_parallel_size : (i + 1) * tensor_parallel_size]
        for i in range(num_workers)
    ]
    file_chunks = [[] for _ in range(num_workers)]
    for i, f in enumerate(response_files):
        file_chunks[i % num_workers].append(f)

    mp.set_start_method("spawn", force=True)
    processes = []
    for worker_id in range(num_workers):
        if file_chunks[worker_id]:
            p = mp.Process(
                target=process_traits_on_worker,
                args=(worker_id, gpu_chunks[worker_id], file_chunks[worker_id], args),
            )
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    return 0


def build_manifest(args) -> Dict:
    return {
        "created_at_utc": utc_now_iso(),
        "model": args.model,
        "responses_dir": str(Path(args.responses_dir).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "positions_saved": POSITION_NAMES,
        "layers": args.layers,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "git_commit": safe_git_commit(),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract all-layer activations from traits40 responses")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--responses_dir",
        type=str,
        default="full_trait_output/traits40_generation/responses",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="full_trait_output/traits40_activations",
    )
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--traits", nargs="+", help="Specific traits to process")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for position_name in POSITION_NAMES:
        (output_root / position_name).mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    write_json(output_root / "manifests" / "run_config.json", build_manifest(args))

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        available_gpus = [
            int(x.strip())
            for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            if x.strip()
        ]
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
        responses_dir = Path(args.responses_dir)

        pm = ProbingModel(args.model)
        pm.model.eval()

        n_layers = len(pm.get_layers())
        if args.layers == "all":
            layers = list(range(n_layers))
        else:
            layers = [int(x.strip()) for x in args.layers.split(",")]

        response_files = sorted(responses_dir.glob("*.jsonl"))
        if args.traits:
            response_files = [f for f in response_files if f.stem in args.traits]

        files_to_process = []
        for f in response_files:
            already_done = True
            for position_name in POSITION_NAMES:
                if not (output_root / position_name / f"{f.stem}.pt").exists():
                    already_done = False
                    break

            if already_done:
                logger.info(f"Skipping {f.stem} (all position outputs already exist)")
                continue

            files_to_process.append(f)

        error_log = output_root / "logs" / "single_worker_errors.log"

        for response_file in tqdm(files_to_process, desc="Processing"):
            try:
                process_trait_file(
                    pm=pm,
                    responses_file=response_file,
                    output_root=output_root,
                    layers=layers,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )
            except Exception as e:
                error_text = f"{type(e).__name__}: {str(e)}"
                write_error_log(error_log, response_file.stem, error_text, traceback_string())
                logger.error(f"Exception processing {response_file.stem}: {error_text}")

    logger.info("Done!")


if __name__ == "__main__":
    main()