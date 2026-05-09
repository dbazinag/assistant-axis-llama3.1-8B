#!/usr/bin/env python3
# HF transformers activation extractor for traits40 generation outputs.
# Saves selected-layer activations for multiple token positions.
# Uses chunked_sdpa for Gemma 4-31B's head_dim=512.

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunked_sdpa import chunked_sdpa_scope


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


POSITION_NAMES = [
    "pre_generation_last_token",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def traceback_string() -> str:
    import traceback
    return traceback.format_exc()


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_error_log(
    log_path: Path,
    trait_name: str,
    error_text: str,
    traceback_text: Optional[str] = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{utc_now_iso()}] trait={trait_name}\n{error_text}\n")
        if traceback_text:
            f.write(traceback_text)
            if not traceback_text.endswith("\n"):
                f.write("\n")
        f.write("\n")


def normalize_token_ids(x) -> List[int]:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()

    if isinstance(x, dict):
        if "input_ids" not in x:
            raise ValueError(f"Tokenizer output dict missing input_ids. Keys: {list(x.keys())}")
        x = x["input_ids"]
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().tolist()

    if hasattr(x, "data") and isinstance(getattr(x, "data"), dict):
        data = x.data
        if "input_ids" in data:
            x = data["input_ids"]

    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
        x = x[0]

    if not isinstance(x, list):
        raise ValueError(f"Could not normalize tokenizer output of type {type(x)}")

    if x and isinstance(x[0], list):
        x = x[0]

    return [int(t) for t in x]


class HFProbingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._sdpa_cm = None

        self.load()

    def load(self) -> None:
        logger.info(f"Loading tokenizer from {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"

        logger.info(f"Loading model from {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda:0",
            trust_remote_code=True,
        )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._sdpa_cm = chunked_sdpa_scope()
        self._sdpa_cm.__enter__()

        logger.info("Model loaded.")

    def unload(self) -> None:
        if self._sdpa_cm is not None:
            self._sdpa_cm.__exit__(None, None, None)
            self._sdpa_cm = None

    def get_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h

        n_layers = getattr(self.model.config, "num_hidden_layers", None)
        if n_layers is None:
            raise ValueError("Could not infer number of hidden layers")
        return list(range(n_layers))


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


def find_subsequence(seq: List[int], sub: List[int], start_at: int = 0) -> int:
    if not sub:
        return -1
    n, m = len(seq), len(sub)
    start_at = max(0, min(start_at, n))
    for i in range(start_at, n - m + 1):
        if seq[i:i + m] == sub:
            return i
    return -1


def get_answer_span_from_row(tokenizer, row: dict, full_token_ids: Optional[List[int]] = None) -> Optional[List[int]]:
    meta = row["chat_template_metadata"]
    answer = row.get("assistant_response", "")

    if not answer:
        return None

    answer_token_ids = normalize_token_ids(tokenizer.encode(answer, add_special_tokens=False))

    if len(answer_token_ids) == 0:
        return None

    expected_start = int(meta["prompt_token_count"])
    expected_end = expected_start + len(answer_token_ids) - 1

    if full_token_ids is None:
        return list(range(expected_start, expected_end + 1))

    # First try metadata-based span.
    if expected_start >= 0 and expected_end < len(full_token_ids):
        return list(range(expected_start, expected_end + 1))

    # Fallback: locate answer token ids in the full chat tokenization.
    found = find_subsequence(full_token_ids, answer_token_ids, start_at=max(0, expected_start - 8))
    if found == -1:
        found = find_subsequence(full_token_ids, answer_token_ids, start_at=0)

    if found == -1:
        return None

    return list(range(found, found + len(answer_token_ids)))


def extract_positions_for_batch(
    pm: HFProbingModel,
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

    # Keep unpadded token ids for robust answer-span fallback.
    unpadded_input_ids = []
    for i in range(encoded["input_ids"].shape[0]):
        valid = encoded["attention_mask"][i].bool()
        unpadded_input_ids.append(encoded["input_ids"][i][valid].detach().cpu().tolist())

    device = get_model_device(model)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states

    selected_hidden_states = []
    for layer_idx in layers:
        if layer_idx < 0:
            raise ValueError(f"Layer index must be non-negative, got {layer_idx}")
        hs_idx = layer_idx + 1
        if hs_idx >= len(hidden_states):
            raise ValueError(
                f"Layer index {layer_idx} out of range. "
                f"Model returned {len(hidden_states) - 1} hidden layers."
            )
        selected_hidden_states.append(hidden_states[hs_idx])

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
        answer_indices = get_answer_span_from_row(
            tokenizer,
            row,
            full_token_ids=unpadded_input_ids[batch_idx],
        )

        def shift_and_validate(indices: Optional[List[int]]) -> Optional[List[int]]:
            if indices is None or len(indices) == 0:
                return None

            shifted = [int(idx) + offset for idx in indices]

            for idx in shifted:
                if idx < offset or idx >= offset + valid_len:
                    return None
                if idx >= encoded["input_ids"].shape[1]:
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
    pm: HFProbingModel,
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
        batch_rows = rows[batch_start: batch_start + batch_size]

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


def parse_layers(layers_arg: str, n_layers: int) -> List[int]:
    if layers_arg == "all":
        return list(range(n_layers))

    layers = [int(x.strip()) for x in layers_arg.split(",") if x.strip()]

    for layer in layers:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} out of range for model with {n_layers} layers")

    return layers


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

    pm = None

    try:
        pm = HFProbingModel(args.model)
        pm.model.eval()

        n_layers = len(pm.get_layers())
        layers = parse_layers(args.layers, n_layers)

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

    finally:
        if pm is not None:
            pm.unload()


def collect_response_files(args) -> List[Path]:
    responses_dir = Path(args.responses_dir)
    output_root = Path(args.output_root)

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

    return response_files


def run_multi_worker(args) -> int:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_ids = [int(x.strip()) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    total_gpus = len(gpu_ids)

    # This HF extractor does not do tensor parallelism. We use one process per GPU.
    tensor_parallel_size = 1

    if total_gpus == 0:
        logger.error("No GPUs available.")
        return 1

    num_workers = total_gpus // tensor_parallel_size

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    response_files = collect_response_files(args)

    if not response_files:
        logger.info("No files to process")
        return 0

    gpu_chunks = [[gpu_id] for gpu_id in gpu_ids[:num_workers]]

    file_chunks = [[] for _ in range(num_workers)]
    for i, f in enumerate(response_files):
        file_chunks[i % num_workers].append(f)

    logger.info(f"Available GPUs: {gpu_ids}")
    logger.info(f"HF mode: one worker per GPU")
    logger.info(f"num_workers: {num_workers}")

    for i in range(num_workers):
        logger.info(f"Worker {i} -> GPU {gpu_chunks[i]} -> {len(file_chunks[i])} files")

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
        "backend": "huggingface_transformers",
    }


def main():
    parser = argparse.ArgumentParser(description="Extract selected-layer activations from traits40 responses")
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
    parser.add_argument("--batch_size", type=int, default=4)
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

    if total_gpus > 1:
        exit_code = run_multi_worker(args)
        if exit_code != 0:
            sys.exit(exit_code)
    else:
        response_files = collect_response_files(args)

        if not response_files:
            logger.info("No files to process")
            logger.info("Done!")
            return

        error_log = output_root / "logs" / "single_worker_errors.log"

        pm = None

        try:
            pm = HFProbingModel(args.model)
            pm.model.eval()

            n_layers = len(pm.get_layers())
            layers = parse_layers(args.layers, n_layers)

            for response_file in tqdm(response_files, desc="Processing"):
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

        finally:
            if pm is not None:
                pm.unload()

    logger.info("Done!")


if __name__ == "__main__":
    main()