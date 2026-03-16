#!/usr/bin/env python3
# Interview note: generates trait responses using only the 40 questions inside each trait JSON, saves rich chat-template metadata for later activation extraction.

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonlines
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------
# data structures
# ----------------------------

@dataclass
class PromptRecord:
    trait: str
    trait_file: str
    polarity: str
    prompt_index: int
    question_index: int
    label: str
    system_prompt: str
    instruction_text: str
    question: str
    messages: List[Dict[str, str]]
    prompt_token_count: int
    full_prompt_last_token_index: int
    assistant_header_token_indices: List[int]
    assistant_header_token_start: int
    assistant_header_token_end: int
    user_content_token_indices: List[int]
    user_content_token_start: int
    user_content_token_end: int
    user_last_token_index: int
    full_prompt_token_ids: List[int]


# ----------------------------
# helpers
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def load_trait(trait_file: Path) -> Dict[str, Any]:
    with open(trait_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "instruction" not in data or "questions" not in data or "eval_prompt" not in data:
        raise ValueError(f"{trait_file} missing one of: instruction, questions, eval_prompt")

    instructions = data["instruction"]
    questions = data["questions"]

    if not isinstance(instructions, list) or len(instructions) != 5:
        raise ValueError(f"{trait_file} must contain exactly 5 instruction pairs")
    if not isinstance(questions, list) or len(questions) != 40:
        raise ValueError(f"{trait_file} must contain exactly 40 questions")

    for i, pair in enumerate(instructions):
        if not isinstance(pair, dict) or "pos" not in pair or "neg" not in pair:
            raise ValueError(f"{trait_file} instruction[{i}] must contain pos and neg")

    return data


def build_messages(system_prompt: str, question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def apply_chat_template_string(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def apply_chat_template_tokens(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> List[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return token_ids


def find_subsequence(sequence: List[int], subsequence: List[int]) -> int:
    if not subsequence:
        return -1
    n, m = len(sequence), len(subsequence)
    for i in range(n - m + 1):
        if sequence[i : i + m] == subsequence:
            return i
    return -1


def build_span_metadata(
    tokenizer: AutoTokenizer,
    full_prompt_token_ids: List[int],
    system_prompt: str,
    question: str,
) -> Tuple[List[int], int, int, List[int], int, int, int]:
    empty_messages = []
    user_only_messages = [{"role": "user", "content": question}]

    full_without_system = apply_chat_template_tokens(tokenizer, user_only_messages)
    full_empty = apply_chat_template_tokens(tokenizer, empty_messages)

    if len(full_prompt_token_ids) <= 0:
        raise ValueError("Prompt token ids are empty")

    # assistant header span = suffix added by add_generation_prompt=True after the last real message
    # find by taking full prompt and subtracting the same template without generation prompt is hard,
    # so we instead locate the final assistant header by tokenizing the messages without generation prompt
    prompt_no_gen = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(prompt_no_gen, torch.Tensor):
        prompt_no_gen = prompt_no_gen.tolist()

    if len(prompt_no_gen) >= len(full_prompt_token_ids):
        raise ValueError("Expected generation prompt to add assistant header tokens")

    assistant_header_start = len(prompt_no_gen)
    assistant_header_end = len(full_prompt_token_ids) - 1
    assistant_header_token_indices = list(range(assistant_header_start, assistant_header_end + 1))

    # locate user-content tokens by searching for tokenized question content
    question_token_ids = tokenizer.encode(question, add_special_tokens=False)
    user_content_start = find_subsequence(full_prompt_token_ids, question_token_ids)
    if user_content_start == -1:
        raise ValueError("Could not locate user-content tokens in prompt token ids")
    user_content_end = user_content_start + len(question_token_ids) - 1
    user_content_token_indices = list(range(user_content_start, user_content_end + 1))

    user_last_token_index = user_content_end

    return (
        assistant_header_token_indices,
        assistant_header_start,
        assistant_header_end,
        user_content_token_indices,
        user_content_start,
        user_content_end,
        user_last_token_index,
    )


def build_prompt_records_for_trait(
    tokenizer: AutoTokenizer,
    trait_name: str,
    trait_file: Path,
    trait_data: Dict[str, Any],
) -> List[PromptRecord]:
    records: List[PromptRecord] = []

    for prompt_index, pair in enumerate(trait_data["instruction"]):
        for polarity, system_prompt in [("positive", pair["pos"]), ("negative", pair["neg"])]:
            for question_index, question in enumerate(trait_data["questions"]):
                messages = build_messages(system_prompt=system_prompt, question=question)
                prompt_text = apply_chat_template_string(tokenizer, messages)
                full_prompt_token_ids = apply_chat_template_tokens(tokenizer, messages)

                (
                    assistant_header_token_indices,
                    assistant_header_start,
                    assistant_header_end,
                    user_content_token_indices,
                    user_content_start,
                    user_content_end,
                    user_last_token_index,
                ) = build_span_metadata(
                    tokenizer=tokenizer,
                    full_prompt_token_ids=full_prompt_token_ids,
                    system_prompt=system_prompt,
                    question=question,
                )

                records.append(
                    PromptRecord(
                        trait=trait_name,
                        trait_file=str(trait_file),
                        polarity=polarity,
                        prompt_index=prompt_index,
                        question_index=question_index,
                        label=f"{polarity}_p{prompt_index}_q{question_index}",
                        system_prompt=system_prompt,
                        instruction_text=system_prompt,
                        question=question,
                        messages=messages,
                        prompt_token_count=len(full_prompt_token_ids),
                        full_prompt_last_token_index=len(full_prompt_token_ids) - 1,
                        assistant_header_token_indices=assistant_header_token_indices,
                        assistant_header_token_start=assistant_header_start,
                        assistant_header_token_end=assistant_header_end,
                        user_content_token_indices=user_content_token_indices,
                        user_content_token_start=user_content_start,
                        user_content_token_end=user_content_end,
                        user_last_token_index=user_last_token_index,
                        full_prompt_token_ids=full_prompt_token_ids,
                    )
                )

    return records


def record_to_output_row(record: PromptRecord, assistant_response: str, model_name: str, generation_params: Dict[str, Any]) -> Dict[str, Any]:
    conversation = [
        {"role": "system", "content": record.system_prompt},
        {"role": "user", "content": record.question},
        {"role": "assistant", "content": assistant_response},
    ]

    return {
        "trait": record.trait,
        "trait_file": record.trait_file,
        "polarity": record.polarity,
        "prompt_index": record.prompt_index,
        "question_index": record.question_index,
        "label": record.label,
        "system_prompt": record.system_prompt,
        "instruction_text": record.instruction_text,
        "question": record.question,
        "conversation": conversation,
        "assistant_response": assistant_response,
        "model": model_name,
        "generation_params": generation_params,
        "chat_template_metadata": {
            "prompt_token_count": record.prompt_token_count,
            "full_prompt_last_token_index": record.full_prompt_last_token_index,
            "assistant_header_token_indices": record.assistant_header_token_indices,
            "assistant_header_token_start": record.assistant_header_token_start,
            "assistant_header_token_end": record.assistant_header_token_end,
            "user_content_token_indices": record.user_content_token_indices,
            "user_content_token_start": record.user_content_token_start,
            "user_content_token_end": record.user_content_token_end,
            "user_last_token_index": record.user_last_token_index,
        },
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_error_log(log_path: Path, trait_name: str, error_text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{utc_now_iso()}] trait={trait_name}\n{error_text}\n\n")


def verify_trait_output_file(output_file: Path, trait_data: Dict[str, Any]) -> Tuple[bool, str]:
    rows = []
    with jsonlines.open(output_file, "r") as reader:
        for row in reader:
            rows.append(row)

    expected_count = 10 * 40
    if len(rows) != expected_count:
        return False, f"Expected {expected_count} rows, found {len(rows)}"

    seen = set()
    for row in rows:
        key = (row["polarity"], row["prompt_index"], row["question_index"])
        if key in seen:
            return False, f"Duplicate key found: {key}"
        seen.add(key)

        qidx = row["question_index"]
        expected_question = trait_data["questions"][qidx]
        if row["question"] != expected_question:
            return False, f"Question mismatch at qidx={qidx}"

        pidx = row["prompt_index"]
        expected_prompt = trait_data["instruction"][pidx]["pos"] if row["polarity"] == "positive" else trait_data["instruction"][pidx]["neg"]
        if row["system_prompt"] != expected_prompt:
            return False, f"System prompt mismatch at {row['polarity']} pidx={pidx}"

    for pidx in range(5):
        for polarity in ("positive", "negative"):
            for qidx in range(40):
                if (polarity, pidx, qidx) not in seen:
                    return False, f"Missing key: {(polarity, pidx, qidx)}"

    return True, "ok"


# ----------------------------
# generation core
# ----------------------------

class TraitResponseGenerator:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        self.tokenizer: Optional[AutoTokenizer] = None
        self.llm: Optional[LLM] = None
        self.sampling_params: Optional[SamplingParams] = None

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=1,
        )

    def generate_trait_rows(self, trait_name: str, trait_file: Path, trait_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.tokenizer is None or self.llm is None or self.sampling_params is None:
            raise RuntimeError("Generator not loaded")

        records = build_prompt_records_for_trait(
            tokenizer=self.tokenizer,
            trait_name=trait_name,
            trait_file=trait_file,
            trait_data=trait_data,
        )

        prompt_texts = [
            apply_chat_template_string(self.tokenizer, record.messages)
            for record in records
        ]

        outputs = self.llm.generate(prompt_texts, sampling_params=self.sampling_params)

        generation_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "n": 1,
        }

        rows: List[Dict[str, Any]] = []
        for record, output in zip(records, outputs):
            assistant_response = output.outputs[0].text if output.outputs else ""
            rows.append(
                record_to_output_row(
                    record=record,
                    assistant_response=assistant_response,
                    model_name=self.model_name,
                    generation_params=generation_params,
                )
            )
        return rows


# ----------------------------
# worker logic
# ----------------------------

def process_traits_on_worker(worker_id: int, gpu_ids: List[int], trait_names: List[str], args) -> None:
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

    worker_logger.info(f"Starting worker {worker_id} on GPUs {gpu_ids_str} for {len(trait_names)} traits")

    responses_dir = Path(args.output_root) / "responses"
    verification_dir = Path(args.output_root) / "verification"
    logs_dir = Path(args.output_root) / "logs"
    errors_log = logs_dir / f"worker_{worker_id}_errors.log"

    try:
        generator = TraitResponseGenerator(
            model_name=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        generator.load()

        traits_dir = Path(args.traits_dir)
        completed = 0
        failed = 0

        for trait_name in tqdm(trait_names, desc=f"Worker-{worker_id}", position=worker_id):
            output_file = responses_dir / f"{trait_name}.jsonl"
            verification_file = verification_dir / f"{trait_name}.json"

            if output_file.exists():
                worker_logger.info(f"Skipping '{trait_name}' (already exists)")
                completed += 1
                continue

            trait_file = traits_dir / f"{trait_name}.json"

            try:
                trait_data = load_trait(trait_file)
                rows = generator.generate_trait_rows(
                    trait_name=trait_name,
                    trait_file=trait_file,
                    trait_data=trait_data,
                )

                output_file.parent.mkdir(parents=True, exist_ok=True)
                with jsonlines.open(output_file, "w") as writer:
                    writer.write_all(rows)

                ok, msg = verify_trait_output_file(output_file, trait_data)
                write_json(
                    verification_file,
                    {
                        "trait": trait_name,
                        "verified_at": utc_now_iso(),
                        "ok": ok,
                        "message": msg,
                        "row_count": len(rows),
                        "expected_row_count": 400,
                    },
                )

                if not ok:
                    raise RuntimeError(f"Verification failed for {trait_name}: {msg}")

                worker_logger.info(f"Saved and verified {len(rows)} responses for '{trait_name}'")
                completed += 1

            except Exception as e:
                failed += 1
                error_text = f"{type(e).__name__}: {str(e)}"
                write_error_log(errors_log, trait_name, error_text)
                worker_logger.error(f"Failed trait '{trait_name}': {error_text}")

        worker_logger.info(f"Worker {worker_id} done. completed={completed}, failed={failed}")

    except Exception as e:
        error_text = f"Fatal worker error: {type(e).__name__}: {str(e)}"
        write_error_log(errors_log, "__worker__", error_text)
        worker_logger.error(error_text)
        worker_logger.error(traceback_string())


def traceback_string() -> str:
    import traceback
    return traceback.format_exc()


# ----------------------------
# orchestration
# ----------------------------

def discover_trait_names(traits_dir: Path, responses_dir: Path, selected_traits: Optional[List[str]]) -> List[str]:
    names = []
    for f in sorted(traits_dir.glob("*.json")):
        name = f.stem
        if selected_traits and name not in selected_traits:
            continue
        if (responses_dir / f"{name}.jsonl").exists():
            logger.info(f"Skipping '{name}' (already exists)")
            continue
        names.append(name)
    return names


def build_run_manifest(args, trait_names: List[str]) -> Dict[str, Any]:
    return {
        "created_at_utc": utc_now_iso(),
        "model": args.model,
        "traits_dir": str(Path(args.traits_dir).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "questions_source": "per-trait JSON files only",
        "expected_questions_per_trait": 40,
        "expected_instruction_pairs_per_trait": 5,
        "expected_rows_per_trait": 400,
        "selected_traits": args.traits,
        "trait_count_to_process": len(trait_names),
        "git_commit": safe_git_commit(),
    }


def run_multi_worker(args) -> int:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_ids = [int(x.strip()) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    total_gpus = len(gpu_ids)
    tensor_parallel_size = args.tensor_parallel_size

    if total_gpus == 0:
        logger.error("No GPUs available")
        return 1
    if tensor_parallel_size > total_gpus:
        logger.error(f"tensor_parallel_size ({tensor_parallel_size}) > available GPUs ({total_gpus})")
        return 1

    num_workers = total_gpus // tensor_parallel_size
    if num_workers == 0:
        logger.error("No workers could be created")
        return 1

    output_root = Path(args.output_root)
    responses_dir = output_root / "responses"
    traits_dir = Path(args.traits_dir)

    trait_names = discover_trait_names(traits_dir, responses_dir, args.traits)
    if not trait_names:
        logger.info("No traits to process")
        return 0

    write_json(output_root / "manifests" / "run_config.json", build_run_manifest(args, trait_names))

    gpu_chunks = [gpu_ids[i * tensor_parallel_size : (i + 1) * tensor_parallel_size] for i in range(num_workers)]
    trait_chunks = [[] for _ in range(num_workers)]
    for i, name in enumerate(trait_names):
        trait_chunks[i % num_workers].append(name)

    logger.info(f"Available GPUs: {gpu_ids}")
    logger.info(f"tensor_parallel_size: {tensor_parallel_size}")
    logger.info(f"num_workers: {num_workers}")
    for i in range(num_workers):
        logger.info(f"Worker {i} -> GPUs {gpu_chunks[i]} -> {len(trait_chunks[i])} traits")

    mp.set_start_method("spawn", force=True)
    processes = []

    for worker_id in range(num_workers):
        if not trait_chunks[worker_id]:
            continue
        p = mp.Process(
            target=process_traits_on_worker,
            args=(worker_id, gpu_chunks[worker_id], trait_chunks[worker_id], args),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logger.info("Generation complete")
    return 0


def run_single_worker(args) -> int:
    output_root = Path(args.output_root)
    responses_dir = output_root / "responses"
    verification_dir = output_root / "verification"
    logs_dir = output_root / "logs"
    errors_log = logs_dir / "single_worker_errors.log"
    traits_dir = Path(args.traits_dir)

    trait_names = discover_trait_names(traits_dir, responses_dir, args.traits)
    if not trait_names:
        logger.info("No traits to process")
        return 0

    write_json(output_root / "manifests" / "run_config.json", build_run_manifest(args, trait_names))

    generator = TraitResponseGenerator(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    generator.load()

    for trait_name in tqdm(trait_names, desc="Processing traits"):
        output_file = responses_dir / f"{trait_name}.jsonl"
        verification_file = verification_dir / f"{trait_name}.json"
        trait_file = traits_dir / f"{trait_name}.json"

        try:
            trait_data = load_trait(trait_file)
            rows = generator.generate_trait_rows(
                trait_name=trait_name,
                trait_file=trait_file,
                trait_data=trait_data,
            )

            output_file.parent.mkdir(parents=True, exist_ok=True)
            with jsonlines.open(output_file, "w") as writer:
                writer.write_all(rows)

            ok, msg = verify_trait_output_file(output_file, trait_data)
            write_json(
                verification_file,
                {
                    "trait": trait_name,
                    "verified_at": utc_now_iso(),
                    "ok": ok,
                    "message": msg,
                    "row_count": len(rows),
                    "expected_row_count": 400,
                },
            )

            if not ok:
                raise RuntimeError(f"Verification failed for {trait_name}: {msg}")

            logger.info(f"Saved and verified {len(rows)} responses for '{trait_name}'")

        except Exception as e:
            error_text = f"{type(e).__name__}: {str(e)}"
            write_error_log(errors_log, trait_name, error_text)
            logger.error(f"Failed trait '{trait_name}': {error_text}")

    logger.info("Generation complete")
    return 0


# ----------------------------
# cli
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate trait responses using only the 40 questions stored inside each trait JSON."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--traits_dir", type=str, default="data/traits/instructions")
    parser.add_argument("--output_root", type=str, default="full_trait_output/traits40_generation")
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--traits", nargs="+", help="Specific traits to process")
    return parser.parse_args()


def main():
    args = parse_args()

    output_root = Path(args.output_root)
    (output_root / "responses").mkdir(parents=True, exist_ok=True)
    (output_root / "verification").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        available_gpus = [int(x.strip()) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
        total_gpus = len(available_gpus)
    else:
        total_gpus = torch.cuda.device_count()

    args.tensor_parallel_size = args.tensor_parallel_size if args.tensor_parallel_size else max(1, total_gpus)

    use_multi_worker = total_gpus > 1 and total_gpus > args.tensor_parallel_size

    if use_multi_worker:
        sys.exit(run_multi_worker(args))
    else:
        sys.exit(run_single_worker(args))


if __name__ == "__main__":
    main()