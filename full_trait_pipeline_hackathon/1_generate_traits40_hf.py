#!/usr/bin/env python3
# HF transformers replacement for the vLLM-based 1_generate_traits40.py
# Uses chunked_sdpa for Gemma 4-31B's head_dim=512.

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
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# chunked_sdpa shim — copy from /scratch/mechhack/starter_code/chunked_sdpa.py
sys.path.insert(0, str(Path(__file__).parent))
from chunked_sdpa import chunked_sdpa_scope


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


def load_trait(trait_file: Path) -> Dict[str, Any]:
    with open(trait_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "instruction" not in data or "questions" not in data or "eval_prompt" not in data:
        raise ValueError(f"{trait_file} missing one of: instruction, questions, eval_prompt")
    if not isinstance(data["instruction"], list) or len(data["instruction"]) != 5:
        raise ValueError(f"{trait_file} must contain exactly 5 instruction pairs")
    if not isinstance(data["questions"], list) or len(data["questions"]) != 40:
        raise ValueError(f"{trait_file} must contain exactly 40 questions")
    for i, pair in enumerate(data["instruction"]):
        if not isinstance(pair, dict) or "pos" not in pair or "neg" not in pair:
            raise ValueError(f"{trait_file} instruction[{i}] must contain pos and neg")
    return data


def build_messages(system_prompt: str, question: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": question}]


def apply_chat_template_string(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def apply_chat_template_tokens(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return ids


def find_subsequence(seq, sub):
    if not sub:
        return -1
    n, m = len(seq), len(sub)
    for i in range(n - m + 1):
        if seq[i:i + m] == sub:
            return i
    return -1


def build_span_metadata(tokenizer, full_ids, system_prompt, question):
    if len(full_ids) == 0:
        raise ValueError("Prompt token ids are empty")
    no_gen = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
        tokenize=True, add_generation_prompt=False,
    )
    if isinstance(no_gen, torch.Tensor):
        no_gen = no_gen.tolist()
    if len(no_gen) >= len(full_ids):
        raise ValueError(f"add_generation_prompt didn't append: {len(no_gen)} vs {len(full_ids)}")
    ah_start = len(no_gen)
    ah_end = len(full_ids) - 1
    ah_idxs = list(range(ah_start, ah_end + 1))
    q_ids = tokenizer.encode(question, add_special_tokens=False)
    if not q_ids:
        raise ValueError("Question encoded to zero tokens")
    uc_start = find_subsequence(full_ids, q_ids)
    if uc_start == -1:
        raise ValueError("Could not locate user-content tokens")
    uc_end = uc_start + len(q_ids) - 1
    uc_idxs = list(range(uc_start, uc_end + 1))
    return ah_idxs, ah_start, ah_end, uc_idxs, uc_start, uc_end, uc_end


def build_prompt_records_for_trait(tokenizer, trait_name, trait_file, trait_data):
    records = []
    for pi, pair in enumerate(trait_data["instruction"]):
        for polarity, sp in [("positive", pair["pos"]), ("negative", pair["neg"])]:
            for qi, q in enumerate(trait_data["questions"]):
                msgs = build_messages(sp, q)
                full_ids = apply_chat_template_tokens(tokenizer, msgs)
                ah_idxs, ah_s, ah_e, uc_idxs, uc_s, uc_e, ult = build_span_metadata(
                    tokenizer, full_ids, sp, q)
                records.append(PromptRecord(
                    trait=trait_name, trait_file=str(trait_file),
                    polarity=polarity, prompt_index=pi, question_index=qi,
                    label=f"{polarity}_p{pi}_q{qi}",
                    system_prompt=sp, instruction_text=sp, question=q, messages=msgs,
                    prompt_token_count=len(full_ids),
                    full_prompt_last_token_index=len(full_ids) - 1,
                    assistant_header_token_indices=ah_idxs,
                    assistant_header_token_start=ah_s,
                    assistant_header_token_end=ah_e,
                    user_content_token_indices=uc_idxs,
                    user_content_token_start=uc_s,
                    user_content_token_end=uc_e,
                    user_last_token_index=ult,
                    full_prompt_token_ids=full_ids,
                ))
    return records


def record_to_output_row(record, response, model_name, gen_params):
    conversation = [
        {"role": "system", "content": record.system_prompt},
        {"role": "user", "content": record.question},
        {"role": "assistant", "content": response},
    ]
    return {
        "trait": record.trait, "trait_file": record.trait_file,
        "polarity": record.polarity, "prompt_index": record.prompt_index,
        "question_index": record.question_index, "label": record.label,
        "system_prompt": record.system_prompt, "instruction_text": record.instruction_text,
        "question": record.question, "conversation": conversation,
        "assistant_response": response, "model": model_name,
        "generation_params": gen_params,
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


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_error_log(path, trait, err, tb=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{utc_now_iso()}] trait={trait}\n{err}\n")
        if tb:
            f.write(tb if tb.endswith("\n") else tb + "\n")
        f.write("\n")


def verify_trait_output_file(output_file, trait_data):
    rows = list(jsonlines.open(output_file, "r"))
    if len(rows) != 400:
        return False, f"Expected 400 rows, found {len(rows)}"
    seen = set()
    for row in rows:
        key = (row["polarity"], row["prompt_index"], row["question_index"])
        if key in seen:
            return False, f"Duplicate key: {key}"
        seen.add(key)
        if row["question"] != trait_data["questions"][row["question_index"]]:
            return False, f"Question mismatch at qidx={row['question_index']}"
        expected_sp = (trait_data["instruction"][row["prompt_index"]]["pos"]
                       if row["polarity"] == "positive"
                       else trait_data["instruction"][row["prompt_index"]]["neg"])
        if row["system_prompt"] != expected_sp:
            return False, f"System prompt mismatch at {row['polarity']} pidx={row['prompt_index']}"
    for pi in range(5):
        for pol in ("positive", "negative"):
            for qi in range(40):
                if (pol, pi, qi) not in seen:
                    return False, f"Missing key: {(pol, pi, qi)}"
    return True, "ok"


class TraitResponseGenerator:
    def __init__(self, model_name, max_model_len, temperature, top_p, max_tokens, batch_size):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self._sdpa_cm = None

    def load(self):
        logger.info(f"Loading tokenizer from {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # required for batched generation

        logger.info(f"Loading model from {self.model_name} (this takes ~1 min)")
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

        # chunked_sdpa needed for Gemma 4-31B's head_dim=512
        self._sdpa_cm = chunked_sdpa_scope()
        self._sdpa_cm.__enter__()
        logger.info("Model loaded.")

    def unload(self):
        if self._sdpa_cm is not None:
            self._sdpa_cm.__exit__(None, None, None)
            self._sdpa_cm = None

    def _generate_batch(self, prompt_texts: List[str]) -> List[str]:
        inputs = self.tokenizer(
            prompt_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_model_len, add_special_tokens=False,
        ).to("cuda:0")
        seq_in = inputs.input_ids.shape[1]
        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature if self.temperature > 0 else 1.0,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        completions = []
        for i in range(gen.shape[0]):
            out_ids = gen[i, seq_in:]
            completions.append(self.tokenizer.decode(out_ids, skip_special_tokens=True))
        del gen, inputs
        torch.cuda.empty_cache()
        return completions

    def generate_trait_rows(self, trait_name, trait_file, trait_data):
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Generator not loaded")
        records = build_prompt_records_for_trait(self.tokenizer, trait_name, trait_file, trait_data)
        prompt_texts = [apply_chat_template_string(self.tokenizer, r.messages) for r in records]

        all_responses: List[str] = []
        for i in tqdm(range(0, len(prompt_texts), self.batch_size),
                     desc=f"  {trait_name}", leave=False):
            batch = prompt_texts[i:i + self.batch_size]
            all_responses.extend(self._generate_batch(batch))

        gen_params = {"temperature": self.temperature, "top_p": self.top_p,
                      "max_tokens": self.max_tokens, "n": 1}
        return [record_to_output_row(r, resp, self.model_name, gen_params)
                for r, resp in zip(records, all_responses)]


def discover_trait_names(traits_dir, responses_dir, selected):
    names = []
    for f in sorted(traits_dir.glob("*.json")):
        n = f.stem
        if selected and n not in selected:
            continue
        if (responses_dir / f"{n}.jsonl").exists():
            logger.info(f"Skipping '{n}' (already exists)")
            continue
        names.append(n)
    return names


def build_run_manifest(args, trait_names):
    return {
        "created_at_utc": utc_now_iso(), "model": args.model,
        "traits_dir": str(Path(args.traits_dir).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "max_model_len": args.max_model_len, "temperature": args.temperature,
        "top_p": args.top_p, "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "questions_source": "per-trait JSON files only",
        "expected_questions_per_trait": 40,
        "expected_instruction_pairs_per_trait": 5,
        "expected_rows_per_trait": 400,
        "selected_traits": args.traits,
        "trait_count_to_process": len(trait_names),
        "git_commit": safe_git_commit(),
        "backend": "huggingface_transformers",
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--traits_dir", type=str, default="data/traits/instructions")
    p.add_argument("--output_root", type=str, default="full_trait_output/traits40_generation")
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--traits", nargs="+", help="Specific traits to process")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_root)
    (out / "responses").mkdir(parents=True, exist_ok=True)
    (out / "verification").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    responses_dir = out / "responses"
    verification_dir = out / "verification"
    error_log = out / "logs" / "errors.log"
    traits_dir = Path(args.traits_dir)

    trait_names = discover_trait_names(traits_dir, responses_dir, args.traits)
    if not trait_names:
        logger.info("No traits to process")
        return 0

    write_json(out / "manifests" / "run_config.json", build_run_manifest(args, trait_names))

    gen = TraitResponseGenerator(
        model_name=args.model, max_model_len=args.max_model_len,
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, batch_size=args.batch_size,
    )
    gen.load()

    try:
        for tn in tqdm(trait_names, desc="Traits"):
            output_file = responses_dir / f"{tn}.jsonl"
            verification_file = verification_dir / f"{tn}.json"
            trait_file = traits_dir / f"{tn}.json"
            try:
                td = load_trait(trait_file)
                rows = gen.generate_trait_rows(tn, trait_file, td)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with jsonlines.open(output_file, "w") as w:
                    w.write_all(rows)
                ok, msg = verify_trait_output_file(output_file, td)
                write_json(verification_file, {
                    "trait": tn, "verified_at": utc_now_iso(),
                    "ok": ok, "message": msg,
                    "row_count": len(rows), "expected_row_count": 400,
                })
                if not ok:
                    raise RuntimeError(f"Verification failed for {tn}: {msg}")
                logger.info(f"Saved+verified {len(rows)} for '{tn}'")
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                write_error_log(error_log, tn, err, traceback_string())
                logger.error(f"Failed '{tn}': {err}")
    finally:
        gen.unload()

    logger.info("Generation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
