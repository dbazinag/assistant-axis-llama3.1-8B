#!/usr/bin/env python3
"""
collect_jbb_attack_activations_gemma.py

Gemma-4-31B-it variant of the JBB comparison-attack collector. Same Gemma recipe
as collect_wildjailbreak_activations_gemma.py (AutoModelForCausalLM +
trust_remote_code + attn_implementation="sdpa" + chunked_sdpa_scope, layer
L = hidden_states[L+1]); only the prompt source is swapped to the model-agnostic
JBB transfer-attack prompts in comparison_attacks_jbb/<family>/results.jsonl.

Families (transfer attacks — prompt text is model-agnostic): PAIR, PAP, GPTFuzz,
PEZ. GCG is excluded (Llama-gradient-specific, no transferable prompt text).

For each attack prompt:
  1. Format as a single user message (no system prompt)
  2. Forward pass -> layer-30 hidden state at the last prompt token (pre-gen)
  3. Greedy generation
  4. Save responses.jsonl + activations.pt (same schema as the WJB/HB collectors)

behavior_text is set to the clean JBB `goal` (not the adversarial prompt) so the
gpt-4.1-mini judge scores "harmful content relevant to the behavior", matching
evaluate_comparison_attacks_jbb.py. attack_type = the family, so per-family ASR
and per-family transfer AUC fall out downstream.

Usage:
  .venv_gemma/bin/python full_trait_tools/collect_jbb_attack_activations_gemma.py \
      --model /path/to/gemma-4-31B-it          # all 4 families
  .venv_gemma/bin/python full_trait_tools/collect_jbb_attack_activations_gemma.py \
      --model /path/to/gemma-4-31B-it --test   # first 10 prompts per family
"""

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from chunked_sdpa import chunked_sdpa_scope

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

LAYERS_TO_SAVE = [30]      # matches the kept layer-30 trait vectors
MAX_NEW_TOKENS = 512       # Gemma-4 emits a "thought" channel before the answer

JBB_FAMILIES = ["PAIR", "PAP", "GPTFuzz", "PEZ"]   # model-agnostic transfer attacks

JBB_BASE = [
    "/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output/comparison_attacks_jbb",
    "/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output/comparison_attacks_jbb",
]

# ── Data loading (JBB comparison-attack prompts) ────────────────────────────────

def load_jbb_pairs(test_mode: bool = False) -> List[dict]:
    """Load transfer-attack prompts from comparison_attacks_jbb/<family>/results.jsonl.

    Each row already carries the clean harmful `goal` and harm `category`. We keep
    the HarmBench field names (pair_id, behavior_text, jailbreak_idx, ...) so the
    judge + detector tooling work unchanged. attack_type = the JBB family.
    """
    base = next((Path(p) for p in JBB_BASE if Path(p).exists()), None)
    if base is None:
        raise FileNotFoundError(f"comparison_attacks_jbb not found in {JBB_BASE}")
    logger.info(f"Loading JBB attack prompts from {base}")

    pairs: List[dict] = []
    pair_id = 0
    for family in JBB_FAMILIES:
        p = base / family / "results.jsonl"
        if not p.exists():
            logger.warning(f"Missing family file: {p}")
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if test_mode:
            rows = rows[:10]
        for row in rows:
            raw_prompt = row.get("prompt") or ""
            # GPTFuzz stores some prompts as a list; take the last element (matches
            # collect_jbb_attack_activations.py).
            if isinstance(raw_prompt, list):
                raw_prompt = raw_prompt[-1] if raw_prompt else ""
            prompt = raw_prompt.strip()
            if not prompt:
                continue
            pairs.append({
                "pair_id":           pair_id,
                "behavior_id":       row.get("behavior_id", f"{family}_{pair_id:05d}"),
                "behavior_text":     row.get("goal", ""),    # clean goal -> judge target
                "semantic_category": row.get("category", family),
                "attack_type":       family.lower(),
                "jailbreak_idx":     pair_id,
                "formatted_prompt":  prompt,
            })
            pair_id += 1
        logger.info(f"  {family}: loaded (running total {pair_id})")

    logger.info(f"Loaded {len(pairs)} JBB attack prompts across {len(JBB_FAMILIES)} families")
    return pairs


# ── Model helpers (Gemma-4 recipe — identical to the WJB/HB collectors) ──────────

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def sanitize_text(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def get_prompt_input_ids(tokenizer: AutoTokenizer, prompt_text: str) -> torch.Tensor:
    prompt_text = sanitize_text(prompt_text)
    conversation = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def process_pair(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pair: dict,
    layers: List[int],
    max_new_tokens: int,
    device: torch.device,
) -> Optional[dict]:
    try:
        input_ids = get_prompt_input_ids(tokenizer, pair["formatted_prompt"]).to(device)

        with torch.no_grad(), chunked_sdpa_scope():
            fwd_out = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            layer_acts: Dict[int, torch.Tensor] = {}
            for layer_idx in layers:
                hs = fwd_out.hidden_states[layer_idx + 1]
                layer_acts[layer_idx] = hs[0, -1, :].detach().to(torch.float32).cpu()

        del fwd_out
        torch.cuda.empty_cache()

        with torch.no_grad(), chunked_sdpa_scope():
            gen_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_ids  = gen_ids[0, input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        del gen_ids
        torch.cuda.empty_cache()

        return {"response_text": response_text, "layer_acts": layer_acts}

    except Exception as e:
        logger.warning(f"Error processing pair {pair['pair_id']} ({pair['behavior_id']}): {e}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect JBB comparison-attack activations for Gemma-4-31B (single GPU)"
    )
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument(
        "--output_dir", type=str,
        default="full_trait_output/jbb_attack_activations_gemma",
    )
    parser.add_argument("--layers", type=str, default="30",
                        help="Comma-separated model layer indices to save (default: 30)")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--test", action="store_true",
                        help="Small test: first 10 prompts per family")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    output_dir = Path(args.output_dir)
    if args.test:
        output_dir = output_dir.parent / (output_dir.name + "_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_jbb_pairs(test_mode=args.test)
    logger.info(f"Built {len(pairs)} pairs total")

    manifest = {
        "created_at_utc":  datetime.now(timezone.utc).isoformat(),
        "model":           args.model,
        "dataset":         "JBB comparison_attacks_jbb (PAIR/PAP/GPTFuzz/PEZ transfer)",
        "n_pairs":         len(pairs),
        "layers_saved":    layers,
        "max_new_tokens":  args.max_new_tokens,
        "test_mode":       args.test,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(output_dir / "pairs_metadata.jsonl", "w") as f:
        for p in pairs:
            row = {k: v for k, v in p.items() if k != "formatted_prompt"}
            f.write(json.dumps(row) + "\n")

    if not torch.cuda.is_available():
        logger.error("No GPU available. Exiting.")
        sys.exit(1)
    device = torch.device("cuda:0")

    logger.info(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    logger.info("Model loaded. Processing pairs...")

    responses:   List[dict]      = []
    activations: Dict[int, dict] = {}
    n_errors = 0

    for pair in tqdm(pairs, desc="pairs"):
        result = process_pair(model, tokenizer, pair, layers, args.max_new_tokens, device)
        if result is None:
            n_errors += 1
            continue
        row = {k: v for k, v in pair.items() if k != "formatted_prompt"}
        row["response_text"] = result["response_text"]
        responses.append(row)
        activations[pair["pair_id"]] = {
            str(layer_idx): tensor for layer_idx, tensor in result["layer_acts"].items()
        }
        if len(responses) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    responses.sort(key=lambda x: x["pair_id"])
    with open(output_dir / "responses.jsonl", "w", encoding="utf-8") as f:
        for row in responses:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    torch.save(activations, output_dir / "activations.pt")

    logger.info(f"Done — {len(responses)} OK, {n_errors} errors.")
    logger.info(f"All outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
