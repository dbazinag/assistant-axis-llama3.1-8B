#!/usr/bin/env python3
"""
collect_wildjailbreak_activations_gemma.py

Gemma-4-31B-it variant of the WildJailbreak collector. Same Gemma recipe as
collect_harmbench_activations_gemma.py (AutoModelForCausalLM + trust_remote_code +
attn_implementation="sdpa" + chunked_sdpa_scope, layer L = hidden_states[L+1]);
only the pair source is swapped from HarmBench to the WildJailbreak eval
adversarial_harmful prompts (2000 full adversarial prompts, no templates).

For each adversarial_harmful prompt:
  1. Format as a single user message (no system prompt)
  2. Forward pass -> layer-30 hidden state at the last prompt token (pre-gen)
  3. Greedy generation
  4. Save responses.jsonl + activations.pt (same schema as the HarmBench collector)

Then judge responses.jsonl with judge_harmbench_responses.py to get per-prompt ASR.

Usage:
  .venv_gemma/bin/python full_trait_tools/collect_wildjailbreak_activations_gemma.py \
      --model /path/to/gemma-4-31B-it          # full 2000 prompts
  .venv_gemma/bin/python full_trait_tools/collect_wildjailbreak_activations_gemma.py \
      --model /path/to/gemma-4-31B-it --test   # first 10 prompts
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

# Pre-exported WildJailbreak eval adversarial_harmful prompts ({"i", "adversarial"}
# per line), so the Gemma venv never needs the `datasets` library. Built once with
# the main .venv via load_dataset("allenai/wildjailbreak", name="eval").
PROMPTS_JSONL = [
    "/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output/wjb_eval_adversarial_harmful.jsonl",
    "/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output/wjb_eval_adversarial_harmful.jsonl",
]

# ── Data loading (WildJailbreak eval adversarial_harmful) ───────────────────────

def load_wildjailbreak_pairs(test_mode: bool = False) -> List[dict]:
    """Load adversarial_harmful prompts from the pre-exported JSONL.

    Each WJB prompt is a single standalone adversarial prompt (NOT a template ×
    behavior pair). We keep the HarmBench field names (pair_id, behavior_text,
    jailbreak_idx, ...) only so the judge + detector tooling work unchanged:
    pair_id is just a row index, jailbreak_idx is the WJB dataset row index.
    """
    src = next((p for p in PROMPTS_JSONL if os.path.exists(p)), None)
    if src is None:
        raise FileNotFoundError(f"WJB prompts JSONL not found in {PROMPTS_JSONL}")
    logger.info(f"Loading WJB adversarial_harmful prompts from {src}")

    pairs: List[dict] = []
    pair_id = 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            adversarial = (rec.get("adversarial") or "").strip()
            if not adversarial:
                continue
            i = rec["i"]
            pairs.append({
                "pair_id":           pair_id,
                "behavior_id":       f"wjb_{i:05d}",
                "behavior_text":     adversarial,
                "semantic_category": "wildjailbreak",
                "attack_type":       "wjb_adversarial",
                "jailbreak_idx":     i,
                "formatted_prompt":  adversarial,
            })
            pair_id += 1
            if test_mode and pair_id >= 10:
                break

    logger.info(f"Loaded {len(pairs)} adversarial_harmful prompts")
    return pairs


# ── Model helpers (Gemma-4 recipe — identical to the HarmBench collector) ────────

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
        description="Collect WildJailbreak activations for Gemma-4-31B (single GPU)"
    )
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument(
        "--output_dir", type=str,
        default="full_trait_output/wildjailbreak_activations_gemma",
    )
    parser.add_argument("--layers", type=str, default="30",
                        help="Comma-separated model layer indices to save (default: 30)")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--test", action="store_true",
                        help="Small test: first 10 adversarial_harmful prompts")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    output_dir = Path(args.output_dir)
    if args.test:
        output_dir = output_dir.parent / (output_dir.name + "_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_wildjailbreak_pairs(test_mode=args.test)
    logger.info(f"Built {len(pairs)} pairs total")

    manifest = {
        "created_at_utc":  datetime.now(timezone.utc).isoformat(),
        "model":           args.model,
        "dataset":         "allenai/wildjailbreak (eval, adversarial_harmful)",
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
