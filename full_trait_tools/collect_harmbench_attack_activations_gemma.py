#!/usr/bin/env python3
"""
collect_harmbench_attack_activations_gemma.py

Gemma-4-31B-it collector for HarmBench attack families (PAIR/PAP/GPTFuzz/PEZ) on
HarmBench behaviors. Data loading mirrors collect_harmbench_attack_activations_olmo3.py
(reads HarmBench/results/HarmBench_{attack}/{exp}/test_cases/test_cases_individual_behaviors/
*/test_cases.json, flattens nested prompt lists, behavior_text = the clean HarmBench
Behavior so the judge scores harmful content relevant to the behavior). The model
recipe is the Gemma-4 one (transformers 5.8, trust_remote_code, attn=sdpa under
chunked_sdpa_scope, layer L = hidden_states[L+1], single GPU) — same as the
WildJailbreak/HarmBench Gemma collectors.

PAP generation is target-agnostic, so --exp can point at the existing
olmo3_7b_top5 PAP test cases (identical distribution for Gemma). Other families
need Gemma-specific test cases (blocked on HarmBench vLLM not supporting gemma4).

Usage:
  .venv_gemma/bin/python full_trait_tools/collect_harmbench_attack_activations_gemma.py \
      --model /path/to/gemma-4-31B-it --attack_type PAP --exp olmo3_7b_top5
  ... --attack_type PAP --exp olmo3_7b_top5 --test   # first 5 behaviors
"""

import argparse
import csv
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

HARMBENCH_ROOT = Path("/dlabscratch1/bazina/HarmBench")
if not HARMBENCH_ROOT.exists():
    HARMBENCH_ROOT = Path("/mnt/dlabscratch1/bazina/HarmBench")

# Default experiment-name per attack (overridable with --exp). PAP is
# target-agnostic so the olmo3_7b_top5 PAP test cases are valid for Gemma.
DEFAULT_EXP = {
    "PAIR":    "olmo3_7b_harmbench",
    "PAP":     "olmo3_7b_top5",
    "GPTFuzz": "olmo3_7b",
    "PEZ":     "olmo3_7b",
}

# ── Data loading (HarmBench attack test cases — same as the OLMo collector) ──────

def load_individual_test_cases(attack_type: str, exp: str) -> dict[str, list]:
    """Merge per-behavior test_cases.json files into {behavior_id: [prompt, ...]}."""
    ind_dir = (HARMBENCH_ROOT / "results" / f"HarmBench_{attack_type}" / exp
               / "test_cases" / "test_cases_individual_behaviors")
    if not ind_dir.exists():
        raise FileNotFoundError(f"Not found: {ind_dir}")

    merged: dict[str, list] = {}
    for bdir in sorted(ind_dir.iterdir()):
        if not bdir.is_dir():
            continue
        tc_file = bdir / "test_cases.json"
        if not tc_file.exists():
            continue
        with tc_file.open() as f:
            d = json.load(f)
        for bid, prompts in d.items():
            flat: list[str] = []
            for p in prompts:                    # flatten nested lists (PAP/GPTFuzz)
                if isinstance(p, list):
                    flat.extend(p)
                elif isinstance(p, str):
                    flat.append(p)
            if flat:
                merged[bid] = merged.get(bid, []) + flat

    logger.info(f"Merged {len(merged)} behaviors for {attack_type} ({exp})")
    return merged


def load_behaviors(csv_path: Path) -> dict[str, dict]:
    behaviors = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            behaviors[row["BehaviorID"]] = row
    logger.info(f"Loaded {len(behaviors)} behaviors from CSV")
    return behaviors


def build_pairs(test_cases: dict, behaviors: dict, attack_type: str) -> list[dict]:
    pairs = []
    pair_id = 0
    skipped = 0
    for behavior_id, prompts in test_cases.items():
        bv = behaviors.get(behavior_id)
        if bv is None:
            skipped += 1
            continue
        for prompt in prompts:
            if not prompt or not isinstance(prompt, str) or not prompt.strip():
                continue
            pairs.append({
                "pair_id":           pair_id,
                "behavior_id":       behavior_id,
                "behavior_text":     bv["Behavior"],          # clean behavior -> judge
                "semantic_category": bv.get("SemanticCategory", ""),
                "attack_type":       attack_type.lower(),
                "jailbreak_idx":     -1,
                "formatted_prompt":  prompt.strip(),
            })
            pair_id += 1
    if skipped:
        logger.warning(f"Skipped {skipped} behaviors not in behaviors CSV")
    logger.info(f"Built {len(pairs)} pairs")
    return pairs


# ── Model helpers (Gemma-4 recipe — identical to the WJB/HB Gemma collectors) ────

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
        description="Collect HarmBench attack-family activations for Gemma-4-31B (single GPU)"
    )
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--attack_type", required=True,
                        choices=["PAIR", "PAP", "GPTFuzz", "PEZ"])
    parser.add_argument("--exp", type=str, default=None,
                        help="HarmBench experiment_name dir (default: per-attack DEFAULT_EXP)")
    parser.add_argument("--behaviors_path", type=str,
                        default=str(HARMBENCH_ROOT / "data/behavior_datasets/harmbench_behaviors_text_all.csv"))
    parser.add_argument("--output_dir", type=str, default=None,
                        help="default: full_trait_output/<attack>_activations_gemma_hb")
    parser.add_argument("--layers", type=str, default="30")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--test", action="store_true",
                        help="Small test: first 5 behaviors")
    args = parser.parse_args()

    exp = args.exp or DEFAULT_EXP[args.attack_type]
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    out = args.output_dir or f"full_trait_output/{args.attack_type.lower()}_activations_gemma_hb"
    output_dir = Path(out)
    if args.test:
        output_dir = output_dir.parent / (output_dir.name + "_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_cases = load_individual_test_cases(args.attack_type, exp)
    if args.test:
        bids = list(test_cases.keys())[:5]
        test_cases = {bid: test_cases[bid] for bid in bids}
    behaviors = load_behaviors(Path(args.behaviors_path))
    pairs = build_pairs(test_cases, behaviors, args.attack_type)
    logger.info(f"Built {len(pairs)} pairs total")

    manifest = {
        "created_at_utc":  datetime.now(timezone.utc).isoformat(),
        "model":           args.model,
        "attack_type":     args.attack_type,
        "experiment_name": exp,
        "n_behaviors":     len(test_cases),
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
