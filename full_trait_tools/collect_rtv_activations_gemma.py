#!/usr/bin/env python3
"""RTV hidden-state collector for Gemma-4-31B-it (single GPU + chunked SDPA).

Mirrors collect_rtv_activations.py but for Gemma-4-31B. Two Gemma-specific changes:

1. Model loading: bf16 + attn_implementation="sdpa" + device_map="cuda:0", and every
   forward is wrapped in chunked_sdpa_scope() (Gemma-4 global layers have head_dim=512,
   which OOMs plain math-SDPA above a few k tokens).
2. Layers: Gemma-4-31B has 60 transformer layers (vs 32 for Llama/OLMo). We keep the
   SAME dict-key labels [18,25,32] the geometry baselines read, but map them to Gemma
   layers [34,48,59] (relative-depth match to Llama's [18,25,31]: ~0.58/0.81/1.0 of
   depth). So MODEL_LAYERS=[34,48,59] -> hidden_states[35],[49],[60] (last). No change
   is needed in the geometry baseline scripts (they read keys "18","25","32").

Everything else — alpaca/harmbench/attack row building, alignment asserts, 5-last-token
slicing, output layout ({dataset}_rows.jsonl + {dataset}_activations.pt) — is reused
verbatim from collect_rtv_activations.py. Attack prompts come from the OLMo test-case
files (the Gemma attack collector consumed exactly those), so pair ordering matches.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from chunked_sdpa import chunked_sdpa_scope

# Reuse all the (model-independent) data-building + IO helpers from the Llama collector.
import collect_rtv_activations as R
from collect_rtv_activations import (
    build_alpaca_harmless_rows,
    build_attack_rows_olmo3,
    build_harmbench_rows,
    load_jsonl,
    merge_dataset_outputs,
    prompt_input_ids,
    write_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GEMMA_SNAPSHOT = ("/mnt/dlabscratch1/bazina/.cache/huggingface/hub/"
                  "models--google--gemma-4-31B-it/snapshots/"
                  "fb9ae262347c3945692f09a612f8bb189def854f")

# Dict-key labels kept identical to Llama/OLMo so the geometry baselines read them
# unchanged; the actual Gemma hidden-state layers they map to are MODEL_LAYERS.
PAPER_LAYERS = [18, 25, 32]
MODEL_LAYERS = [34, 48, 59]   # Gemma-4-31B: mid / late / last of 60 layers
N_LAST_TOKENS = R.N_LAST_TOKENS

OUTPUT_ROOT_GEMMA = "full_trait_output/rtv_activations_gemma"

# Gemma reuses the OLMo attack test-cases (same exp dirs); only the response paths change.
DATASET_CONFIGS_GEMMA = {
    "harmbench": {"responses": "full_trait_output/harmbench_activations_gemma/classified_responses.jsonl"},
    "pair":    {"responses": "full_trait_output/pair_activations_gemma_hb/classified_responses.jsonl",    "exp": "olmo3_7b_harmbench"},
    "pap":     {"responses": "full_trait_output/pap_activations_gemma_hb/classified_responses.jsonl",     "exp": "olmo3_7b_top5"},
    "gptfuzz": {"responses": "full_trait_output/gptfuzz_activations_gemma_hb/classified_responses.jsonl", "exp": "olmo3_7b"},
    "pez":     {"responses": "full_trait_output/pez_activations_gemma_hb/classified_responses.jsonl",     "exp": "olmo3_7b"},
    # WJB prompts are standalone adversarial strings stored in behavior_text (no templates),
    # so the RTV prompt is behavior_text verbatim — same prompt used for the layer-30 collection.
    "wildjailbreak": {"responses": "full_trait_output/wildjailbreak_activations_gemma/classified_responses.jsonl"},
}


def build_wildjailbreak_rows(responses_path: Path) -> List[dict]:
    """WJB RTV rows: prompt_text = behavior_text (the standalone adversarial prompt)."""
    rows = []
    for r in load_jsonl(responses_path):
        rows.append({
            "pair_id":       r["pair_id"],
            "behavior_id":   r.get("behavior_id", f"wjb_{r['pair_id']}"),
            "attack_type":   "wjb_adversarial",
            "jailbreak_idx": -1,
            "jailbroken":    r.get("jailbroken"),
            "prompt_text":   r["behavior_text"],
        })
    return rows


def build_rows_for_dataset(name: str, args) -> List[dict]:
    if name == "alpaca_harmless":
        return build_alpaca_harmless_rows(args.n_alpaca_harmless, args.seed)
    cfg = DATASET_CONFIGS_GEMMA[name]
    if name == "harmbench":
        return build_harmbench_rows(Path(cfg["responses"]))
    if name == "wildjailbreak":
        return build_wildjailbreak_rows(Path(cfg["responses"]))
    return build_attack_rows_olmo3(name, cfg)


def load_model_and_tokenizer(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_SNAPSHOT, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_SNAPSHOT,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def process_row(model, tokenizer, row: dict, device: torch.device):
    input_ids = prompt_input_ids(tokenizer, row["prompt_text"], device)
    attention_mask = torch.ones_like(input_ids)
    if input_ids.shape[1] < N_LAST_TOKENS:
        raise ValueError(f"prompt has only {input_ids.shape[1]} tokens")
    with torch.no_grad(), chunked_sdpa_scope():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    layer_acts = {}
    for paper_layer, model_layer in zip(PAPER_LAYERS, MODEL_LAYERS):
        hidden = outputs.hidden_states[model_layer + 1][0, -N_LAST_TOKENS:, :].detach().cpu()
        layer_acts[str(paper_layer)] = hidden
    return layer_acts


def collect_dataset(dataset: str, rows: List[dict], model, tokenizer, args, output_dir: Path):
    if args.test:
        rows = rows[: min(len(rows), args.test_rows)]
    logger.info(f"{dataset}: collecting {len(rows)} rows")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    activations: Dict[int, dict] = {}
    ok_rows: List[dict] = []
    n_errors = 0
    for row in tqdm(rows, desc=dataset):
        try:
            activations[row["pair_id"]] = process_row(model, tokenizer, row, device)
            ok_rows.append({k: v for k, v in row.items() if k != "prompt_text"})
        except Exception as exc:
            n_errors += 1
            logger.warning(f"{dataset} row {row.get('pair_id')} failed: {exc}")
        if len(activations) % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    # single worker -> write worker-0 shards, then merge into {dataset}_{rows,activations}
    torch.save(activations, output_dir / f"{dataset}_worker_0_activations.pt")
    write_jsonl(output_dir / f"{dataset}_worker_0_rows.jsonl", ok_rows)
    merge_dataset_outputs(output_dir, dataset, 1)
    logger.info(f"{dataset}: ok={len(ok_rows)} errors={n_errors}")


def main():
    parser = argparse.ArgumentParser(description="Collect RTV activations for Gemma-4-31B (single GPU)")
    parser.add_argument("--datasets", default="alpaca_harmless,harmbench,pair,pap,gptfuzz,pez")
    parser.add_argument("--output_dir", default=OUTPUT_ROOT_GEMMA)
    parser.add_argument("--n_alpaca_harmless", type=int, default=R.N_ALPACA_HARMLESS)
    parser.add_argument("--seed", type=int, default=R.RANDOM_SEED)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_rows", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.test:
        output_dir = output_dir.parent / f"{output_dir.name}_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
    valid = {"alpaca_harmless", *DATASET_CONFIGS_GEMMA.keys()}
    for d in requested:
        if d not in valid:
            raise ValueError(f"Unknown dataset {d!r}; valid: {sorted(valid)}")

    manifest = {
        "model": GEMMA_SNAPSHOT,
        "paper_layers": PAPER_LAYERS,
        "model_layers": MODEL_LAYERS,
        "n_last_tokens": N_LAST_TOKENS,
        "datasets": requested,
        "note": "keys are Llama labels [18,25,32]; actual Gemma layers are [34,48,59]",
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Loading Gemma from {GEMMA_SNAPSHOT} ...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(device)
    logger.info("Model loaded.")

    for dataset in requested:
        rows = build_rows_for_dataset(dataset, args)
        collect_dataset(dataset, rows, model, tokenizer, args, output_dir)

    logger.info(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
