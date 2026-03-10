#!/usr/bin/env python3
"""
extract_trait_vectors.py
------------------------
Extract trait vectors from the Assistant Axis repo's data/extraction_questions/traits/
for Llama-3.1-8B-Instruct, following the Persona Vectors paper methodology.

Method:
  1. For each (system_prompt, question) pair, generate a response and collect
     mean response activations at all 32 layers via post-MLP residual stream hooks
  2. Judge each response using the trait's eval_prompt via gpt-4o-mini:
       - pos responses: keep if score > 50
       - neg responses: keep if score < 50
  3. Compute mean(filtered_pos_acts) - mean(filtered_neg_acts) at layer 16
  4. Normalise to unit vector and save as {trait}.pt

Output saved to: --out_dir/{trait}.pt  (shape [4096], float32)
Also saves {trait}_metadata.json with filtering stats.

Usage:
    uv run tools/extract_trait_vectors.py \\
        --trait_dir  $BASE/assistant-axis-llama3.1-8B/data/extraction_questions/traits \\
        --model_id   meta-llama/Llama-3.1-8B-Instruct \\
        --out_dir    $BASE/assistant_axis_outputs/llama-3.1-8b/vectors_q50 \\
        --layer      16 \\
        --openai_api_key sk-...
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("transformers not found. Run: uv add transformers")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai not found. Run: uv add openai")


# ── LLM Judge ────────────────────────────────────────────────────────────────

def judge_response(client: OpenAI, eval_prompt_template: str, question: str,
                   response: str, model: str = "gpt-4o-mini") -> int | None:
    """
    Score a response using the trait's eval_prompt template.
    Returns integer score 0-100, or None on failure/refusal.
    The eval_prompt uses {question} and {answer} placeholders.
    """
    prompt = eval_prompt_template.replace("{question}", question).replace("{answer}", response)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = completion.choices[0].message.content.strip()
        if raw.upper() == "REFUSAL":
            return None
        return int(raw)
    except Exception as e:
        print(f"    [WARN] Judge failed: {e}")
        return None


# ── Activation collection ─────────────────────────────────────────────────────

class MeanResponseHook:
    """Post-MLP residual stream hook. Collects mean activation over response tokens."""

    def __init__(self):
        self.activations: list[torch.Tensor] = []
        self._handle = None

    def hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.activations.append(hidden[0].detach().float().cpu())  # (seq_len, hidden_dim)

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None

    def clear(self):
        self.activations.clear()


def get_layer(model, layer_idx: int):
    for attr in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj[layer_idx]
    raise RuntimeError(f"Cannot find layer {layer_idx}.")


def generate_and_collect(
    model,
    tokenizer,
    hooks: dict[int, MeanResponseHook],
    system_prompt: str,
    question: str,
    max_new_tokens: int = 200,
) -> tuple[str, dict[int, torch.Tensor]]:
    """
    Generate a response and collect mean response activations per layer.
    Returns (response_text, {layer_idx: mean_act tensor (hidden_dim,)})
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    for h in hooks.values():
        h.clear()

    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode response text
    new_tokens = output_ids[0, prompt_ids.shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Collect mean response-token activations per layer.
    # Prefill pass: seq_len > 1. Decode passes: seq_len == 1.
    result = {}
    for layer_idx, hook in hooks.items():
        acts = hook.activations
        response_acts = [a[0] for a in acts if a.shape[0] == 1]
        if not response_acts:
            response_acts = [acts[0][-1]]  # fallback: last prefill token
        result[layer_idx] = torch.stack(response_acts).mean(dim=0)
        hook.clear()

    return response_text, result


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_trait_vector(
    llm_model,
    tokenizer,
    openai_client: OpenAI,
    trait_json: dict,
    trait_name: str,
    target_layer: int,
    all_layers: list[int],
    judge_model: str = "gpt-4o-mini",
    max_new_tokens: int = 200,
) -> tuple[torch.Tensor, dict]:
    """
    Extract a trait vector with judge filtering.
    Returns (unit_vector [hidden_dim], metadata_dict).
    """
    instructions = trait_json["instruction"]   # list of {"pos": ..., "neg": ...}
    questions    = trait_json["questions"]
    eval_prompt  = trait_json["eval_prompt"]

    hooks = {l: MeanResponseHook() for l in all_layers}
    for l, h in hooks.items():
        h.register(get_layer(llm_model, l))

    pos_acts: list[torch.Tensor] = []
    neg_acts: list[torch.Tensor] = []

    stats = {
        "trait": trait_name,
        "total_pos": 0, "kept_pos": 0,
        "total_neg": 0, "kept_neg": 0,
        "judge_failures": 0,
    }

    n_total = len(instructions) * len(questions) * 2
    done = 0

    try:
        for instr in instructions:
            for question in questions:

                # ── Positive pass ──────────────────────────────────────────
                response_text, layer_acts = generate_and_collect(
                    llm_model, tokenizer, hooks,
                    instr["pos"], question, max_new_tokens
                )
                done += 1
                stats["total_pos"] += 1

                score = judge_response(openai_client, eval_prompt, question,
                                       response_text, judge_model)
                if score is None:
                    stats["judge_failures"] += 1
                elif score > 50:
                    pos_acts.append(layer_acts[target_layer])
                    stats["kept_pos"] += 1

                # ── Negative pass ──────────────────────────────────────────
                response_text, layer_acts = generate_and_collect(
                    llm_model, tokenizer, hooks,
                    instr["neg"], question, max_new_tokens
                )
                done += 1
                stats["total_neg"] += 1

                score = judge_response(openai_client, eval_prompt, question,
                                       response_text, judge_model)
                if score is None:
                    stats["judge_failures"] += 1
                elif score < 50:
                    neg_acts.append(layer_acts[target_layer])
                    stats["kept_neg"] += 1

                if done % 20 == 0:
                    print(f"  [{done}/{n_total}] kept pos={stats['kept_pos']} "
                          f"neg={stats['kept_neg']}", flush=True)

    finally:
        for h in hooks.values():
            h.remove()

    print(f"\n  Filtering results:")
    print(f"    pos: {stats['kept_pos']}/{stats['total_pos']} kept (score > 50)")
    print(f"    neg: {stats['kept_neg']}/{stats['total_neg']} kept (score < 50)")
    print(f"    judge failures: {stats['judge_failures']}")

    if not pos_acts or not neg_acts:
        raise RuntimeError(
            f"No filtered responses for '{trait_name}': "
            f"pos={len(pos_acts)}, neg={len(neg_acts)}."
        )

    pos_mean = torch.stack(pos_acts).mean(dim=0)
    neg_mean = torch.stack(neg_acts).mean(dim=0)
    diff     = pos_mean - neg_mean
    vec      = F.normalize(diff, dim=0)

    print(f"  diff norm (pre-normalise): {diff.norm():.3f}")

    return vec, stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract trait vectors with LLM judge filtering for Llama-3.1-8B"
    )
    parser.add_argument("--trait_dir",      required=True, type=Path,
                        help="Path to data/extraction_questions/traits/")
    parser.add_argument("--model_id",       required=True,
                        help="HuggingFace model ID")
    parser.add_argument("--out_dir",        required=True, type=Path,
                        help="Output dir — saves {trait}.pt and {trait}_metadata.json")
    parser.add_argument("--layer",          type=int, default=16,
                        help="Target layer (default: 16 for Llama-3.1-8B)")
    parser.add_argument("--traits",         nargs="+", default=None,
                        help="Specific traits to run. Default: all in trait_dir.")
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--judge_model",    default="gpt-4o-mini")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: provide --openai_api_key or set OPENAI_API_KEY")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Discover trait files
    if args.traits:
        trait_files = [args.trait_dir / f"{t}.json" for t in args.traits]
    else:
        trait_files = sorted(args.trait_dir.glob("*.json"))

    if not trait_files:
        sys.exit(f"No JSON files found in {args.trait_dir}")

    print(f"Traits to process: {[f.stem for f in trait_files]}")
    print(f"Model:       {args.model_id}")
    print(f"Layer:       {args.layer}")
    print(f"Judge model: {args.judge_model}")
    print(f"Out dir:     {args.out_dir}\n")

    openai_client = OpenAI(api_key=api_key)

    # Load LLM once
    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    llm_model.eval()
    n_layers = llm_model.config.num_hidden_layers
    all_layers = list(range(n_layers))
    print(f"Model loaded. {n_layers} layers.\n")

    all_metadata = []

    for trait_file in trait_files:
        trait    = trait_file.stem
        out_vec  = args.out_dir / f"{trait}.pt"
        out_meta = args.out_dir / f"{trait}_metadata.json"

        if out_vec.exists():
            print(f"[SKIP] {trait}.pt already exists.")
            continue

        print(f"{'='*60}")
        print(f"Trait: {trait}")
        print(f"{'='*60}")

        trait_data = json.loads(trait_file.read_text())
        n_instr = len(trait_data["instruction"])
        n_q     = len(trait_data["questions"])
        print(f"  {n_instr} instruction pairs × {n_q} questions "
              f"= {n_instr * n_q * 2} total forward passes\n")

        vec, stats = extract_trait_vector(
            llm_model, tokenizer, openai_client,
            trait_json=trait_data,
            trait_name=trait,
            target_layer=args.layer,
            all_layers=all_layers,
            judge_model=args.judge_model,
            max_new_tokens=args.max_new_tokens,
        )

        torch.save(vec, out_vec)
        out_meta.write_text(json.dumps(stats, indent=2))
        all_metadata.append(stats)
        print(f"  Saved → {out_vec}  shape={vec.shape}\n")

    # Summary
    print("=" * 60)
    print("DONE. Summary:")
    for s in all_metadata:
        total = max(s['total_pos'] + s['total_neg'], 1)
        keep_rate = (s['kept_pos'] + s['kept_neg']) / total
        print(f"  {s['trait']:20s}  pos={s['kept_pos']}/{s['total_pos']}  "
              f"neg={s['kept_neg']}/{s['total_neg']}  "
              f"keep_rate={keep_rate:.0%}")


if __name__ == "__main__":
    main()
