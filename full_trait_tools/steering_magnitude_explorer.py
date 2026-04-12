#!/usr/bin/env python3
"""
steering_magnitude_explorer.py

Explores steering magnitudes for jailbreak suppression by printing raw
model outputs for manual inspection. Use this to pick the right alpha
before running the full evaluation with GPT judging.

Reconstructs the actual jailbreak-wrapped prompts from responses.jsonl
using the same jailbreak templates fetched from HarmBench GitHub.

Steering directions tested:
  - Full w (protective: -w, push away from jailbreak)
  - Full w (jailbreak: +w, causal validation)
  - PC1 direction (protective: +PC1, assistant_axis aligned at layer 16)
  - Top 3 protective traits: erudite, cautious, regulatory
  - Top 3 jailbreak traits steered protectively: naive, impulsive, confident

Usage:
  OPENAI_API_KEY=sk-... uv run full_trait_tools/steering_magnitude_explorer.py
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME       = "meta-llama/Llama-3.1-8B-Instruct"
LAYER_INDEX      = 16
RANDOM_SEED      = 42
MAX_NEW_TOKENS   = 200
N_PCA            = 4

PROTECTIVE_TRAITS   = ["erudite", "cautious", "regulatory"]
JAILBREAK_TRAITS    = ["naive", "impulsive", "confident"]
DEFAULT_ALPHAS      = [0.05, 0.10, 0.15, 0.20, 0.25]

JAILBREAKS_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/baselines/human_jailbreaks/jailbreaks.py"
)
SKIP_JAILBREAK_INDICES = {16, 44}


# ── Prompt reconstruction ──────────────────────────────────────────────────────

def load_jailbreak_templates(seed: int = 42, n_samples: int = 20) -> Dict[int, str]:
    print("  Fetching jailbreak templates from HarmBench GitHub...")
    with urllib.request.urlopen(JAILBREAKS_URL) as resp:
        source = resp.read().decode("utf-8")
    ns: dict = {}
    exec(source, ns)  # noqa: S102
    raw = ns["JAILBREAKS"]
    valid = [
        (i, jb) for i, jb in enumerate(raw)
        if i not in SKIP_JAILBREAK_INDICES and "{0}" in jb
    ]
    rng = random.Random(seed)
    sampled = rng.sample(valid, min(n_samples, len(valid)))
    print(f"  Loaded {len(sampled)} templates "
          f"(indices: {sorted(i for i, _ in sampled)})")
    return {idx: template for idx, template in sampled}


def reconstruct_prompt(row: dict, templates: Dict[int, str]) -> str:
    if row.get("attack_type") == "direct_request":
        return row["behavior_text"]
    jb_idx = row["jailbreak_idx"]
    template = templates.get(jb_idx)
    if template is None:
        print(f"  WARNING: template idx {jb_idx} not found, using raw behavior")
        return row["behavior_text"]
    try:
        return template.format(row["behavior_text"])
    except (KeyError, IndexError):
        return row["behavior_text"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_activations(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_trait_vector(vectors_dir: Path, trait: str, layer: int) -> np.ndarray:
    path = vectors_dir / f"{trait}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Trait vector not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    v = data["vector"][layer].float().numpy()
    return v / (np.linalg.norm(v) + 1e-12)


def load_hyperplane(path: Path) -> np.ndarray:
    data = torch.load(path, map_location="cpu", weights_only=False)
    v = data["vector"].float().numpy()
    return v / (np.linalg.norm(v) + 1e-12)


# ── Sample selection ───────────────────────────────────────────────────────────

def sample_jailbroken_pairs(
    classified_path: Path,
    n_samples: int,
    seed: int,
    templates: Dict[int, str],
) -> List[dict]:
    rows = load_jsonl(classified_path)
    rows = [r for r in rows if r.get("attack_type") == "human_jailbreak"
            and r.get("jailbroken", False)]

    by_behavior: Dict[str, List[dict]] = {}
    for r in rows:
        by_behavior.setdefault(r["behavior_id"], []).append(r)

    rng = random.Random(seed)
    sampled = []
    for behavior_id, behavior_rows in by_behavior.items():
        sampled.append(rng.choice(behavior_rows))

    rng.shuffle(sampled)
    sampled = sampled[:n_samples]

    for r in sampled:
        r["formatted_prompt"] = reconstruct_prompt(r, templates)

    return sampled


# ── Model helpers ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    return model, tokenizer


def get_residual_norm(model, tokenizer, layer_index, device, n: int = 6) -> float:
    prompts = [
        "Introduce yourself.",
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Describe your communication style.",
        "What makes a good leader?",
        "Explain gravity simply.",
    ][:n]
    norms = []

    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        norms.append(hidden.float().norm(dim=-1).mean().item())

    handle = model.model.layers[layer_index].register_forward_hook(hook_fn)
    try:
        for prompt in prompts:
            msgs = [{"role": "user", "content": prompt}]
            ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                model(input_ids=ids)
    finally:
        handle.remove()

    return float(np.mean(norms))


def sanitize_text(text: str) -> str:
    """Remove surrogate characters that break the fast tokenizer."""
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def generate_with_steering(
    model, tokenizer, prompt, unit_vec,
    layer_index, residual_norm, alpha,
    max_new_tokens, device,
) -> str:
    prompt = sanitize_text(prompt)
    msgs = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    handle = None
    if unit_vec is not None and alpha != 0.0:
        vec_t = torch.from_numpy(unit_vec).float()

        def hook_fn(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            delta = (alpha * residual_norm * vec_t).to(hidden.device, hidden.dtype)
            hidden = hidden + delta.view(1, 1, -1)
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        handle = model.model.layers[layer_index].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if handle is not None:
            handle.remove()

    new_ids = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── PCA ────────────────────────────────────────────────────────────────────────

def fit_pca(classified_path, activations_path, layer, n_pca):
    rows = load_jsonl(classified_path)
    rows = [r for r in rows if r.get("attack_type") == "human_jailbreak"]
    activations = load_activations(activations_path)
    layer_key = str(layer)
    X_list = []
    for row in rows:
        pid = row["pair_id"]
        if pid in activations and layer_key in activations[pid]:
            X_list.append(activations[pid][layer_key].float().numpy())
    X_all = np.stack(X_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
    pca.fit(X_scaled)
    print(f"  PCA fitted, var explained: "
          f"{[f'{100*v:.1f}%' for v in pca.explained_variance_ratio_]}")
    return pca, scaler


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classified_path", type=str,
        default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--activations_path", type=str,
        default="full_trait_output/harmbench_activations/activations.pt")
    parser.add_argument("--hyperplane_path", type=str,
        default="full_trait_output/harmbench_logreg/stable_hyperplane_layer16.pt")
    parser.add_argument("--trait_vectors_dir", type=str,
        default="full_trait_output/traits40_vectors/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--layer_index", type=int, default=LAYER_INDEX)
    parser.add_argument("--n_examples", type=int, default=5)
    parser.add_argument("--alphas", type=str,
        default=",".join(str(a) for a in DEFAULT_ALPHAS))
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]

    # ── Load templates + sample ────────────────────────────────────────────────
    print("Loading jailbreak templates...")
    templates = load_jailbreak_templates(seed=RANDOM_SEED, n_samples=20)

    print("Sampling jailbroken pairs...")
    pairs = sample_jailbroken_pairs(
        Path(args.classified_path), args.n_examples, args.seed, templates
    )
    print(f"  Sampled {len(pairs)} pairs")
    for i, p in enumerate(pairs):
        print(f"  [{i}] {p['behavior_id']}: {p['behavior_text'][:60]}...")
        print(f"       template_idx={p['jailbreak_idx']}, "
              f"prompt_len={len(p['formatted_prompt'])} chars")

    # ── Load model ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model on {device}...")
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    print("Calibrating residual norm...")
    residual_norm = get_residual_norm(model, tokenizer, args.layer_index, device)
    print(f"  Residual norm at layer {args.layer_index}: {residual_norm:.4f}")

    # ── Load steering vectors ──────────────────────────────────────────────────
    print("\nLoading steering vectors...")
    w_vec = load_hyperplane(Path(args.hyperplane_path))
    print("  Loaded w")

    print("  Fitting PCA for PC1 direction...")
    pca, _ = fit_pca(
        Path(args.classified_path), Path(args.activations_path),
        args.layer_index, N_PCA,
    )
    pc1_vec = pca.components_[0]
    pc1_vec = pc1_vec / (np.linalg.norm(pc1_vec) + 1e-12)

    trait_vecs = {}
    vectors_dir = Path(args.trait_vectors_dir)
    for trait in PROTECTIVE_TRAITS + JAILBREAK_TRAITS:
        try:
            trait_vecs[trait] = load_trait_vector(vectors_dir, trait, args.layer_index)
            print(f"  Loaded trait: {trait}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    # ── Conditions ─────────────────────────────────────────────────────────────
    conditions = [
        ("baseline",                     None,     0),
        ("w_protective (-w)",            w_vec,   -1),
        ("w_jailbreak (+w)",             w_vec,   +1),
        ("PC1_protective (+PC1)",        pc1_vec, +1),
    ]
    for trait in PROTECTIVE_TRAITS:
        if trait in trait_vecs:
            conditions.append((f"{trait} (+)", trait_vecs[trait], +1))
    for trait in JAILBREAK_TRAITS:
        if trait in trait_vecs:
            conditions.append((f"{trait} steered against (-)", trait_vecs[trait], -1))

    # ── Run ────────────────────────────────────────────────────────────────────
    sep = "█" * 100

    for pair_idx, pair in enumerate(pairs):
        prompt   = pair["formatted_prompt"]
        behavior = pair["behavior_text"]

        print(f"\n\n{sep}")
        print(f"EXAMPLE {pair_idx+1}/{len(pairs)}")
        print(f"Behavior : {behavior}")
        print(f"Template : {pair['jailbreak_idx']}")
        print(f"Prompt (first 300 chars):\n  {prompt[:300]}...")
        print(sep)

        for cond_name, unit_vec, sign in conditions:
            print(f"\n{'─'*100}")
            print(f"CONDITION: {cond_name}")
            print(f"{'─'*100}")

            if unit_vec is None:
                response = generate_with_steering(
                    model, tokenizer, prompt, None,
                    args.layer_index, residual_norm, 0.0,
                    args.max_new_tokens, device,
                )
                print(f"\n  [alpha=0.00 baseline]")
                print(f"  {response[:400]}")
            else:
                for alpha in alphas:
                    effective_alpha = sign * alpha
                    response = generate_with_steering(
                        model, tokenizer, prompt, unit_vec,
                        args.layer_index, residual_norm, effective_alpha,
                        args.max_new_tokens, device,
                    )
                    print(f"\n  [alpha={effective_alpha:+.2f}]")
                    print(f"  {response[:400]}")
                    if len(response) > 400:
                        print("  ...")

    print(f"\n\n{sep}")
    print("DONE. Review outputs above and pick your alpha.")
    print("Then run: OPENAI_API_KEY=sk-... uv run full_trait_tools/steering_robustness_eval.py --alpha <value>")
    print(sep)


if __name__ == "__main__":
    main()
