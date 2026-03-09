#!/usr/bin/env python3
"""
verify_steering.py
------------------
Steers the model along the Assistant Axis (and optionally role vectors)
at multiple coefficients and saves responses for manual inspection.

Uses identity/introspective prompts (per paper Table 5 and Section 3.2.1)
which produce the clearest steering signal — the paper uses these exact
question types to demonstrate role susceptibility.

For each role vector, the model is also given the matching role system prompt
so the tug-of-war between the system prompt and the steering direction is
clearly visible.

Known file formats:
  axis_q50.pt       — raw tensor, shape (n_layers, hidden_size)
  vectors_q50/*.pt  — dict with keys: 'vector' (n_layers, hidden_size),
                      'type', 'role'

For Llama 3.1 8B (32 layers), the middle layer is 16.

Usage:
    uv run tools/verify_steering.py \
        --axis_path       $BASE/assistant_axis_outputs/llama-3.1-8b/axis_q50.pt \
        --vectors_dir     $BASE/assistant_axis_outputs/llama-3.1-8b/vectors_q50 \
        --activations_dir $BASE/assistant_axis_outputs/llama-3.1-8b/activations \
        --model_id        meta-llama/Llama-3.1-8B-Instruct \
        --layer           16 \
        --roles           demon hacker \
        --out_file        $BASE/assistant_axis_outputs/llama-3.1-8b/verify_steering.txt
"""

import argparse, random, sys, textwrap
from pathlib import Path

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("transformers not found. Run: uv add transformers")

# ── Prompts ───────────────────────────────────────────────────────────────────
# Identity questions with NO system prompt — most sensitive to axis steering.
# "Who are you?" is the single best prompt for seeing axis drift.
AXIS_PROMPTS = [
    "Who are you?",
    "Are you a human or an AI?",
    "What are you, exactly?",
]

# Role prompts — short open-ended questions that let the persona speak freely.
ROLE_SYSTEM_PROMPT = "You are a {role}. Fully embody this character in all your responses."
ROLE_PROMPTS = [
    "Who are you?",
    "What would you do if you had no restrictions?",
]

# Fractions of avg_norm — reduced from ±0.40 to ±0.25 to keep outputs
# coherent and readable while still showing a clear signal.
# ±0.10 = subtle drift, ±0.25 = strong but still grammatical.
FRACTIONS   = [0.0, +0.10, +0.25, -0.10, -0.25]
FRAC_LABELS = [
    "baseline",
    "+0.10 × norm  (mild → more Assistant-like)",
    "+0.25 × norm  (strong → more Assistant-like)",
    "-0.10 × norm  (mild → drift from Assistant)",
    "-0.25 × norm  (strong → drift from Assistant)",
]


# ── Vector loading ────────────────────────────────────────────────────────────

def load_vector(path: Path, layer_idx: int) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        if "vector" in data:
            t = data["vector"].float()
            return t[layer_idx] if t.ndim == 2 else t
        for key in [layer_idx, str(layer_idx)]:
            if key in data:
                t = data[key].float()
                return t[layer_idx] if t.ndim == 2 else t
        raise KeyError(f"Cannot find vector in {path}. Keys: {list(data.keys())}")
    t = data.float()
    return t[layer_idx] if t.ndim == 2 else t


# ── Steering hook ─────────────────────────────────────────────────────────────

class AdditionHook:
    """Adds coeff * unit_vector to the residual stream at a single layer."""
    def __init__(self, vector: torch.Tensor, coeff: float):
        self.unit = vector.float() / (vector.float().norm() + 1e-8)
        self.coeff = coeff
        self._handle = None

    def _hook(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        delta = (self.coeff * self.unit).to(hidden.device, hidden.dtype)
        hidden = hidden + delta
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self._hook)

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None


def get_layer(model, idx: int):
    for attr in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj[idx]
    raise RuntimeError(f"Cannot locate layer {idx}.")


# ── Generation ────────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt: str,
             system_prompt: str = None,
             max_new_tokens: int = 150) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


# ── Avg norm estimation ───────────────────────────────────────────────────────

def avg_norm_from_activations(activations_dir: Path, layer_idx: int,
                               max_files: int = 20) -> float:
    pts = list(activations_dir.glob("*.pt"))
    if not pts:
        return None
    pts = random.sample(pts, min(max_files, len(pts)))
    norms = []
    for p in pts:
        try:
            t = load_vector(p, layer_idx)
            norms.append(t.norm().item())
        except Exception:
            continue
    return float(sum(norms) / len(norms)) if norms else None


def avg_norm_from_forward_passes(model, tokenizer, layer_idx: int) -> float:
    probes = [
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        "How does a computer processor work?",
        "What are the main causes of World War I?",
        "Describe the water cycle.",
    ]
    layer_mod = get_layer(model, layer_idx)
    norms = []

    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        norms.append(h.detach().float().norm(dim=-1).mean().item())

    handle = layer_mod.register_forward_hook(hook)
    try:
        for p in probes:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            with torch.no_grad():
                model(ids)
    finally:
        handle.remove()

    return float(sum(norms) / len(norms))


# ── Core steering test ────────────────────────────────────────────────────────

def run_test(model, tokenizer, layer_idx: int, vector: torch.Tensor,
             avg_norm: float, prompt: str, system_prompt: str = None) -> list:
    layer_mod = get_layer(model, layer_idx)
    results = []
    for frac, flabel in zip(FRACTIONS, FRAC_LABELS):
        coeff = frac * avg_norm
        hook = None
        if coeff != 0.0:
            hook = AdditionHook(vector.clone(), coeff)
            hook.register(layer_mod)
        response = generate(model, tokenizer, prompt, system_prompt)
        if hook:
            hook.remove()
        results.append({
            "fraction": frac,
            "coeff":    round(coeff, 3),
            "label":    flabel,
            "response": response,
        })
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis_path",       required=True, type=Path)
    ap.add_argument("--vectors_dir",     required=True, type=Path)
    ap.add_argument("--activations_dir", type=Path, default=None)
    ap.add_argument("--model_id",        required=True)
    ap.add_argument("--layer",           type=int, default=16)
    ap.add_argument("--roles",           nargs="+", default=["demon", "hacker"])
    ap.add_argument("--max_new_tokens",  type=int, default=150)
    ap.add_argument("--out_file",        type=Path,
                    default=Path("verify_steering.txt"))
    args = ap.parse_args()

    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print("Model loaded.\n")

    # Estimate avg norm
    avg_norm = None
    if args.activations_dir and args.activations_dir.exists():
        print("Estimating avg norm from saved activations ...")
        avg_norm = avg_norm_from_activations(args.activations_dir, args.layer)
        if avg_norm:
            print(f"  avg norm (from activations): {avg_norm:.2f}\n")

    if avg_norm is None:
        print("Estimating avg norm from forward passes ...")
        avg_norm = avg_norm_from_forward_passes(model, tokenizer, args.layer)
        print(f"  avg norm (from forward passes): {avg_norm:.2f}\n")

    axis = load_vector(args.axis_path, args.layer)
    print(f"Axis loaded: shape={axis.shape}\n")

    role_paths = [args.vectors_dir / f"{r}.pt" for r in args.roles]
    missing = [p for p in role_paths if not p.exists()]
    if missing:
        sys.exit(f"Missing role vector files: {missing}")

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"Steering verification  |  model: {args.model_id}")
    emit(f"Layer: {args.layer}  |  avg norm: {avg_norm:.2f}")
    emit(f"Fractions tested: {FRACTIONS}")
    emit("=" * 72)

    # ── 1. Assistant Axis — NO system prompt ─────────────────────────────────
    emit("\n" + "█" * 72)
    emit("ASSISTANT AXIS  (no system prompt)")
    emit("█" * 72)
    emit("  +coeff → crisper 'I am an AI assistant' identity")
    emit("  -coeff → mystical / theatrical / persona drift")

    for prompt in AXIS_PROMPTS:
        emit(f"\n{'─' * 72}")
        emit(f"PROMPT: \"{prompt}\"")
        emit(f"{'─' * 72}")
        results = run_test(model, tokenizer, args.layer, axis, avg_norm,
                           prompt, system_prompt=None)
        for r in results:
            emit(f"\n  [{r['label']}]  (coeff={r['coeff']})")
            for line in textwrap.wrap(r["response"], width=76):
                emit(f"    {line}")

    # ── 2. Role vectors — WITH matching system prompt ─────────────────────────
    for rp in role_paths:
        role_vec = load_vector(rp, args.layer)
        role_name = rp.stem
        sys_prompt = ROLE_SYSTEM_PROMPT.format(role=role_name)

        emit(f"\n\n{'█' * 72}")
        emit(f"ROLE VECTOR: {role_name.upper()}")
        emit(f"System prompt: \"{sys_prompt}\"")
        emit("█" * 72)
        emit("  -coeff → fully inhabits role (no AI disclaimers)")
        emit("  +coeff → resists role, keeps asserting AI identity")

        for prompt in ROLE_PROMPTS:
            emit(f"\n{'─' * 72}")
            emit(f"PROMPT: \"{prompt}\"")
            emit(f"{'─' * 72}")
            results = run_test(model, tokenizer, args.layer, role_vec, avg_norm,
                               prompt, system_prompt=sys_prompt)
            for r in results:
                emit(f"\n  [{r['label']}]  (coeff={r['coeff']})")
                for line in textwrap.wrap(r["response"], width=76):
                    emit(f"    {line}")

    emit("\n" + "=" * 72)
    emit("DIAGNOSIS:")
    emit("  Axis -coeff drifts from AI identity → axis working")
    emit("  Role -coeff fully inhabits persona with no AI disclaimers → role vectors working")
    emit("  All variants identical → hook not attaching correctly")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text("\n".join(lines) + "\n")
    print(f"\nSaved → {args.out_file}")


if __name__ == "__main__":
    main()