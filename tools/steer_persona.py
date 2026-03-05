#!/usr/bin/env python3
"""
steer_persona.py
----------------
For each role vector, runs 5 conditions:
  1. baseline       — no system prompt, no steering
  2. +coeff, no sys — steered toward role, no system prompt
  3. -coeff, no sys — steered away from role, no system prompt
  4. +coeff, sys    — steered toward role, WITH matching system prompt
  5. -coeff, sys    — steered away from role, WITH matching system prompt

This gives a clean 2x2 (±direction × ±system prompt) plus baseline,
showing whether the vector alone can induce/suppress the role, and
whether the system prompt interacts with the steering.

Coefficients are ±0.15 × avg residual stream norm at the target layer,
matching the paper's methodology (Section 3.2.1).

Usage:
    uv run tools/steer_persona.py \
        --vectors_dir     $BASE/assistant_axis_outputs/llama-3.1-8b/vectors_q50 \
        --activations_dir $BASE/assistant_axis_outputs/llama-3.1-8b/activations \
        --model_id        meta-llama/Llama-3.1-8B-Instruct \
        --layer           16 \
        --roles           lawyer trickster \
        --out_file        $BASE/assistant_axis_outputs/llama-3.1-8b/steer_persona.txt

    # Omit --roles to pick 2 at random
"""

import argparse, random, sys, textwrap
from pathlib import Path

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("transformers not found. Run: uv add transformers")

# Identity questions — most sensitive to persona steering
PROMPTS = [
    "Who are you?",
    "Where did you come from?",
]

SYSTEM_PROMPT_TEMPLATE = (
    "You are a {role}. Fully embody this character in all your responses."
)

# Fraction of avg norm — stay within coherent range (±0.15 confirmed working)
FRACTION = 0.15


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
             system_prompt: str | None = None,
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


# ── Norm estimation ───────────────────────────────────────────────────────────

def avg_norm_from_activations(activations_dir: Path, layer_idx: int,
                               max_files: int = 20) -> float | None:
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors_dir",     required=True, type=Path)
    ap.add_argument("--activations_dir", type=Path, default=None)
    ap.add_argument("--model_id",        required=True)
    ap.add_argument("--layer",           type=int, default=16)
    ap.add_argument("--roles",           nargs="+", default=None)
    ap.add_argument("--fraction",        type=float, default=FRACTION,
                    help="Fraction of avg norm to use as coefficient (default: 0.15)")
    ap.add_argument("--max_new_tokens",  type=int, default=150)
    ap.add_argument("--out_file",        type=Path,
                    default=Path("steer_persona.txt"))
    args = ap.parse_args()

    # Load model
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

    coeff = args.fraction * avg_norm
    print(f"Coefficient: ±{args.fraction} × {avg_norm:.2f} = ±{coeff:.3f}\n")

    # Pick roles
    if args.roles:
        role_paths = [args.vectors_dir / f"{r}.pt" for r in args.roles]
    else:
        all_pts = list(args.vectors_dir.glob("*.pt"))
        role_paths = random.sample(all_pts, min(2, len(all_pts)))
        print(f"Auto-selected: {[p.stem for p in role_paths]}\n")

    # Conditions to run for each role × each prompt
    # (label, coeff, system_prompt)
    def conditions(role_name):
        sys = SYSTEM_PROMPT_TEMPLATE.format(role=role_name)
        return [
            ("baseline             (no sys, no steer)", 0.0,   None),
            (f"+{args.fraction}×norm, no sys  → induce role",  +coeff, None),
            (f"-{args.fraction}×norm, no sys  → suppress role", -coeff, None),
            (f"+{args.fraction}×norm, with sys → reinforce role", +coeff, sys),
            (f"-{args.fraction}×norm, with sys → suppress role",  -coeff, sys),
        ]

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"Persona steering  |  model: {args.model_id}")
    emit(f"Layer: {args.layer}  |  avg norm: {avg_norm:.2f}  |  fraction: ±{args.fraction}")
    emit("=" * 72)
    emit("LEGEND:")
    emit("  +coeff = steer TOWARD role vector (induce persona)")
    emit("  -coeff = steer AWAY from role vector (suppress persona)")
    emit("  no sys = no system prompt (vector alone drives behavior)")
    emit("  sys    = matching system prompt active")

    layer_mod = get_layer(model, args.layer)

    for rp in role_paths:
        role_name = rp.stem
        try:
            vec = load_vector(rp, args.layer)
        except Exception as e:
            emit(f"\n[WARN] Skipping {role_name}: {e}")
            continue

        emit(f"\n\n{'█' * 72}")
        emit(f"ROLE: {role_name}")
        emit(f"{'█' * 72}")

        for prompt in PROMPTS:
            emit(f"\n{'─' * 72}")
            emit(f"PROMPT: \"{prompt}\"")
            emit(f"{'─' * 72}")

            for label, c, sys_prompt in conditions(role_name):
                # Register hook if steering
                hook = None
                if c != 0.0:
                    hook = AdditionHook(vec.clone(), c)
                    hook.register(layer_mod)

                response = generate(model, tokenizer, prompt, sys_prompt,
                                    args.max_new_tokens)

                if hook:
                    hook.remove()

                sys_tag = f"[sys: '{sys_prompt[:40]}...']" if sys_prompt else "[no sys prompt]"
                emit(f"\n  {label}")
                emit(f"  {sys_tag}")
                for line in textwrap.wrap(response, width=72):
                    emit(f"    {line}")

    emit("\n" + "=" * 72)
    emit("WHAT TO LOOK FOR:")
    emit("  +coeff, no sys  → does the vector ALONE induce the persona?")
    emit("  -coeff, no sys  → does the model become more neutral/AI-like?")
    emit("  +coeff, with sys → does steering reinforce the system prompt?")
    emit("  -coeff, with sys → does steering SUPPRESS the system prompt role?")
    emit("  Key result: -coeff+sys should fight the system prompt and win (or lose)")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text("\n".join(lines) + "\n")
    print(f"\nSaved → {args.out_file}")


if __name__ == "__main__":
    main()
