#!/usr/bin/env python3
"""
verify_steering.py
------------------
Steers the model along the Assistant Axis (and optionally role vectors)
at multiple coefficients and saves responses for manual inspection.

Coefficients are scaled as fractions of the average residual stream norm
at the target layer, matching the paper's methodology (Section 3.2.1).
The paper's Figure 4 x-axis uses this same "fraction of avg norm" scale.

For Llama 3.1 8B (32 layers), the middle layer is 16.

The axis is a raw tensor of shape (hidden_size,) — one vector for the
chosen layer, already stored that way in axis_q50.pt.

Role vectors from vectors_q50/ may be stored as either a raw tensor
(hidden_size,) or a dict keyed by layer index — both are handled.

Usage:
    uv run tools/verify_steering.py \
        --axis_path       $BASE/assistant_axis_outputs/llama-3.1-8b/axis_q50.pt \
        --vectors_dir     $BASE/assistant_axis_outputs/llama-3.1-8b/vectors_q50 \
        --activations_dir $BASE/assistant_axis_outputs/llama-3.1-8b/activations \
        --model_id        meta-llama/Llama-3.1-8B-Instruct \
        --layer           16 \
        --roles           pirate villain \
        --out_file        $BASE/assistant_axis_outputs/llama-3.1-8b/verify_steering.txt
"""

import argparse, json, random, sys, textwrap
from pathlib import Path

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("transformers not found. Run: uv add transformers")

TEST_PROMPT = "What causes the seasons on Earth?"

# Fractions of avg norm to test — matches paper Figure 4 x-axis range for Llama
FRACTIONS = [0.0, +0.05, +0.15, -0.05, -0.15]
FRAC_LABELS = [
    "baseline",
    "+0.05 × norm  (mild → more Assistant-like)",
    "+0.15 × norm  (strong → more Assistant-like)",
    "-0.05 × norm  (mild → drift from Assistant)",
    "-0.15 × norm  (strong → drift from Assistant)",
]


# ── Steering hook ─────────────────────────────────────────────────────────────

class AdditionHook:
    """Adds coeff * unit_vector to the residual stream at a single layer."""
    def __init__(self, vector: torch.Tensor, coeff: float):
        # Normalize to unit vector, then scale by coeff at call time
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
    raise RuntimeError(
        f"Cannot locate layer {idx} in model. "
        "Inspect model architecture and update get_layer()."
    )


# ── Generation ────────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    messages = [{"role": "user", "content": prompt}]
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
                               max_files: int = 20) -> float | None:
    """
    Estimate avg residual stream norm from saved .pt activation files.
    Each file is expected to be either:
      - a dict {layer_idx: tensor of shape (hidden_size,)}
      - a raw tensor of shape (hidden_size,)
    """
    pts = list(activations_dir.glob("*.pt"))
    if not pts:
        return None
    pts = random.sample(pts, min(max_files, len(pts)))
    norms = []
    for p in pts:
        try:
            data = torch.load(p, map_location="cpu", weights_only=True)
            if isinstance(data, dict):
                t = data.get(layer_idx, data.get(str(layer_idx)))
            else:
                t = data
            if t is not None:
                norms.append(t.float().norm().item())
        except Exception:
            continue
    return float(sum(norms) / len(norms)) if norms else None


def avg_norm_from_forward_passes(model, tokenizer, layer_idx: int) -> float:
    """Fallback: run a few prompts and capture the norm live."""
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


# ── Vector loading ────────────────────────────────────────────────────────────

def load_raw_vector(path: Path, layer_idx: int) -> torch.Tensor:
    """
    Load a steering vector. Handles:
      - raw tensor (axis_q50.pt is stored this way)
      - dict keyed by int or str layer index (some role vectors may be stored this way)
    """
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        for key in [layer_idx, str(layer_idx)]:
            if key in data:
                return data[key].float()
        raise KeyError(
            f"Layer {layer_idx} not found in {path}. "
            f"Available keys: {list(data.keys())[:10]}"
        )
    # Raw tensor — assumed to already be for the correct layer
    return data.float()


# ── Core test ─────────────────────────────────────────────────────────────────

def run_test(model, tokenizer, layer_idx: int, vector: torch.Tensor,
             avg_norm: float, label: str, prompt: str) -> list[dict]:
    """Runs baseline + 4 steered generations. Returns list of result dicts."""
    layer_mod = get_layer(model, layer_idx)
    results = []

    for frac, flabel in zip(FRACTIONS, FRAC_LABELS):
        coeff = frac * avg_norm

        hook = None
        if coeff != 0.0:
            hook = AdditionHook(vector.clone(), coeff)
            hook.register(layer_mod)

        response = generate(model, tokenizer, prompt)

        if hook:
            hook.remove()

        results.append({
            "fraction": frac,
            "coeff": round(coeff, 4),
            "label": flabel,
            "response": response,
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis_path",       required=True, type=Path,
                    help="Path to axis_q50.pt (raw tensor, shape: hidden_size)")
    ap.add_argument("--vectors_dir",     required=True, type=Path,
                    help="Dir containing per-role .pt vector files")
    ap.add_argument("--activations_dir", type=Path, default=None,
                    help="Dir with saved activation .pt files for norm estimation")
    ap.add_argument("--model_id",        required=True,
                    help="HuggingFace model ID")
    ap.add_argument("--layer",           type=int, default=16,
                    help="Layer to steer (middle layer for 8B = 16). Default: 16")
    ap.add_argument("--roles",           nargs="+", default=None,
                    help="Role vector file stems to test (e.g. pirate villain). "
                         "Omit to pick 2 at random.")
    ap.add_argument("--prompt",          default=TEST_PROMPT)
    ap.add_argument("--max_new_tokens",  type=int, default=200)
    ap.add_argument("--out_file",        type=Path,
                    default=Path("verify_steering.txt"))
    args = ap.parse_args()

    # Load model
    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map="auto"
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

    # Load axis (raw tensor)
    axis = load_raw_vector(args.axis_path, args.layer)
    print(f"Axis loaded: shape={axis.shape}\n")

    # Pick role vectors
    if args.roles:
        role_paths = [args.vectors_dir / f"{r}.pt" for r in args.roles]
    else:
        all_pts = list(args.vectors_dir.glob("*.pt"))
        role_paths = random.sample(all_pts, min(2, len(all_pts))) if all_pts else []
        if role_paths:
            print(f"Auto-selected role vectors: {[p.stem for p in role_paths]}\n")

    # Run tests and collect output
    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"Steering verification  |  model: {args.model_id}")
    emit(f"Layer: {args.layer}  |  avg norm: {avg_norm:.2f}")
    emit(f"Prompt: {args.prompt}")
    emit("=" * 72)

    all_vectors = [("assistant_axis", axis)] + [
        (p.stem, load_raw_vector(p, args.layer))
        for p in role_paths
        if p.exists()
    ]

    for label, vec in all_vectors:
        emit(f"\n{'─' * 72}")
        emit(f"VECTOR: {label}")
        emit(f"{'─' * 72}")

        results = run_test(model, tokenizer, args.layer, vec, avg_norm,
                           label, args.prompt)

        for r in results:
            emit(f"\n  [{r['label']}]  (coeff={r['coeff']})")
            # Wrap response text for readability
            for line in textwrap.wrap(r["response"], width=76):
                emit(f"    {line}")

    emit("\n" + "=" * 72)
    emit("WHAT TO LOOK FOR:")
    emit("  +coeff → more Assistant-like: clear, helpful, grounded answer")
    emit("  -coeff → drift from Assistant: theatrical, mystical, or adopts a persona")
    emit("  All 5 identical → hook not attaching; check get_layer() for your arch")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text("\n".join(lines) + "\n")
    print(f"\nSaved → {args.out_file}")


if __name__ == "__main__":
    main()
