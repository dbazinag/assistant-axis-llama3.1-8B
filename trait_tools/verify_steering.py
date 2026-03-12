#!/usr/bin/env python3
# Paper-aligned trait steering verifier for Llama-style models.
# Key change: steer at a single layer by default, scale by calibrated residual norm, and pass attention_mask.

import argparse
import random
import sys
import textwrap
from pathlib import Path
from typing import Iterable

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("transformers not found. Run: uv add transformers")

# More open-ended prompts make persona/style shifts easier to see.
TRAIT_PROMPTS = {
    "verbose": [
        "Describe your communication style.",
        "Explain how to boil an egg.",
        "Write advice for a young student starting university.",
        "Tell me what makes a good leader.",
    ],
    "bombastic": [
        "Introduce yourself dramatically.",
        "Write a short motivational speech.",
        "Describe ambition in one paragraph.",
        "Explain how to make toast as if it were a grand quest.",
    ],
    "chaotic": [
        "Give me a neat step-by-step plan to organize my desk.",
        "Explain how to prepare for a job interview.",
        "Describe your communication style.",
        "Write a short answer about time management.",
    ],
    "pedantic": [
        "What is 2+2?",
        "Explain what a triangle is.",
        "Define a rectangle.",
        "What time is it if I leave at 3 and arrive 2 hours later?",
    ],
    "casual": [
        "Tell me how to make toast.",
        "What makes a good first impression?",
        "How do I calm down before an interview?",
        "Explain budgeting like you're talking to a friend.",
    ],
    "edgy": [
        "Describe modern society in one paragraph.",
        "What do you think about ambition?",
        "Introduce yourself in one paragraph.",
        "Write a short monologue about success.",
    ],
}

DEFAULT_PROMPTS = [
    "Describe your communication style.",
    "Write advice for a young student starting university.",
    "Introduce yourself in one paragraph.",
]

# Keep the default sweep in the useful range.
DEFAULT_FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.00, -0.25, -0.50, -0.75, -1.00]


def load_full_vector(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        if "vector" not in data:
            raise KeyError(f"Cannot find 'vector' key in {path}. Keys: {list(data.keys())}")
        t = data["vector"].float()
    else:
        t = data.float()

    if t.ndim != 2:
        raise ValueError(f"Expected 2D tensor in {path}, got shape {tuple(t.shape)}")
    return t


def get_layers(model):
    for attr in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        ok = True
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok:
            return obj
    raise RuntimeError("Cannot locate transformer layers.")


def format_chat(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    return enc


def move_batch_to_device(batch: dict, device):
    return {k: v.to(device) for k, v in batch.items()}


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    enc = format_chat(tokenizer, prompt)

    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"])

    enc = move_batch_to_device(enc, model.device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = enc["input_ids"].shape[1]
    text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    return text.strip()


def pick_layer_indices(
    model,
    full_vector: torch.Tensor,
    mode: str,
    top_k: int,
    explicit: str | None,
) -> list[int]:
    n_layers = len(get_layers(model))
    norms = full_vector.norm(dim=-1)

    if mode == "middle":
        return [n_layers // 2]
    if mode == "strongest":
        return [int(torch.argmax(norms).item())]
    if mode == "late":
        return [max(0, n_layers - 3)]
    if mode == "topk":
        k = max(1, min(top_k, n_layers))
        idxs = torch.topk(norms, k=k).indices.tolist()
        return sorted(idxs)
    if mode == "explicit":
        if not explicit:
            raise ValueError("--explicit_layers required when --layer_mode explicit")
        out = []
        for part in explicit.split(","):
            idx = int(part.strip())
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"Layer index {idx} out of range for model with {n_layers} layers")
            out.append(idx)
        if not out:
            raise ValueError("No explicit layers parsed")
        return out

    raise ValueError(f"Unknown layer mode: {mode}")


class ResidualNormCalibrator:
    # Measures mean L2 norm of the chosen layer outputs on a prompt set.
    # This approximates the paper's residual-norm scaling without needing a large external dataset.
    def __init__(self, layer_indices: list[int]):
        self.layer_indices = list(layer_indices)
        self.layer_norm_sums = {i: 0.0 for i in self.layer_indices}
        self.layer_counts = {i: 0 for i in self.layer_indices}
        self.handles = []

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output  # [B, T, D]
            with torch.no_grad():
                norms = hidden.float().norm(dim=-1)  # [B, T]
                self.layer_norm_sums[layer_idx] += norms.mean().item()
                self.layer_counts[layer_idx] += 1
        return hook

    def register(self, model):
        layers = get_layers(model)
        for idx in self.layer_indices:
            h = layers[idx].register_forward_hook(self._make_hook(idx))
            self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def calibrate(
        self,
        model,
        tokenizer,
        prompts: Iterable[str],
    ) -> dict[int, float]:
        self.register(model)
        try:
            with torch.no_grad():
                for prompt in prompts:
                    enc = format_chat(tokenizer, prompt)
                    if "attention_mask" not in enc:
                        enc["attention_mask"] = torch.ones_like(enc["input_ids"])
                    enc = move_batch_to_device(enc, model.device)
                    _ = model(**enc)
        finally:
            self.remove()

        out = {}
        for idx in self.layer_indices:
            count = self.layer_counts[idx]
            if count == 0:
                raise RuntimeError(f"Failed to calibrate residual norm for layer {idx}")
            out[idx] = self.layer_norm_sums[idx] / count
        return out


class MultiLayerAdditionHook:
    # Adds scaled trait directions at chosen layer outputs (paper-aligned simple additive steering).
    def __init__(
        self,
        full_vector: torch.Tensor,
        layer_indices: list[int],
        layer_coeffs: dict[int, float],
        divide_across_layers: bool = True,
    ):
        self.full_vector = full_vector.float()
        self.layer_indices = list(layer_indices)
        self.layer_coeffs = dict(layer_coeffs)
        self.divide_across_layers = divide_across_layers
        self.handles = []

    def _make_hook(self, layer_idx: int):
        vec = self.full_vector[layer_idx]
        unit = vec / (vec.norm() + 1e-8)

        coeff = float(self.layer_coeffs[layer_idx])
        if self.divide_across_layers:
            coeff = coeff / max(1, len(self.layer_indices))

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            delta = (coeff * unit).to(hidden.device, hidden.dtype)
            hidden = hidden + delta.view(1, 1, -1)
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        return hook

    def register(self, model):
        layers = get_layers(model)
        for idx in self.layer_indices:
            h = layers[idx].register_forward_hook(self._make_hook(idx))
            self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def run_test(
    model,
    tokenizer,
    full_vector: torch.Tensor,
    layer_indices: list[int],
    layer_residual_norms: dict[int, float],
    fractions: list[float],
    prompt: str,
    max_new_tokens: int,
) -> list[dict]:
    results = []

    for frac in fractions:
        layer_coeffs = {idx: frac * layer_residual_norms[idx] for idx in layer_indices}
        hook = None
        try:
            if frac != 0.0:
                hook = MultiLayerAdditionHook(
                    full_vector=full_vector,
                    layer_indices=layer_indices,
                    layer_coeffs=layer_coeffs,
                    divide_across_layers=True,
                )
                hook.register(model)

            response = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        finally:
            if hook is not None:
                hook.remove()

        results.append(
            {
                "fraction": frac,
                "coeffs": {k: round(v, 4) for k, v in layer_coeffs.items()},
                "response": response,
            }
        )
    return results


def layer_norm_report(full_vector: torch.Tensor) -> list[str]:
    norms = full_vector.norm(dim=-1).tolist()
    return [f"layer {i:2d}: {n:.4f}" for i, n in enumerate(norms)]


def parse_fractions(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors_dir", required=True, type=Path)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--traits", nargs="+", default=["verbose", "bombastic", "chaotic", "pedantic"])
    ap.add_argument("--layer_mode", type=str, default="strongest", choices=["middle", "strongest", "late", "topk", "explicit"])
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--explicit_layers", type=str, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--fractions", type=str, default=",".join(str(x) for x in DEFAULT_FRACTIONS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_file", type=Path, default=Path("trait_outputs/verify_trait_steering_paper_aligned.txt"))
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.\n")

    trait_paths = [args.vectors_dir / f"{t}.pt" for t in args.traits]
    missing = [str(p) for p in trait_paths if not p.exists()]
    if missing:
        sys.exit(f"Missing trait vector files: {missing}")

    fractions = parse_fractions(args.fractions)
    calibration_prompts = []
    for t in args.traits:
        calibration_prompts.extend(TRAIT_PROMPTS.get(t, DEFAULT_PROMPTS))
    calibration_prompts = calibration_prompts[: min(len(calibration_prompts), 16)]

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"Trait steering verification  |  model: {args.model_id}")
    emit(f"Fractions tested: {fractions}")
    emit(f"Layer mode: {args.layer_mode}")
    emit("=" * 110)

    for tp in trait_paths:
        trait_name = tp.stem
        full_vector = load_full_vector(tp)
        prompts = TRAIT_PROMPTS.get(trait_name, DEFAULT_PROMPTS)
        layer_indices = pick_layer_indices(
            model=model,
            full_vector=full_vector,
            mode=args.layer_mode,
            top_k=args.top_k,
            explicit=args.explicit_layers,
        )

        calibrator = ResidualNormCalibrator(layer_indices)
        layer_residual_norms = calibrator.calibrate(model, tokenizer, calibration_prompts)

        emit(f"\n\n{'█' * 110}")
        emit(f"TRAIT VECTOR: {trait_name.upper()}")
        emit(f"Vector shape: {tuple(full_vector.shape)}")
        emit(f"Selected layers: {layer_indices}")
        emit("Top per-layer vector norms:")
        top_pairs = sorted(
            [(i, full_vector[i].norm().item()) for i in range(full_vector.shape[0])],
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        emit("  " + ", ".join([f"L{i}={n:.3f}" for i, n in top_pairs]))
        emit("Calibrated residual norms at selected layers:")
        emit("  " + ", ".join([f"L{i}={layer_residual_norms[i]:.3f}" for i in layer_indices]))
        emit("█" * 110)

        emit("Per-layer vector norm report:")
        for row in layer_norm_report(full_vector):
            emit("  " + row)

        for prompt in prompts:
            emit(f"\n{'─' * 110}")
            emit(f'PROMPT: "{prompt}"')
            emit(f"{'─' * 110}")

            results = run_test(
                model=model,
                tokenizer=tokenizer,
                full_vector=full_vector,
                layer_indices=layer_indices,
                layer_residual_norms=layer_residual_norms,
                fractions=fractions,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )

            for r in results:
                emit(f"\n  [frac={r['fraction']:+.2f}]  layer_coeffs={r['coeffs']}")
                for line in textwrap.wrap(r["response"], width=104):
                    emit(f"    {line}")

    emit("\n" + "=" * 110)
    emit("NOTES:")
    emit("  This version is paper-aligned: additive steering at layer output, every token, scaled by residual norm.")
    emit("  Defaults use a single strongest layer for visibility; use --layer_mode middle for the closest paper-style setting.")
    emit("  If outputs collapse, reduce fractions or use a single layer instead of topk.")
    emit("  If steering is too subtle, try --layer_mode strongest with fractions up to ±1.5 before going larger.")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text("\n".join(lines) + "\n")
    print(f"\nSaved → {args.out_file}")


if __name__ == "__main__":
    main()