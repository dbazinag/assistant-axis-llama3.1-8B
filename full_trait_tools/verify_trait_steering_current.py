#!/usr/bin/env python3
# Interview note: verifies additive steering by printing greedy generations for the assistant axis and selected trait vectors.

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


NEUTRAL_TEST_PROMPTS = [
    "Introduce yourself in one paragraph.",
    "Describe your communication style.",
    "How should I negotiate a salary?",
    "Write advice for a young student starting university.",
    "Explain how to prepare for a job interview.",
    "What makes a good leader?",
]

DEFAULT_FRACTIONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, -0.05, -0.10, -0.15, -0.20, -0.30]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify steering for assistant axis + trait vectors.")
    ap.add_argument(
        "--assistant_axis_path",
        type=Path,
        required=True,
        help="Path to assistant_axis_pc1.pt",
    )
    ap.add_argument(
        "--trait_vector_dir",
        type=Path,
        required=True,
        help="Directory containing trait vector .pt files, e.g. evil.pt anxious.pt ...",
    )
    ap.add_argument(
        "--traits",
        nargs="+",
        default=["evil", "anxious", "sassy", "manic"],
        help="Trait names to load from --trait_vector_dir",
    )
    ap.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HF model id, e.g. meta-llama/Llama-3.1-8B-Instruct",
    )
    ap.add_argument(
        "--layer_index",
        type=int,
        default=None,
        help="Explicit layer index. If omitted, uses middle layer.",
    )
    ap.add_argument(
        "--fractions",
        type=str,
        default=",".join(str(x) for x in DEFAULT_FRACTIONS),
        help="Comma-separated steering fractions.",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=180,
        help="Greedy generation length.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    ap.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model load dtype.",
    )
    ap.add_argument(
        "--flip_assistant_axis",
        action="store_true",
        help="Multiply assistant axis by -1 if you want to manually flip PC1 orientation.",
    )
    ap.add_argument(
        "--calibration_prompts",
        nargs="*",
        default=None,
        help="Optional prompts used to calibrate residual norms. Defaults to neutral prompts.",
    )
    ap.add_argument(
        "--test_prompts",
        nargs="*",
        default=None,
        help="Optional prompts to test. Defaults to neutral prompts.",
    )
    return ap.parse_args()


def get_torch_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def parse_fractions(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


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
    raise RuntimeError("Cannot locate transformer layers in model.")


def choose_middle_layer(model) -> int:
    return len(get_layers(model)) // 2


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


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
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
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = enc["input_ids"].shape[1]
    text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    return text.strip()


def layer_norm_report(full_vector: torch.Tensor) -> list[str]:
    norms = full_vector.norm(dim=-1).tolist()
    return [f"layer {i:2d}: {n:.4f}" for i, n in enumerate(norms)]


class ResidualNormCalibrator:
    # Measures average residual-stream norm at the chosen layer output.
    def __init__(self, layer_index: int):
        self.layer_index = int(layer_index)
        self.norm_sum = 0.0
        self.count = 0
        self.handles = []

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            norms = hidden.float().norm(dim=-1)
            self.norm_sum += norms.mean().item()
            self.count += 1

    def register(self, model):
        layers = get_layers(model)
        self.handles.append(layers[self.layer_index].register_forward_hook(self._hook))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def calibrate(self, model, tokenizer, prompts: Iterable[str]) -> float:
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

        if self.count == 0:
            raise RuntimeError(f"Failed to calibrate residual norm for layer {self.layer_index}")

        return self.norm_sum / self.count


class AdditiveSteeringHook:
    # Adds alpha * residual_norm * unit_vector at every token position of the selected layer output.
    def __init__(self, full_vector: torch.Tensor, layer_index: int, layer_residual_norm: float, alpha: float):
        self.full_vector = full_vector.float()
        self.layer_index = int(layer_index)
        self.layer_residual_norm = float(layer_residual_norm)
        self.alpha = float(alpha)
        self.handles = []

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        vec = self.full_vector[self.layer_index]
        unit = vec / (vec.norm() + 1e-8)
        coeff = self.alpha * self.layer_residual_norm
        delta = (coeff * unit).to(hidden.device, hidden.dtype)
        hidden = hidden + delta.view(1, 1, -1)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    def register(self, model):
        layers = get_layers(model)
        self.handles.append(layers[self.layer_index].register_forward_hook(self._hook))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def run_test(
    model,
    tokenizer,
    full_vector: torch.Tensor,
    layer_index: int,
    layer_residual_norm: float,
    fractions: list[float],
    prompt: str,
    max_new_tokens: int,
) -> list[dict]:
    results = []

    for frac in fractions:
        hook = None
        try:
            if frac != 0.0:
                hook = AdditiveSteeringHook(
                    full_vector=full_vector,
                    layer_index=layer_index,
                    layer_residual_norm=layer_residual_norm,
                    alpha=frac,
                )
                hook.register(model)

            response = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        finally:
            if hook is not None:
                hook.remove()

        coeff = round(frac * layer_residual_norm, 4)
        results.append(
            {
                "fraction": frac,
                "coeff": coeff,
                "response": response,
            }
        )

    return results


def emit_header(title: str):
    print("\n" + "█" * 110)
    print(title)
    print("█" * 110)


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    fractions = parse_fractions(args.fractions)
    test_prompts = args.test_prompts if args.test_prompts else list(NEUTRAL_TEST_PROMPTS)
    calibration_prompts = args.calibration_prompts if args.calibration_prompts else list(NEUTRAL_TEST_PROMPTS)

    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=get_torch_dtype(args.torch_dtype),
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")

    layers = get_layers(model)
    n_layers = len(layers)
    layer_index = args.layer_index if args.layer_index is not None else choose_middle_layer(model)

    if layer_index < 0 or layer_index >= n_layers:
        raise ValueError(f"layer_index={layer_index} out of range for model with {n_layers} layers")

    print(f"Using layer index: {layer_index} / {n_layers - 1}")
    print(f"Fractions tested: {fractions}")
    print(f"Calibration prompts: {len(calibration_prompts)}")
    print(f"Test prompts: {len(test_prompts)}")

    named_vectors: list[tuple[str, torch.Tensor]] = []

    assistant_axis = load_full_vector(args.assistant_axis_path)
    if args.flip_assistant_axis:
        assistant_axis = -assistant_axis
    named_vectors.append(("assistant_axis", assistant_axis))

    for trait in args.traits:
        path = args.trait_vector_dir / f"{trait}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing trait vector: {path}")
        named_vectors.append((trait, load_full_vector(path)))

    for name, full_vector in named_vectors:
        if full_vector.shape[0] != n_layers:
            raise ValueError(
                f"Vector {name} has {full_vector.shape[0]} layers but model has {n_layers} layers"
            )

        calibrator = ResidualNormCalibrator(layer_index)
        layer_residual_norm = calibrator.calibrate(
            model=model,
            tokenizer=tokenizer,
            prompts=calibration_prompts,
        )

        emit_header(f"VECTOR: {name.upper()}")
        print(f"Vector shape: {tuple(full_vector.shape)}")
        print(f"Selected layer: {layer_index}")
        print(f"Vector norm at selected layer: {full_vector[layer_index].norm().item():.4f}")
        print(f"Calibrated residual norm at selected layer: {layer_residual_norm:.4f}")

        print("Top per-layer vector norms:")
        top_pairs = sorted(
            [(i, full_vector[i].norm().item()) for i in range(full_vector.shape[0])],
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        print("  " + ", ".join([f"L{i}={n:.3f}" for i, n in top_pairs]))

        print("Per-layer vector norm report:")
        for row in layer_norm_report(full_vector):
            print("  " + row)

        for prompt in test_prompts:
            print("\n" + "─" * 110)
            print(f'PROMPT: "{prompt}"')
            print("─" * 110)

            results = run_test(
                model=model,
                tokenizer=tokenizer,
                full_vector=full_vector,
                layer_index=layer_index,
                layer_residual_norm=layer_residual_norm,
                fractions=fractions,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )

            for r in results:
                print(f"\n  [frac={r['fraction']:+.2f}] coeff={r['coeff']:+.4f}")
                for line in textwrap.wrap(r["response"], width=104):
                    print(f"    {line}")

    print("\n" + "=" * 110)
    print("Done.")


if __name__ == "__main__":
    main()