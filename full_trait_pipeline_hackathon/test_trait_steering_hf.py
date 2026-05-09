#!/usr/bin/env python3
# Tests additive steering using selected-layer trait vectors from Gemma traits40 pipeline.

import argparse
import random
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from chunked_sdpa import chunked_sdpa_scope


NEUTRAL_TEST_PROMPTS = [
    "Introduce yourself in one paragraph.",
    "Describe your communication style.",
    "How should I negotiate a salary?",
    "Write advice for a young student starting university.",
    "Explain how to prepare for a job interview.",
    "What makes a good leader?",
]

DEFAULT_FRACTIONS = [0.0, 0.05, 0.10, 0.15, 0.20, -0.05, -0.10, -0.15, -0.20]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Test additive steering with selected-layer trait vectors.")

    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--trait_vector_dir", type=Path, required=True)
    ap.add_argument("--traits", nargs="+", required=True)

    ap.add_argument(
        "--vector_layers",
        type=str,
        required=True,
        help="Comma-separated original model layer indices stored in the vector rows, e.g. 15,20,25,30,35,40,45",
    )
    ap.add_argument(
        "--layer_index",
        type=int,
        required=True,
        help="Original model layer to steer at. Must be one of --vector_layers.",
    )

    ap.add_argument("--fractions", type=str, default=",".join(str(x) for x in DEFAULT_FRACTIONS))
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    ap.add_argument("--calibration_prompts", nargs="*", default=None)
    ap.add_argument("--test_prompts", nargs="*", default=None)

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


def parse_vector_layers(s: str) -> list[int]:
    layers = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not layers:
        raise ValueError("--vector_layers cannot be empty")
    return layers


def load_trait_vector(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        if "vector" not in data:
            raise KeyError(f"Expected key 'vector' in {path}. Keys: {list(data.keys())}")
        vector = data["vector"].float()
    else:
        vector = data.float()

    if vector.ndim != 2:
        raise ValueError(f"Expected 2D vector tensor in {path}, got shape {tuple(vector.shape)}")

    return vector


def get_layers(model):
    candidates = [
        "language_model.model.layers",
        "language_model.layers",
        "model.layers",
        "model.model.layers",
        "model.language_model.layers",
        "model.language_model.model.layers",
        "text_model.layers",
        "transformer.h",
        "model.decoder.layers",
    ]

    for attr in candidates:
        obj = model
        ok = True
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok:
            return obj

    # Generic fallback: find the first ModuleList with length matching text_config.num_hidden_layers.
    target_n = None
    cfg = getattr(model, "config", None)

    if cfg is not None:
        text_config = getattr(cfg, "text_config", None)
        if text_config is not None:
            target_n = getattr(text_config, "num_hidden_layers", None)

        if target_n is None:
            target_n = getattr(cfg, "num_hidden_layers", None)

    if target_n is not None:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) == int(target_n):
                print(f"Found layer ModuleList via fallback: {name}", flush=True)
                return module

    print("Could not locate layers. Top-level modules:", flush=True)
    for name, module in model.named_children():
        print(f"  {name}: {type(module)}", flush=True)

    raise RuntimeError("Cannot locate transformer layers in model.")


def move_batch_to_device(batch: dict, device):
    return {k: v.to(device) for k, v in batch.items()}


def get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    enc = format_chat(tokenizer, prompt)

    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"])

    enc = move_batch_to_device(enc, get_model_device(model))

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    prompt_len = enc["input_ids"].shape[1]
    text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    return text.strip()


class ResidualNormCalibrator:
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
                    enc = move_batch_to_device(enc, get_model_device(model))
                    _ = model(**enc, use_cache=False)
        finally:
            self.remove()

        if self.count == 0:
            raise RuntimeError(f"Failed to calibrate residual norm for layer {self.layer_index}")

        return self.norm_sum / self.count


class AdditiveSteeringHook:
    def __init__(
        self,
        selected_layer_vector: torch.Tensor,
        layer_index: int,
        layer_residual_norm: float,
        alpha: float,
    ):
        self.selected_layer_vector = selected_layer_vector.float()
        self.layer_index = int(layer_index)
        self.layer_residual_norm = float(layer_residual_norm)
        self.alpha = float(alpha)
        self.handles = []

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output

        unit = self.selected_layer_vector / (self.selected_layer_vector.norm() + 1e-8)
        coeff = self.alpha * self.layer_residual_norm
        delta = (coeff * unit).to(hidden.device, hidden.dtype)

        hidden = hidden + delta.view(1, 1, -1)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

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
    selected_layer_vector: torch.Tensor,
    layer_index: int,
    layer_residual_norm: float,
    fractions: list[float],
    prompt: str,
    max_new_tokens: int,
) -> list[dict]:
    results = []

    for frac in fractions:
        hook: Optional[AdditiveSteeringHook] = None

        try:
            if frac != 0.0:
                hook = AdditiveSteeringHook(
                    selected_layer_vector=selected_layer_vector,
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
    print("\n" + "█" * 110, flush=True)
    print(title, flush=True)
    print("█" * 110, flush=True)


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    fractions = parse_fractions(args.fractions)
    vector_layers = parse_vector_layers(args.vector_layers)

    if args.layer_index not in vector_layers:
        raise ValueError(
            f"--layer_index {args.layer_index} is not in --vector_layers {vector_layers}. "
            "You can only steer at layers that exist in the saved trait vector rows."
        )

    vector_row_index = vector_layers.index(args.layer_index)

    test_prompts = args.test_prompts if args.test_prompts else list(NEUTRAL_TEST_PROMPTS)
    calibration_prompts = args.calibration_prompts if args.calibration_prompts else list(NEUTRAL_TEST_PROMPTS)

    print(f"Loading tokenizer: {args.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    print(f"Loading model: {args.model_id}", flush=True)

    sdpa_cm = chunked_sdpa_scope()
    sdpa_cm.__enter__()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=get_torch_dtype(args.torch_dtype),
            attn_implementation="sdpa",
            device_map="cuda:0",
            trust_remote_code=True,
        )
        model.eval()

        for p in model.parameters():
            p.requires_grad_(False)

        layers = get_layers(model)
        n_layers = len(layers)

        if args.layer_index < 0 or args.layer_index >= n_layers:
            raise ValueError(f"layer_index={args.layer_index} out of range for model with {n_layers} layers")

        print("Model loaded.", flush=True)
        print(f"Model layers: {n_layers}", flush=True)
        print(f"Vector layers: {vector_layers}", flush=True)
        print(f"Steering model layer: {args.layer_index}", flush=True)
        print(f"Using vector row: {vector_row_index}", flush=True)
        print(f"Fractions tested: {fractions}", flush=True)
        print(f"Calibration prompts: {len(calibration_prompts)}", flush=True)
        print(f"Test prompts: {len(test_prompts)}", flush=True)

        calibrator = ResidualNormCalibrator(args.layer_index)
        layer_residual_norm = calibrator.calibrate(
            model=model,
            tokenizer=tokenizer,
            prompts=calibration_prompts,
        )

        print(f"Calibrated residual norm at layer {args.layer_index}: {layer_residual_norm:.4f}", flush=True)

        for trait in args.traits:
            path = args.trait_vector_dir / f"{trait}.pt"

            if not path.exists():
                print(f"\nMissing trait vector, skipping: {path}", flush=True)
                continue

            full_vector = load_trait_vector(path)

            if full_vector.shape[0] != len(vector_layers):
                raise ValueError(
                    f"Trait {trait} vector has {full_vector.shape[0]} rows, "
                    f"but --vector_layers has {len(vector_layers)} entries."
                )

            selected_layer_vector = full_vector[vector_row_index]

            emit_header(f"TRAIT VECTOR: {trait.upper()}")
            print(f"Path: {path}", flush=True)
            print(f"Vector tensor shape: {tuple(full_vector.shape)}", flush=True)
            print(f"Selected vector row: {vector_row_index}", flush=True)
            print(f"Original model layer: {args.layer_index}", flush=True)
            print(f"Selected vector norm: {selected_layer_vector.norm().item():.4f}", flush=True)

            row_norms = [(layer, full_vector[i].norm().item()) for i, layer in enumerate(vector_layers)]
            print("Per-selected-layer vector norms:", flush=True)
            print("  " + ", ".join([f"L{layer}={norm:.3f}" for layer, norm in row_norms]), flush=True)

            for prompt in test_prompts:
                print("\n" + "─" * 110, flush=True)
                print(f'PROMPT: "{prompt}"', flush=True)
                print("─" * 110, flush=True)

                results = run_test(
                    model=model,
                    tokenizer=tokenizer,
                    selected_layer_vector=selected_layer_vector,
                    layer_index=args.layer_index,
                    layer_residual_norm=layer_residual_norm,
                    fractions=fractions,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                )

                for r in results:
                    print(f"\n  [frac={r['fraction']:+.2f}] coeff={r['coeff']:+.4f}", flush=True)
                    if not r["response"]:
                        print("    <empty response>", flush=True)
                    else:
                        for line in textwrap.wrap(r["response"], width=104):
                            print(f"    {line}", flush=True)

        print("\n" + "=" * 110, flush=True)
        print("Done.", flush=True)

    finally:
        sdpa_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()