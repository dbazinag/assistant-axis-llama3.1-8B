#!/usr/bin/env python3
# Stronger steering verifier: sweeps multiple layers, uses larger coefficients, and uses trait-sensitive prompts.

import argparse
import random
import sys
import textwrap
from pathlib import Path

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("transformers not found. Run: uv add transformers")


TRAIT_PROMPTS = {
    "verbose": [
        "Explain how to boil an egg.",
        "Describe your communication style.",
    ],
    "bombastic": [
        "Introduce yourself in one paragraph.",
        "Explain how to make toast.",
    ],
    "chaotic": [
        "Give me a step-by-step plan to clean my room.",
        "Explain how to prepare for a job interview.",
    ],
    "pedantic": [
        "What time is it if I leave at 3 and arrive 2 hours later?",
        "Explain what a triangle is.",
    ],
    "casual": [
        "How should I negotiate a salary?",
        "Tell me how to make a good first impression.",
    ],
    "edgy": [
        "What do you think about ambition?",
        "Describe modern society in one paragraph.",
    ],
}

DEFAULT_PROMPTS = [
    "Introduce yourself in one paragraph.",
    "How should I negotiate a salary?",
]

# Use a much wider sweep than before so the effect is actually visible if it exists.
FRACTIONS = [0.0, +0.25, +0.50, +1.00, -0.25, -0.50, -1.00]
FRAC_LABELS = [
    "baseline",
    "+0.25 × norm",
    "+0.50 × norm",
    "+1.00 × norm",
    "-0.25 × norm",
    "-0.50 × norm",
    "-1.00 × norm",
]


def load_full_vector(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        if "vector" in data:
            t = data["vector"].float()
            if t.ndim != 2:
                raise ValueError(f"Expected 2D vector tensor in {path}, got shape {tuple(t.shape)}")
            return t
        raise KeyError(f"Cannot find 'vector' key in {path}. Keys: {list(data.keys())}")
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


class MultiLayerAdditionHook:
    """Adds coeff * normalized_vector[layer] to every token at selected layers."""
    def __init__(self, full_vector: torch.Tensor, coeff: float, layer_indices: list[int]):
        self.full_vector = full_vector.float()
        self.coeff = float(coeff)
        self.layer_indices = list(layer_indices)
        self.handles = []

    def _make_hook(self, layer_idx: int):
        unit = self.full_vector[layer_idx] / (self.full_vector[layer_idx].norm() + 1e-8)

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            delta = (self.coeff * unit).to(hidden.device, hidden.dtype)
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


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 180) -> str:
    messages = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def avg_norm_from_vectors(vectors_dir: Path, layer_indices: list[int], max_files: int = 20) -> float:
    pts = list(vectors_dir.glob("*.pt"))
    if not pts:
        raise RuntimeError(f"No vector files found in {vectors_dir}")

    pts = random.sample(pts, min(max_files, len(pts)))
    norms = []
    for p in pts:
        try:
            t = load_full_vector(p)
            selected = t[layer_indices]
            norms.append(selected.norm(dim=-1).mean().item())
        except Exception:
            continue

    if not norms:
        raise RuntimeError("Could not estimate avg norm from vectors.")
    return float(sum(norms) / len(norms))


def run_test(
    model,
    tokenizer,
    full_vector: torch.Tensor,
    avg_norm: float,
    layer_indices: list[int],
    prompt: str,
    max_new_tokens: int,
) -> list[dict]:
    results = []
    for frac, flabel in zip(FRACTIONS, FRAC_LABELS):
        coeff = frac * avg_norm
        hook = None
        if coeff != 0.0:
            hook = MultiLayerAdditionHook(full_vector.clone(), coeff, layer_indices)
            hook.register(model)

        response = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)

        if hook:
            hook.remove()

        results.append(
            {
                "fraction": frac,
                "coeff": round(coeff, 4),
                "label": flabel,
                "response": response,
            }
        )
    return results


def parse_layers_arg(model, layers_arg: str) -> list[int]:
    n_layers = len(get_layers(model))

    if layers_arg == "late":
        return list(range(max(0, n_layers - 8), n_layers))
    if layers_arg == "middle":
        start = max(0, n_layers // 2 - 4)
        end = min(n_layers, n_layers // 2 + 4)
        return list(range(start, end))
    if layers_arg == "all":
        return list(range(n_layers))

    out = []
    for part in layers_arg.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"Layer index {idx} out of range for model with {n_layers} layers")
        out.append(idx)

    if not out:
        raise ValueError("No valid layers parsed")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors_dir", required=True, type=Path)
    ap.add_argument("--model_id", required=True)
    ap.add_argument(
        "--layers",
        type=str,
        default="middle",
        help="Layer set: 'middle', 'late', 'all', or comma-separated indices like '16,20,24'",
    )
    ap.add_argument("--traits", nargs="+", default=["verbose", "bombastic", "chaotic", "pedantic"])
    ap.add_argument("--max_new_tokens", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_file", type=Path, default=Path("verify_trait_steering.txt"))
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.\n")

    layer_indices = parse_layers_arg(model, args.layers)
    avg_norm = avg_norm_from_vectors(args.vectors_dir, layer_indices)
    print(f"Selected layers: {layer_indices}")
    print(f"Estimated avg norm over selected layers: {avg_norm:.4f}\n")

    trait_paths = [args.vectors_dir / f"{t}.pt" for t in args.traits]
    missing = [str(p) for p in trait_paths if not p.exists()]
    if missing:
        sys.exit(f"Missing trait vector files: {missing}")

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"Trait steering verification  |  model: {args.model_id}")
    emit(f"Layers: {layer_indices}")
    emit(f"Avg norm: {avg_norm:.4f}")
    emit(f"Fractions tested: {FRACTIONS}")
    emit("=" * 100)

    for tp in trait_paths:
        trait_name = tp.stem
        full_vector = load_full_vector(tp)
        prompts = TRAIT_PROMPTS.get(trait_name, DEFAULT_PROMPTS)

        emit(f"\n\n{'█' * 100}")
        emit(f"TRAIT VECTOR: {trait_name.upper()}")
        emit(f"Vector shape: {tuple(full_vector.shape)}")
        emit("█" * 100)

        for prompt in prompts:
            emit(f"\n{'─' * 100}")
            emit(f'PROMPT: "{prompt}"')
            emit(f"{'─' * 100}")

            results = run_test(
                model=model,
                tokenizer=tokenizer,
                full_vector=full_vector,
                avg_norm=avg_norm,
                layer_indices=layer_indices,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )

            for r in results:
                emit(f"\n  [{r['label']}]  (coeff={r['coeff']})")
                for line in textwrap.wrap(r["response"], width=96):
                    emit(f"    {line}")

    emit("\n" + "=" * 100)
    emit("DIAGNOSIS:")
    emit("  Strong visible changes at larger coefficients mean the vector is steerable.")
    emit("  If baseline and ±1.00 still look identical, the trait may project well but steer weakly.")
    emit("  Try --layers late or explicit layers like 20,24,28 if the effect is still weak.")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text("\n".join(lines) + "\n")
    print(f"\nSaved → {args.out_file}")


if __name__ == "__main__":
    main()