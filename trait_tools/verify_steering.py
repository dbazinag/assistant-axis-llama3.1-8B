#!/usr/bin/env python3
# Stronger trait steering verifier: steers across the strongest late layers, uses larger coefficients, and leaves output length unconstrained by small max_new_tokens.

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
        "Answer this in one sentence: what is a triangle?",
        "Explain how to boil an egg.",
        "How should I negotiate a salary?",
        "Describe your communication style.",
    ],
    "bombastic": [
        "Introduce yourself dramatically.",
        "Explain how to make toast.",
        "Describe ambition in one paragraph.",
        "How should I negotiate a salary?",
    ],
    "chaotic": [
        "Give me a neat step-by-step plan to organize my desk.",
        "Explain how to prepare for a job interview.",
        "How should I negotiate a salary?",
        "Describe your communication style.",
    ],
    "pedantic": [
        "What is 2+2?",
        "Explain what a triangle is.",
        "What time is it if I leave at 3 and arrive 2 hours later?",
        "Define a rectangle.",
    ],
    "casual": [
        "Tell me how to make toast.",
        "How should I negotiate a salary?",
        "What makes a good first impression?",
        "How do I calm down before an interview?",
    ],
    "edgy": [
        "Describe modern society in one paragraph.",
        "What do you think about ambition?",
        "Introduce yourself in one paragraph.",
        "How should I negotiate a salary?",
    ],
}

DEFAULT_PROMPTS = [
    "Introduce yourself in one paragraph.",
    "How should I negotiate a salary?",
    "Describe your communication style.",
]

# Wider sweep so you can actually see the style move.
FRACTIONS = [0.0, +0.50, +1.00, +2.00, -0.50, -1.00, -2.00]
FRAC_LABELS = [
    "baseline",
    "+0.50 × norm",
    "+1.00 × norm",
    "+2.00 × norm",
    "-0.50 × norm",
    "-1.00 × norm",
    "-2.00 × norm",
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


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 300) -> str:
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


def pick_topk_layers_from_trait(full_vector: torch.Tensor, top_k: int) -> list[int]:
    top_k = max(1, min(top_k, full_vector.shape[0]))
    norms = full_vector.norm(dim=-1)
    idxs = torch.topk(norms, k=top_k).indices.tolist()
    return sorted(idxs)


def parse_layers_arg(model, layers_arg: str, full_vector: torch.Tensor | None, top_k: int) -> list[int]:
    n_layers = len(get_layers(model))

    if layers_arg == "late":
        return list(range(max(0, n_layers - 8), n_layers))
    if layers_arg == "middle":
        start = max(0, n_layers // 2 - 4)
        end = min(n_layers, n_layers // 2 + 4)
        return list(range(start, end))
    if layers_arg == "all":
        return list(range(n_layers))
    if layers_arg == "topk":
        if full_vector is None:
            raise ValueError("full_vector required for topk layer selection")
        return pick_topk_layers_from_trait(full_vector, top_k=top_k)

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


def layer_norm_report(full_vector: torch.Tensor) -> list[str]:
    norms = full_vector.norm(dim=-1).tolist()
    return [f"layer {i:2d}: {n:.4f}" for i, n in enumerate(norms)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors_dir", required=True, type=Path)
    ap.add_argument("--model_id", required=True)
    ap.add_argument(
        "--layers",
        type=str,
        default="topk",
        help="Layer set: 'topk', 'middle', 'late', 'all', or comma-separated indices like '24,28,30,31'",
    )
    ap.add_argument("--top_k", type=int, default=6, help="Used when --layers topk")
    ap.add_argument("--traits", nargs="+", default=["verbose", "bombastic", "chaotic", "pedantic"])
    ap.add_argument("--max_new_tokens", type=int, default=300, help="Set high so you have enough text to judge")
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

    trait_paths = [args.vectors_dir / f"{t}.pt" for t in args.traits]
    missing = [str(p) for p in trait_paths if not p.exists()]
    if missing:
        sys.exit(f"Missing trait vector files: {missing}")

    # Use a global avg norm estimate from late layers so coefficient scale is not tiny.
    late_layers = list(range(max(0, len(get_layers(model)) - 8), len(get_layers(model))))
    avg_norm = avg_norm_from_vectors(args.vectors_dir, late_layers)
    print(f"Global avg norm estimate from late layers {late_layers}: {avg_norm:.4f}\n")

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"Trait steering verification  |  model: {args.model_id}")
    emit(f"Fractions tested: {FRACTIONS}")
    emit(f"Global avg norm used for scaling: {avg_norm:.4f}")
    emit("=" * 110)

    for tp in trait_paths:
        trait_name = tp.stem
        full_vector = load_full_vector(tp)
        prompts = TRAIT_PROMPTS.get(trait_name, DEFAULT_PROMPTS)
        layer_indices = parse_layers_arg(model, args.layers, full_vector, args.top_k)

        emit(f"\n\n{'█' * 110}")
        emit(f"TRAIT VECTOR: {trait_name.upper()}")
        emit(f"Vector shape: {tuple(full_vector.shape)}")
        emit(f"Selected layers: {layer_indices}")
        emit("Top per-layer norms:")
        top_pairs = sorted(
            [(i, full_vector[i].norm().item()) for i in range(full_vector.shape[0])],
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        emit("  " + ", ".join([f"L{i}={n:.3f}" for i, n in top_pairs]))
        emit("█" * 110)

        emit("Per-layer norm report:")
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
                avg_norm=avg_norm,
                layer_indices=layer_indices,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )

            for r in results:
                emit(f"\n  [{r['label']}]  (coeff={r['coeff']})")
                for line in textwrap.wrap(r["response"], width=104):
                    emit(f"    {line}")

    emit("\n" + "=" * 110)
    emit("DIAGNOSIS:")
    emit("  topk should usually be the strongest debug setting because it picks the trait's highest-norm layers.")
    emit("  If baseline and ±2.00 still look nearly identical, steering is weak even if projections may still be useful.")
    emit("  For verbose/pedantic, look for longer, more exhaustive, more over-specified phrasing on positive coeffs.")
    emit("  For chaotic/bombastic, look for structure breakdown or dramatic language shifts at larger magnitudes.")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text("\n".join(lines) + "\n")
    print(f"\nSaved → {args.out_file}")


if __name__ == "__main__":
    main()