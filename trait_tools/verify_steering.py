#!/usr/bin/env python3
# Steers the model along selected trait vectors and saves responses for manual inspection.

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


TRAIT_PROMPTS = [
    "Who are you?",
    "How would you describe your communication style?",
    "How should I negotiate a salary?",
]

FRACTIONS = [0.0, +0.10, +0.25, -0.10, -0.25]
FRAC_LABELS = [
    "baseline",
    "+0.10 × norm",
    "+0.25 × norm",
    "-0.10 × norm",
    "-0.25 × norm",
]


def load_vector(path: Path, layer_idx: int) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
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


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 180) -> str:
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


def avg_norm_from_vectors(vectors_dir: Path, layer_idx: int, max_files: int = 20) -> float:
    pts = list(vectors_dir.glob("*.pt"))
    if not pts:
        raise RuntimeError(f"No vector files found in {vectors_dir}")
    pts = random.sample(pts, min(max_files, len(pts)))
    norms = []
    for p in pts:
        try:
            t = load_vector(p, layer_idx)
            norms.append(t.norm().item())
        except Exception:
            continue
    if not norms:
        raise RuntimeError("Could not estimate avg norm from vectors")
    return float(sum(norms) / len(norms))


def run_test(model, tokenizer, layer_idx: int, vector: torch.Tensor,
             avg_norm: float, prompt: str) -> list[dict]:
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
            "coeff": round(coeff, 3),
            "label": flabel,
            "response": response,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors_dir", required=True, type=Path)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--traits", nargs="+", default=["verbose", "chaotic", "bombastic"])
    ap.add_argument("--max_new_tokens", type=int, default=180)
    ap.add_argument("--out_file", type=Path, default=Path("verify_trait_steering.txt"))
    args = ap.parse_args()

    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print("Model loaded.\n")

    avg_norm = avg_norm_from_vectors(args.vectors_dir, args.layer)
    print(f"Estimated avg norm from vectors: {avg_norm:.2f}\n")

    trait_paths = [args.vectors_dir / f"{t}.pt" for t in args.traits]
    missing = [p for p in trait_paths if not p.exists()]
    if missing:
        sys.exit(f"Missing trait vector files: {missing}")

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"Trait steering verification  |  model: {args.model_id}")
    emit(f"Layer: {args.layer}  |  avg norm: {avg_norm:.2f}")
    emit(f"Fractions tested: {FRACTIONS}")
    emit("=" * 80)

    for tp in trait_paths:
        trait_vec = load_vector(tp, args.layer)
        trait_name = tp.stem

        emit(f"\n\n{'█' * 80}")
        emit(f"TRAIT VECTOR: {trait_name.upper()}")
        emit("█" * 80)

        for prompt in TRAIT_PROMPTS:
            emit(f"\n{'─' * 80}")
            emit(f'PROMPT: "{prompt}"')
            emit(f"{'─' * 80}")
            results = run_test(model, tokenizer, args.layer, trait_vec, avg_norm, prompt)
            for r in results:
                emit(f"\n  [{r['label']}]  (coeff={r['coeff']})")
                for line in textwrap.wrap(r["response"], width=84):
                    emit(f"    {line}")

    emit("\n" + "=" * 80)
    emit("DIAGNOSIS:")
    emit("  Positive and negative coefficients should visibly change the style/trait.")
    emit("  If all variants look identical, the hook or scaling may be wrong.")
    emit("  Use visually obvious kept traits first, like verbose, chaotic, bombastic, pedantic.")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text("\n".join(lines) + "\n")
    print(f"\nSaved → {args.out_file}")


if __name__ == "__main__":
    main()