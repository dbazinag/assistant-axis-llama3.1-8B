#!/usr/bin/env python3
# Interview note: project old jailbreak activations onto the new traits40 assistant axis and all trait vectors, for both pre-response last token and answer mean.

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch


DEFAULT_ACTS_DIR = Path("../assistant_axis_outputs/llama-3.1-8b/step2/activations")
DEFAULT_AXIS_PATH = Path(
    "full_trait_output/traits40_axes/answer_mean/filter_matched_pairs_ge_50_count_ge_10_total/assistant_axis_pc1.pt"
)
DEFAULT_TRAITS_DIR = Path(
    "full_trait_output/traits40_vectors/answer_mean/filter_matched_pairs_ge_50_count_ge_10_total"
)
DEFAULT_OUT_DIR = Path("full_trait_output/jailbreak_projections_traits40_pre_and_answer")


def load_vector(path: Path, n_layers: int | None = None) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        if "vector" in data:
            vec = data["vector"]
        elif "axis" in data:
            vec = data["axis"]
        elif "pc1" in data:
            vec = data["pc1"]
        else:
            raise ValueError(f"Could not find vector tensor in {path}")
    else:
        vec = data

    vec = vec.float()

    if vec.ndim == 1:
        if n_layers is None:
            raise ValueError(f"Vector {path} is 1D but n_layers is unknown")
        vec = vec.unsqueeze(0).expand(n_layers, -1)

    if vec.ndim != 2:
        raise ValueError(f"Expected 2D vector for {path}, got shape {tuple(vec.shape)}")

    return vec


def dot_projection_per_layer(acts: torch.Tensor, vec: torch.Tensor) -> list[float]:
    denom = vec.norm(dim=1) + 1e-8
    proj = (acts * vec).sum(dim=1) / denom
    return proj.tolist()


def cosine_per_layer(acts: torch.Tensor, vec: torch.Tensor) -> list[float]:
    acts_denom = acts.norm(dim=1) + 1e-8
    vec_denom = vec.norm(dim=1) + 1e-8
    cos = (acts * vec).sum(dim=1) / (acts_denom * vec_denom)
    return cos.tolist()


def delta_list(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def mean_val(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def label_prompt(rec: dict) -> str:
    is_benign = rec.get("attack_method") == "benign" or rec.get("is_jailbreak") is False
    judge = rec.get("jailbroken_our_judge")

    if is_benign:
        return "benign"
    if judge is True:
        return "successful_jailbreak"
    if judge is False:
        return "unsuccessful_jailbreak"
    return "unjudged_jailbreak"


def find_activation_files(acts_dir: Path) -> list[Path]:
    return sorted(acts_dir.glob("*.pt"))


def select_test_subset(
    act_files: list[Path],
    test_mode: bool,
    test_n: int,
    test_seed: int,
    test_balanced: bool,
) -> list[Path]:
    if not test_mode:
        return act_files

    if not act_files:
        return act_files

    random.seed(test_seed)

    if not test_balanced:
        return act_files[: min(test_n, len(act_files))]

    buckets = {
        "benign": [],
        "successful_jailbreak": [],
        "unsuccessful_jailbreak": [],
        "unjudged_jailbreak": [],
    }

    for path in act_files:
        try:
            rec = torch.load(path, map_location="cpu", weights_only=False)
            buckets[label_prompt(rec)].append(path)
        except Exception:
            continue

    per_bucket = max(1, test_n // max(1, len(buckets)))
    selected = []

    for files in buckets.values():
        if not files:
            continue
        if len(files) <= per_bucket:
            selected.extend(files)
        else:
            selected.extend(random.sample(files, per_bucket))

    if len(selected) < test_n:
        selected_names = {p.name for p in selected}
        remaining = [p for p in act_files if p.name not in selected_names]
        extra_needed = min(test_n - len(selected), len(remaining))
        if extra_needed > 0:
            selected.extend(random.sample(remaining, extra_needed))

    selected = sorted(selected, key=lambda p: p.name)
    return selected[: min(test_n, len(selected))]


def validate_shapes(name: str, acts: torch.Tensor, vec: torch.Tensor, expected_layers: int) -> None:
    if acts.ndim != 2:
        raise ValueError(f"{name}: acts must be 2D, got {tuple(acts.shape)}")
    if vec.ndim != 2:
        raise ValueError(f"{name}: vec must be 2D, got {tuple(vec.shape)}")
    if acts.shape[0] != expected_layers or vec.shape[0] != expected_layers:
        raise ValueError(
            f"{name}: layer mismatch acts={tuple(acts.shape)} vec={tuple(vec.shape)} expected_layers={expected_layers}"
        )
    if acts.shape[1] != vec.shape[1]:
        raise ValueError(f"{name}: hidden mismatch acts={tuple(acts.shape)} vec={tuple(vec.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", type=Path, default=DEFAULT_ACTS_DIR)
    ap.add_argument("--axis_path", type=Path, default=DEFAULT_AXIS_PATH)
    ap.add_argument("--traits_dir", type=Path, default=DEFAULT_TRAITS_DIR)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--trait_names", nargs="*", default=None)
    ap.add_argument("--write_per_prompt_pt", action="store_true")

    ap.add_argument("--test_mode", action="store_true")
    ap.add_argument("--test_n", type=int, default=12)
    ap.add_argument("--test_seed", type=int, default=42)
    ap.add_argument("--test_balanced", action="store_true")
    ap.add_argument("--fail_fast", action="store_true")
    args = ap.parse_args()

    base_out_dir = args.out_dir
    if args.test_mode:
        args.out_dir = args.out_dir / "test_mode"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_prompt_dir = args.out_dir / "per_prompt"
    if args.write_per_prompt_pt:
        per_prompt_dir.mkdir(parents=True, exist_ok=True)

    act_files = find_activation_files(args.acts_dir)
    if not act_files:
        raise FileNotFoundError(f"No .pt files found in {args.acts_dir}")

    act_files = select_test_subset(
        act_files=act_files,
        test_mode=args.test_mode,
        test_n=args.test_n,
        test_seed=args.test_seed,
        test_balanced=args.test_balanced,
    )
    if not act_files:
        raise RuntimeError("No activation files selected")

    sample = torch.load(act_files[0], map_location="cpu", weights_only=False)
    if "last_prompt_acts" not in sample or "mean_response_acts" not in sample:
        raise KeyError("Activation files must contain both 'last_prompt_acts' and 'mean_response_acts'")

    pre_response_sample = sample["last_prompt_acts"].float()
    answer_mean_sample = sample["mean_response_acts"].float()

    if pre_response_sample.shape != answer_mean_sample.shape:
        raise ValueError(
            f"Shape mismatch between last_prompt_acts={tuple(pre_response_sample.shape)} "
            f"and mean_response_acts={tuple(answer_mean_sample.shape)}"
        )

    n_layers = pre_response_sample.shape[0]
    hidden_size = pre_response_sample.shape[1]

    axis = load_vector(args.axis_path, n_layers=n_layers)
    if axis.shape[0] != n_layers:
        raise ValueError(f"Axis layers mismatch: {axis.shape[0]} vs {n_layers}")
    if axis.shape[1] != hidden_size:
        raise ValueError(f"Axis hidden mismatch: axis={axis.shape[1]} acts={hidden_size}")

    if args.trait_names:
        trait_files = [args.traits_dir / f"{name}.pt" for name in args.trait_names]
    else:
        trait_files = sorted(args.traits_dir.glob("*.pt"))

    trait_vectors: dict[str, torch.Tensor] = {}
    for path in trait_files:
        if not path.exists():
            print(f"[WARN] Missing trait vector: {path}")
            continue
        try:
            vec = load_vector(path, n_layers=n_layers)
            if vec.shape[0] != n_layers:
                print(f"[WARN] Skipping {path.name}: layer mismatch {vec.shape[0]} vs {n_layers}")
                continue
            if vec.shape[1] != hidden_size:
                print(f"[WARN] Skipping {path.name}: hidden mismatch {vec.shape[1]} vs {hidden_size}")
                continue
            trait_vectors[path.stem] = vec
        except Exception as e:
            print(f"[WARN] Failed to load {path.name}: {e}")

    if not trait_vectors:
        print("[WARN] No trait vectors loaded")

    run_meta = {
        "date": datetime.now().isoformat(),
        "acts_dir": str(args.acts_dir),
        "requested_out_dir": str(base_out_dir),
        "actual_out_dir": str(args.out_dir),
        "axis_path": str(args.axis_path),
        "traits_dir": str(args.traits_dir),
        "n_prompt_files_selected": len(act_files),
        "n_layers": n_layers,
        "hidden_size": int(hidden_size),
        "axis_shape": list(axis.shape),
        "trait_names": sorted(trait_vectors.keys()),
        "write_per_prompt_pt": args.write_per_prompt_pt,
        "test_mode": args.test_mode,
        "test_n": args.test_n,
        "test_seed": args.test_seed,
        "test_balanced": args.test_balanced,
        "fail_fast": args.fail_fast,
        "activation_sources": {
            "pre_response": "last_prompt_acts",
            "answer_mean": "mean_response_acts",
            "delta": "answer_mean - pre_response",
        },
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    summary_jsonl = args.out_dir / "projection_summary.jsonl"
    aggregate_json = args.out_dir / "aggregate_stats.json"

    counts = {
        "total": 0,
        "benign": 0,
        "successful_jailbreak": 0,
        "unsuccessful_jailbreak": 0,
        "unjudged_jailbreak": 0,
        "failed_files": 0,
    }

    aggregate = {
        "axis": {},
        "traits": {},
    }

    def init_group_slot(container: dict, group_name: str):
        if group_name not in container:
            container[group_name] = {
                "count": 0,
                "pre_response_proj_sum": [0.0] * n_layers,
                "answer_mean_proj_sum": [0.0] * n_layers,
                "delta_proj_sum": [0.0] * n_layers,
                "pre_response_cos_sum": [0.0] * n_layers,
                "answer_mean_cos_sum": [0.0] * n_layers,
                "delta_cos_sum": [0.0] * n_layers,
            }

    def add_to_group(
        container: dict,
        group_name: str,
        pre_proj: list[float],
        answer_proj: list[float],
        delta_proj: list[float],
        pre_cos: list[float],
        answer_cos: list[float],
        delta_cos: list[float],
    ):
        init_group_slot(container, group_name)
        slot = container[group_name]
        slot["count"] += 1
        for i in range(n_layers):
            slot["pre_response_proj_sum"][i] += pre_proj[i]
            slot["answer_mean_proj_sum"][i] += answer_proj[i]
            slot["delta_proj_sum"][i] += delta_proj[i]
            slot["pre_response_cos_sum"][i] += pre_cos[i]
            slot["answer_mean_cos_sum"][i] += answer_cos[i]
            slot["delta_cos_sum"][i] += delta_cos[i]

    with open(summary_jsonl, "w", encoding="utf-8") as f_out:
        for idx, pt_path in enumerate(act_files, start=1):
            try:
                rec = torch.load(pt_path, map_location="cpu", weights_only=False)
                pre_response_acts = rec["last_prompt_acts"].float()
                answer_mean_acts = rec["mean_response_acts"].float()

                validate_shapes("axis:pre_response", pre_response_acts, axis, n_layers)
                validate_shapes("axis:answer_mean", answer_mean_acts, axis, n_layers)

                prompt_label = label_prompt(rec)
                counts["total"] += 1
                counts[prompt_label] += 1

                axis_pre_proj = dot_projection_per_layer(pre_response_acts, axis)
                axis_answer_proj = dot_projection_per_layer(answer_mean_acts, axis)
                axis_delta_proj = delta_list(axis_answer_proj, axis_pre_proj)

                axis_pre_cos = cosine_per_layer(pre_response_acts, axis)
                axis_answer_cos = cosine_per_layer(answer_mean_acts, axis)
                axis_delta_cos = delta_list(axis_answer_cos, axis_pre_cos)

                trait_out = {}
                for trait_name, trait_vec in trait_vectors.items():
                    validate_shapes(f"{trait_name}:pre_response", pre_response_acts, trait_vec, n_layers)
                    validate_shapes(f"{trait_name}:answer_mean", answer_mean_acts, trait_vec, n_layers)

                    t_pre_proj = dot_projection_per_layer(pre_response_acts, trait_vec)
                    t_answer_proj = dot_projection_per_layer(answer_mean_acts, trait_vec)
                    t_delta_proj = delta_list(t_answer_proj, t_pre_proj)

                    t_pre_cos = cosine_per_layer(pre_response_acts, trait_vec)
                    t_answer_cos = cosine_per_layer(answer_mean_acts, trait_vec)
                    t_delta_cos = delta_list(t_answer_cos, t_pre_cos)

                    trait_out[trait_name] = {
                        "pre_response_proj_all_layers": t_pre_proj,
                        "answer_mean_proj_all_layers": t_answer_proj,
                        "delta_proj_all_layers": t_delta_proj,
                        "pre_response_cos_all_layers": t_pre_cos,
                        "answer_mean_cos_all_layers": t_answer_cos,
                        "delta_cos_all_layers": t_delta_cos,
                        "pre_response_proj_mean": mean_val(t_pre_proj),
                        "answer_mean_proj_mean": mean_val(t_answer_proj),
                        "delta_proj_mean": mean_val(t_delta_proj),
                        "pre_response_cos_mean": mean_val(t_pre_cos),
                        "answer_mean_cos_mean": mean_val(t_answer_cos),
                        "delta_cos_mean": mean_val(t_delta_cos),
                    }

                    add_to_group(
                        aggregate["traits"].setdefault(trait_name, {}),
                        prompt_label,
                        t_pre_proj,
                        t_answer_proj,
                        t_delta_proj,
                        t_pre_cos,
                        t_answer_cos,
                        t_delta_cos,
                    )

                add_to_group(
                    aggregate["axis"],
                    prompt_label,
                    axis_pre_proj,
                    axis_answer_proj,
                    axis_delta_proj,
                    axis_pre_cos,
                    axis_answer_cos,
                    axis_delta_cos,
                )

                row = {
                    "pt_file": pt_path.name,
                    "prompt_idx": rec.get("prompt_idx"),
                    "attack_method": rec.get("attack_method"),
                    "artifact_model": rec.get("artifact_model"),
                    "behavior": rec.get("behavior"),
                    "goal": rec.get("goal"),
                    "category": rec.get("category"),
                    "is_jailbreak": rec.get("is_jailbreak"),
                    "jailbroken_jbb": rec.get("jailbroken_jbb"),
                    "jailbroken_our_judge": rec.get("jailbroken_our_judge"),
                    "prompt_label": prompt_label,
                    "prompt_len_tokens": rec.get("prompt_len_tokens"),
                    "response_text": rec.get("response_text"),
                    "activation_sources": {
                        "pre_response": "last_prompt_acts",
                        "answer_mean": "mean_response_acts",
                        "delta": "answer_mean - pre_response",
                    },
                    "axis": {
                        "pre_response_proj_all_layers": axis_pre_proj,
                        "answer_mean_proj_all_layers": axis_answer_proj,
                        "delta_proj_all_layers": axis_delta_proj,
                        "pre_response_cos_all_layers": axis_pre_cos,
                        "answer_mean_cos_all_layers": axis_answer_cos,
                        "delta_cos_all_layers": axis_delta_cos,
                        "pre_response_proj_mean": mean_val(axis_pre_proj),
                        "answer_mean_proj_mean": mean_val(axis_answer_proj),
                        "delta_proj_mean": mean_val(axis_delta_proj),
                        "pre_response_cos_mean": mean_val(axis_pre_cos),
                        "answer_mean_cos_mean": mean_val(axis_answer_cos),
                        "delta_cos_mean": mean_val(axis_delta_cos),
                    },
                    "traits": trait_out,
                }

                f_out.write(json.dumps(row) + "\n")

                if args.write_per_prompt_pt:
                    torch.save(row, per_prompt_dir / pt_path.name)

                print(f"[{idx}/{len(act_files)}] processed {pt_path.name}")

            except Exception as e:
                counts["failed_files"] += 1
                msg = f"[WARN] Failed on {pt_path.name}: {e}"
                if args.fail_fast:
                    raise RuntimeError(msg) from e
                print(msg)

    for _, slot in aggregate["axis"].items():
        c = max(slot["count"], 1)
        for key in [
            "pre_response_proj_sum",
            "answer_mean_proj_sum",
            "delta_proj_sum",
            "pre_response_cos_sum",
            "answer_mean_cos_sum",
            "delta_cos_sum",
        ]:
            slot[key.replace("_sum", "_mean")] = [x / c for x in slot[key]]

    for _, groups in aggregate["traits"].items():
        for _, slot in groups.items():
            c = max(slot["count"], 1)
            for key in [
                "pre_response_proj_sum",
                "answer_mean_proj_sum",
                "delta_proj_sum",
                "pre_response_cos_sum",
                "answer_mean_cos_sum",
                "delta_cos_sum",
            ]:
                slot[key.replace("_sum", "_mean")] = [x / c for x in slot[key]]

    aggregate_payload = {
        "date": datetime.now().isoformat(),
        "test_mode": args.test_mode,
        "counts": counts,
        "activation_sources": {
            "pre_response": "last_prompt_acts",
            "answer_mean": "mean_response_acts",
            "delta": "answer_mean - pre_response",
        },
        "axis": aggregate["axis"],
        "traits": aggregate["traits"],
    }
    aggregate_json.write_text(json.dumps(aggregate_payload, indent=2))

    print("\nDone.")
    print(f"Test mode: {args.test_mode}")
    print(f"Processed prompt files: {counts['total']}")
    print(f"Failed files: {counts['failed_files']}")
    print(f"Benign: {counts['benign']}")
    print(f"Successful jailbreak: {counts['successful_jailbreak']}")
    print(f"Unsuccessful jailbreak: {counts['unsuccessful_jailbreak']}")
    print(f"Unjudged jailbreak: {counts['unjudged_jailbreak']}")
    print(f"Summary JSONL: {summary_jsonl}")
    print(f"Aggregate stats: {aggregate_json}")
    if args.write_per_prompt_pt:
        print(f"Per-prompt projection files: {per_prompt_dir}")


if __name__ == "__main__":
    main()