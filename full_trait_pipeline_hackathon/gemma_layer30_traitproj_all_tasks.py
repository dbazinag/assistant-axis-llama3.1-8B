#!/usr/bin/env python3
"""
Gemma layer-30 trait-projection classifier for Level-1 Gemma-compatible tasks.

Modes:
  extract: extract final prompt-token activation from Gemma model.layers[layer_idx]
  train:   project activations onto trait vectors and train/evaluate official split probes
  all:     extract then train

Recommended fast usage:
  Launch 8 GPU extraction shards:
    python gemma_layer30_traitproj_all_tasks.py --mode extract --num_shards 8 --shard_idx 0
    ...
  Then train once on CPU:
    python gemma_layer30_traitproj_all_tasks.py --mode train

Outputs:
  summary.json with per-task AUC, mean AUC, across-task std/var
  top_trait_coefficients.json for interpretability
  gemma_layer30_trait_projection_probe.pt
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_prompt(row: Dict[str, Any]) -> str:
    for key in ("attack_prompt", "prompt", "input", "question", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError(f"Could not find prompt field. Keys={list(row.keys())}")


def norm_row(row: Dict[str, Any], dataset: str, split: Optional[str]) -> Dict[str, Any]:
    r = dict(row)
    r["dataset"] = dataset
    if split is not None and "split" not in r:
        r["split"] = split
    if "sample_id" not in r:
        raise KeyError(f"row missing sample_id: keys={list(r.keys())}")
    return r


def load_all_rows(repo_root: Path) -> List[Dict[str, Any]]:
    rows = []
    refusal = repo_root / "datasets/refusal_probes/gemma4_31b/attacks_full.jsonl"
    cyber_train = repo_root / "datasets/cyber_probes/train.jsonl"
    cyber_test = repo_root / "datasets/cyber_probes/test.jsonl"

    for r in read_jsonl(refusal):
        rows.append(norm_row(r, "refusal_gemma4_31b", r.get("split")))

    for r in read_jsonl(cyber_train):
        rows.append(norm_row(r, "cyber", "train"))

    for r in read_jsonl(cyber_test):
        rows.append(norm_row(r, "cyber", "test"))

    dedup = {}
    for r in rows:
        dedup.setdefault(r["sample_id"], r)
    return list(dedup.values())


def get_layers(model):
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.language_model.layers,
    ]
    for getter in candidates:
        try:
            layers = getter(model)
            if layers is not None:
                return layers
        except Exception:
            pass
    raise RuntimeError("Could not locate transformer layers")


class ActivationCatcher:
    def __init__(self, layer, final_idx: int):
        self.final_idx = int(final_idx)
        self.activation = None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.activation = hidden[0, self.final_idx, :].detach().float().cpu()

    def remove(self):
        self.handle.remove()


def load_model_and_tokenizer(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def extract_one(model, tok, layer, row: Dict[str, Any], out_path: Path, args) -> Dict[str, Any]:
    prompt = get_prompt(row)
    if not args.no_chat_template:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=False,
    ).to("cuda:0")

    final_idx = int(enc["attention_mask"][0].bool().nonzero().max().item())
    catcher = ActivationCatcher(layer, final_idx)
    t0 = time.time()

    try:
        with torch.no_grad():
            _ = model(**enc, use_cache=False, output_hidden_states=False, return_dict=True)
        torch.cuda.synchronize()
        fwd = time.time() - t0

        if catcher.activation is None:
            raise RuntimeError("hook captured no activation")

        payload = {
            "sample_id": row["sample_id"],
            "dataset": row.get("dataset"),
            "split": row.get("split"),
            "category": row.get("category") or row.get("label"),
            "is_refusal": row.get("is_refusal"),
            "activation": catcher.activation.to(torch.float16).contiguous(),
            "n_tokens": int(enc["input_ids"].shape[1]),
            "layer_idx": int(args.layer_idx),
            "fwd_seconds": round(float(fwd), 4),
        }
        torch.save(payload, out_path)
        return payload
    finally:
        catcher.remove()
        del enc
        torch.cuda.empty_cache()


def run_extract(args) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(args.repo_root / "starter_code"))
    from chunked_sdpa import chunked_sdpa_scope

    rows = sorted(load_all_rows(args.repo_root), key=lambda r: r["sample_id"])
    if args.sample_limit > 0:
        rows = rows[:args.sample_limit]
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_idx]

    out_extracts = args.out_root / "extracts"
    out_extracts.mkdir(parents=True, exist_ok=True)
    logs = args.out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    print(f"extract rows={len(rows)} shard={args.shard_idx}/{args.num_shards}")
    print(f"out_extracts={out_extracts}")
    print(f"layer_idx={args.layer_idx}")

    model, tok = load_model_and_tokenizer(args.model_path)
    layers = get_layers(model)
    layer = layers[args.layer_idx]
    print(f"model loaded, n_layers={len(layers)}, hooked layers[{args.layer_idx}]")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    ok = fail = skip = 0
    t_start = time.time()
    err_path = logs / f"errors_shard{args.shard_idx}.jsonl"

    cm = chunked_sdpa_scope()
    cm.__enter__()
    try:
        for j, row in enumerate(rows, start=1):
            sid = row["sample_id"]
            out_path = out_extracts / f"{sid}.pt"

            if out_path.exists() and not args.overwrite_extracts:
                skip += 1
                continue

            try:
                payload = extract_one(model, tok, layer, row, out_path, args)
                ok += 1
                if ok % 25 == 0 or j == len(rows):
                    elapsed = time.time() - t_start
                    rate = (ok + fail) / max(elapsed, 1e-6)
                    eta = (len(rows) - j) / max(rate, 1e-6) / 60
                    print(
                        f"[{j}/{len(rows)}] ok={ok} fail={fail} skip={skip} "
                        f"last={sid} N={payload['n_tokens']} rate={rate:.2f}/s eta={eta:.1f}min",
                        flush=True,
                    )

            except Exception as e:
                fail += 1
                torch.cuda.empty_cache()
                gc.collect()
                with open(err_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "sample_id": sid,
                        "error": type(e).__name__,
                        "message": str(e)[:500],
                    }) + "\n")
                print(f"[{j}/{len(rows)}] FAIL {sid}: {type(e).__name__}: {str(e)[:120]}", flush=True)
    finally:
        cm.__exit__(None, None, None)

    summary = {
        "ok": ok,
        "failed": fail,
        "skipped_existing": skip,
        "rows_seen": len(rows),
        "elapsed_min": (time.time() - t_start) / 60,
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
    }
    write_json(logs / f"extract_summary_shard{args.shard_idx}.json", summary)
    print(json.dumps(summary, indent=2))


def load_trait_matrix(trait_vector_dir: Path, layer_idx: int) -> Tuple[torch.Tensor, List[str]]:
    files = sorted(trait_vector_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files in {trait_vector_dir}")

    default_layers = [15, 20, 25, 30, 35, 40, 45]
    vecs, names = [], []

    for p in files:
        d = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(d, dict):
            vec = d["vector"].float()
            name = d.get("trait", p.stem)
            vector_layers = d.get("vector_layers")
            row_to_model_layer = d.get("row_to_model_layer")
        else:
            vec = d.float()
            name = p.stem
            vector_layers = None
            row_to_model_layer = None

        if vec.shape[0] == 1:
            selected = vec[0]
        else:
            row = None
            if isinstance(vector_layers, list) and layer_idx in vector_layers:
                row = vector_layers.index(layer_idx)
            elif isinstance(row_to_model_layer, dict):
                for k, v in row_to_model_layer.items():
                    if int(v) == layer_idx:
                        row = int(k)
                        break
            elif vec.shape[0] == len(default_layers) and layer_idx in default_layers:
                row = default_layers.index(layer_idx)
            if row is None:
                raise ValueError(f"Could not map layer {layer_idx} to row for {p}, shape={tuple(vec.shape)}")
            selected = vec[row]

        selected = selected / (selected.norm() + 1e-8)
        vecs.append(selected)
        names.append(name)

    return torch.stack(vecs, dim=0).float(), names


def load_axis(axis_path: Optional[Path], layer_idx: int) -> Optional[torch.Tensor]:
    if axis_path is None or not axis_path.exists():
        return None

    d = torch.load(axis_path, map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        axis = d.get("axis", d.get("vector"))
        vector_layers = d.get("vector_layers")
        row_to_model_layer = d.get("row_to_model_layer")
    else:
        axis = d
        vector_layers = None
        row_to_model_layer = None

    if axis is None:
        return None

    axis = axis.float()
    default_layers = [15, 20, 25, 30, 35, 40, 45]

    if axis.shape[0] == 1:
        selected = axis[0]
    else:
        row = None
        if isinstance(vector_layers, list) and layer_idx in vector_layers:
            row = vector_layers.index(layer_idx)
        elif isinstance(row_to_model_layer, dict):
            for k, v in row_to_model_layer.items():
                if int(v) == layer_idx:
                    row = int(k)
                    break
        elif axis.shape[0] == len(default_layers) and layer_idx in default_layers:
            row = default_layers.index(layer_idx)
        if row is None:
            raise ValueError(f"Could not map layer {layer_idx} to axis row, shape={tuple(axis.shape)}")
        selected = axis[row]

    return selected / (selected.norm() + 1e-8)


def cyber_probe_label(category: Optional[str], probe_id: int) -> Optional[int]:
    if category is None:
        return None
    pos = {1: "dual_use", 2: "high_risk_dual_use", 3: "prohibited"}[probe_id]
    neg = {
        1: {"benign"},
        2: {"dual_use", "benign"},
        3: {"high_risk_dual_use", "dual_use", "benign"},
    }[probe_id]
    if category == pos:
        return 1
    if category in neg:
        return 0
    return None


def load_activation(extracts_dir: Path, sample_id: str) -> Optional[torch.Tensor]:
    p = extracts_dir / f"{sample_id}.pt"
    if not p.exists():
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    if "activation" in d:
        return d["activation"].float()
    return None


def build_features(rows: List[Dict[str, Any]], extracts_dir: Path, trait_mat: torch.Tensor, axis: Optional[torch.Tensor]):
    good_rows, feats = [], []
    skipped = 0

    for r in rows:
        h = load_activation(extracts_dir, r["sample_id"])
        if h is None or h.shape[-1] != trait_mat.shape[-1]:
            skipped += 1
            continue

        trait_proj = torch.mv(trait_mat, h.float())
        parts = [trait_proj]
        if axis is not None:
            parts.append(torch.dot(axis, h.float()).view(1))
        feats.append(torch.cat(parts).numpy().astype(np.float32))
        good_rows.append(r)

    if skipped:
        print(f"[warn] skipped {skipped} rows without usable activation")
    if not feats:
        raise RuntimeError("No usable features")
    return good_rows, np.stack(feats, axis=0)


def fit_eval(task_name: str, Xtr, ytr, Xte, yte, seed: int) -> Dict[str, Any]:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    clf = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(Xtr, ytr)

    scores = clf.decision_function(Xte)
    probs = 1.0 / (1.0 + np.exp(-scores))
    preds = (probs >= 0.5).astype(int)

    return {
        "task": task_name,
        "auc": float(roc_auc_score(yte, scores)),
        "acc": float(accuracy_score(yte, preds)),
        "f1": float(f1_score(yte, preds, zero_division=0)),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "train_pos": int(np.sum(ytr)),
        "train_neg": int(len(ytr) - np.sum(ytr)),
        "test_pos": int(np.sum(yte)),
        "test_neg": int(len(yte) - np.sum(yte)),
        "coef": clf.coef_[0].astype(np.float32),
        "intercept": float(clf.intercept_[0]),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
    }


def add_task(tasks, name, rows, X, labels, seed):
    tr_idx, te_idx, ytr, yte = [], [], [], []
    for i, (r, lab) in enumerate(zip(rows, labels)):
        if lab is None:
            continue
        if r.get("split") == "train":
            tr_idx.append(i); ytr.append(int(lab))
        elif r.get("split") == "test":
            te_idx.append(i); yte.append(int(lab))

    if len(ytr) == 0 or len(yte) == 0 or len(set(ytr)) < 2 or len(set(yte)) < 2:
        print(f"[warn] {name}: insufficient split/classes")
        return

    result = fit_eval(
        name,
        X[np.array(tr_idx)],
        np.array(ytr, dtype=int),
        X[np.array(te_idx)],
        np.array(yte, dtype=int),
        seed,
    )
    tasks[name] = result
    print(f"{name}: AUC={result['auc']:.4f} acc={result['acc']:.4f} f1={result['f1']:.4f} train={result['n_train']} test={result['n_test']}")


def safe_task(d):
    return {k: v for k, v in d.items() if k not in {"coef", "scaler_mean", "scaler_scale"}}


def run_train(args) -> None:
    rows = sorted(load_all_rows(args.repo_root), key=lambda r: r["sample_id"])
    extracts_dir = args.out_root / "extracts"

    trait_mat, trait_names = load_trait_matrix(args.trait_vector_dir, args.layer_idx)
    axis = load_axis(args.assistant_axis_path, args.layer_idx)

    print(f"rows={len(rows)} extracts_dir={extracts_dir}")
    print(f"trait_mat={tuple(trait_mat.shape)} axis={'yes' if axis is not None else 'no'}")

    good_rows, X = build_features(rows, extracts_dir, trait_mat, axis)
    print(f"X={X.shape} usable_rows={len(good_rows)}")

    tasks = {}

    refusal_labels = [
        int(r["is_refusal"]) if r.get("dataset") == "refusal_gemma4_31b" and "is_refusal" in r else None
        for r in good_rows
    ]
    add_task(tasks, "refusal_gemma4_31b", good_rows, X, refusal_labels, args.seed)

    for pid in [1, 2, 3]:
        labels = []
        for r in good_rows:
            if r.get("dataset") != "cyber":
                labels.append(None)
            else:
                labels.append(cyber_probe_label(r.get("category") or r.get("label"), pid))

        name = {
            1: "cyber_probe_1_dual_use_vs_benign",
            2: "cyber_probe_2_hdu_vs_du_benign",
            3: "cyber_probe_3_prohibited_vs_rest",
        }[pid]
        add_task(tasks, name, good_rows, X, labels, args.seed)

    if not tasks:
        raise RuntimeError("No tasks trained")

    aucs = np.array([t["auc"] for t in tasks.values()], dtype=float)

    summary = {k: safe_task(v) for k, v in tasks.items()}
    summary["_mean_auc_over_trained_tasks"] = float(aucs.mean())
    summary["_std_auc_over_trained_tasks"] = float(aucs.std())
    summary["_var_auc_over_trained_tasks"] = float(aucs.var())
    summary["_tasks_trained"] = list(tasks.keys())

    package = {
        "method": "gemma_layer30_trait_projection_logistic_regression",
        "layer_idx": args.layer_idx,
        "trait_names": trait_names,
        "uses_axis": axis is not None,
        "feature_dim": int(X.shape[1]),
        "tasks": tasks,
        "summary": summary,
    }

    args.out_root.mkdir(parents=True, exist_ok=True)
    torch.save(package, args.out_root / "gemma_layer30_trait_projection_probe.pt")
    write_json(args.out_root / "summary.json", summary)

    interp = {}
    for task_name, result in tasks.items():
        coef = result["coef"]
        n = len(trait_names)
        trait_coef = coef[:n]
        top_pos = np.argsort(trait_coef)[-20:][::-1]
        top_neg = np.argsort(trait_coef)[:20]
        interp[task_name] = {
            "top_positive": [{"trait": trait_names[i], "coef": float(trait_coef[i])} for i in top_pos],
            "top_negative": [{"trait": trait_names[i], "coef": float(trait_coef[i])} for i in top_neg],
            "assistant_axis_coef": float(coef[n]) if len(coef) > n else None,
        }
    write_json(args.out_root / "top_trait_coefficients.json", interp)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {args.out_root / 'summary.json'}")
    print(f"Saved {args.out_root / 'top_trait_coefficients.json'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["extract", "train", "all"], required=True)
    p.add_argument("--repo_root", type=Path, default=Path("/scratch/mechhack"))
    p.add_argument("--model_path", type=str, default="/data/Gemma-4-31B-it")
    p.add_argument("--out_root", type=Path, default=Path("/scratch/hackathon_traitproj_layer30_all"))
    p.add_argument("--trait_vector_dir", type=Path, default=Path("/scratch/gemma_trait_output/traits40_vectors/pre_generation_last_token/all_traits_no_filter"))
    p.add_argument("--assistant_axis_path", type=Path, default=Path("/scratch/gemma_trait_output/traits40_axes/pre_generation_last_token/all_traits_no_filter/assistant_axis_pc1.pt"))
    p.add_argument("--layer_idx", type=int, default=30)
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--sample_limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite_extracts", action="store_true")
    p.add_argument("--no_chat_template", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode in ("extract", "all"):
        run_extract(args)
    if args.mode in ("train", "all"):
        run_train(args)


if __name__ == "__main__":
    main()
