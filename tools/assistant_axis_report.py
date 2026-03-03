#!/usr/bin/env python3
"""
assistant_axis_report.py

A lightweight analysis / reporting script for the Assistant-Axis pipeline outputs.
Reads your (responses, scores, activations, vectors, axis) and produces:

- A JSON summary report with counts + key stats
- CSVs:
  - role_scores_summary.csv (per-role counts of 0/1/2/3 and 3-rate)
  - vectors_summary.csv (per-role vector meta + norms)
  - axis_layer_norms.csv (per-layer norms)
- PNG plots:
  - score_distribution_overall.png
  - score3_rate_by_role.png
  - axis_layer_norms.png
  - vector_layer_norms_default_vs_role_mean.png
  - axis_default_rolemean_cosine_by_layer.png

CPU-only. No OpenAI calls. No GPU needed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Optional deps (we’ll fail gracefully with clear messages)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import pandas as pd
except Exception:
    pd = None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _torch_load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> float:
    # a,b: (hidden_dim,)
    a = a.float()
    b = b.float()
    na = torch.norm(a).item()
    nb = torch.norm(b).item()
    if na < eps or nb < eps:
        return float("nan")
    return float(torch.dot(a, b).item() / (na * nb))


@dataclass
class RoleScoreStats:
    role: str
    n_total: int
    n0: int
    n1: int
    n2: int
    n3: int

    @property
    def rate3(self) -> float:
        return (self.n3 / self.n_total) if self.n_total else 0.0


def load_scores_dir(scores_dir: Path) -> Dict[str, Dict[str, int]]:
    """
    Returns: role -> { key -> score }
    """
    scores: Dict[str, Dict[str, int]] = {}
    if not scores_dir.exists():
        return scores
    for f in sorted(scores_dir.glob("*.json")):
        role = f.stem
        try:
            d = _read_json(f)
            dd: Dict[str, int] = {}
            for k, v in d.items():
                iv = _safe_int(v)
                if iv is None:
                    continue
                if iv < 0 or iv > 3:
                    continue
                dd[k] = iv
            scores[role] = dd
        except Exception:
            continue
    return scores


def summarize_scores(scores_by_role: Dict[str, Dict[str, int]]) -> Tuple[List[RoleScoreStats], Dict[str, int]]:
    per_role: List[RoleScoreStats] = []
    overall = {"n_total": 0, "n0": 0, "n1": 0, "n2": 0, "n3": 0}
    for role, d in scores_by_role.items():
        c0 = sum(1 for v in d.values() if v == 0)
        c1 = sum(1 for v in d.values() if v == 1)
        c2 = sum(1 for v in d.values() if v == 2)
        c3 = sum(1 for v in d.values() if v == 3)
        n = len(d)
        per_role.append(RoleScoreStats(role=role, n_total=n, n0=c0, n1=c1, n2=c2, n3=c3))
        overall["n_total"] += n
        overall["n0"] += c0
        overall["n1"] += c1
        overall["n2"] += c2
        overall["n3"] += c3
    per_role.sort(key=lambda r: r.rate3)
    return per_role, overall


def count_jsonl_rows(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def summarize_responses_dir(responses_dir: Path) -> Dict[str, int]:
    """
    Returns summary counts. Does not parse JSONL, just counts files and lines.
    """
    out = {"n_files": 0, "n_rows_total": 0}
    if not responses_dir.exists():
        return out
    files = sorted(responses_dir.glob("*.jsonl"))
    out["n_files"] = len(files)
    total = 0
    for f in files:
        total += count_jsonl_rows(f)
    out["n_rows_total"] = total
    return out


def summarize_activations_dir(activations_dir: Path) -> Dict[str, int]:
    """
    Counts .pt activation files.
    """
    out = {"n_files": 0}
    if not activations_dir.exists():
        return out
    out["n_files"] = len(list(activations_dir.glob("*.pt")))
    return out


def load_vectors_dir(vectors_dir: Path) -> Dict[str, dict]:
    """
    Returns: role -> saved_data (expects {"vector": Tensor, "type": str, "role": str})
    """
    vectors: Dict[str, dict] = {}
    if not vectors_dir.exists():
        return vectors
    for f in sorted(vectors_dir.glob("*.pt")):
        try:
            d = _torch_load(f)
            role = str(d.get("role", f.stem))
            if "vector" not in d:
                continue
            vectors[role] = d
        except Exception:
            continue
    return vectors


def vector_layer_norms(vec: torch.Tensor) -> torch.Tensor:
    # vec: (n_layers, hidden_dim)
    # Ensure float dtype so matplotlib + numpy are happy (bf16 causes TypeError)
    return torch.norm(vec.float(), dim=1)


def mean_vector(vectors: List[torch.Tensor]) -> torch.Tensor:
    # keep compute stable even if inputs are bf16
    return torch.stack([v.float() for v in vectors]).mean(dim=0)


def plot_or_skip(fig_path: Path) -> bool:
    if plt is None:
        return False
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses_dir", type=str, required=False, default="")
    ap.add_argument("--scores_dir", type=str, required=False, default="")
    ap.add_argument("--activations_dir", type=str, required=False, default="")
    ap.add_argument("--vectors_dir", type=str, required=False, default="")
    ap.add_argument("--axis_path", type=str, required=False, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--min_count", type=int, default=30, help="Used for pass/fail summaries only (Step 4 threshold).")
    ap.add_argument("--top_k_roles", type=int, default=25, help="How many roles to show in some plots.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    responses_dir = Path(args.responses_dir) if args.responses_dir else None
    scores_dir = Path(args.scores_dir) if args.scores_dir else None
    activations_dir = Path(args.activations_dir) if args.activations_dir else None
    vectors_dir = Path(args.vectors_dir) if args.vectors_dir else None
    axis_path = Path(args.axis_path) if args.axis_path else None

    report: dict = {
        "paths": {
            "responses_dir": str(responses_dir) if responses_dir else "",
            "scores_dir": str(scores_dir) if scores_dir else "",
            "activations_dir": str(activations_dir) if activations_dir else "",
            "vectors_dir": str(vectors_dir) if vectors_dir else "",
            "axis_path": str(axis_path) if axis_path else "",
            "out_dir": str(out_dir),
        },
        "counts": {},
        "scores": {},
        "vectors": {},
        "axis": {},
        "notes": [],
    }

    # ---------- Responses ----------
    if responses_dir and responses_dir.exists():
        resp_summary = summarize_responses_dir(responses_dir)
        report["counts"]["responses"] = resp_summary
    else:
        report["notes"].append("responses_dir missing or not provided")

    # ---------- Scores ----------
    scores_by_role: Dict[str, Dict[str, int]] = {}
    per_role_scores: List[RoleScoreStats] = []
    overall_scores: Dict[str, int] = {}
    if scores_dir and scores_dir.exists():
        scores_by_role = load_scores_dir(scores_dir)
        per_role_scores, overall_scores = summarize_scores(scores_by_role)
        report["counts"]["scores"] = {"n_files": len(scores_by_role)}
        report["scores"]["overall"] = overall_scores
        report["scores"]["per_role_min_rate3"] = per_role_scores[0].rate3 if per_role_scores else None
        report["scores"]["per_role_max_rate3"] = per_role_scores[-1].rate3 if per_role_scores else None
        report["scores"]["min_count_threshold"] = args.min_count

        if per_role_scores:
            n_pass = sum(1 for r in per_role_scores if r.n3 >= args.min_count)
            n_fail = sum(1 for r in per_role_scores if r.n3 < args.min_count)
            report["scores"]["roles_pass_min_count"] = n_pass
            report["scores"]["roles_fail_min_count"] = n_fail
    else:
        report["notes"].append("scores_dir missing or not provided")

    # ---------- Activations ----------
    if activations_dir and activations_dir.exists():
        act_summary = summarize_activations_dir(activations_dir)
        report["counts"]["activations"] = act_summary
    else:
        report["notes"].append("activations_dir missing or not provided")

    # ---------- Vectors ----------
    vectors: Dict[str, dict] = {}
    default_vectors: List[torch.Tensor] = []
    role_vectors: List[torch.Tensor] = []
    default_roles: List[str] = []
    role_roles: List[str] = []
    if vectors_dir and vectors_dir.exists():
        vectors = load_vectors_dir(vectors_dir)
        report["counts"]["vectors"] = {"n_files": len(vectors)}

        for role, d in vectors.items():
            v = d["vector"]
            vtype = str(d.get("type", "unknown"))
            if ("default" in role.lower()) or (vtype == "mean"):
                default_vectors.append(v)
                default_roles.append(role)
            else:
                role_vectors.append(v)
                role_roles.append(role)

        report["vectors"]["n_default_vectors"] = len(default_vectors)
        report["vectors"]["n_role_vectors"] = len(role_vectors)
        report["vectors"]["default_roles"] = default_roles

        vec_rows = []
        for role, d in vectors.items():
            v = d["vector"]
            vtype = str(d.get("type", "unknown"))
            ln = vector_layer_norms(v)
            vec_rows.append(
                {
                    "role": role,
                    "type": vtype,
                    "n_layers": int(v.shape[0]),
                    "hidden_dim": int(v.shape[1]),
                    "mean_layer_norm": float(ln.mean().item()),
                    "max_layer_norm": float(ln.max().item()),
                    "max_layer_norm_layer": int(torch.argmax(ln).item()),
                }
            )

        report["vectors"]["role_norms_summary"] = {
            "mean_of_mean_layer_norm": float(sum(r["mean_layer_norm"] for r in vec_rows) / max(1, len(vec_rows))),
            "mean_of_max_layer_norm": float(sum(r["max_layer_norm"] for r in vec_rows) / max(1, len(vec_rows))),
        }

        # Save CSV
        if pd is not None:
            pd.DataFrame(vec_rows).sort_values(["type", "role"]).to_csv(out_dir / "vectors_summary.csv", index=False)
        else:
            csv_path = out_dir / "vectors_summary.csv"
            with csv_path.open("w", encoding="utf-8") as f:
                cols = list(vec_rows[0].keys()) if vec_rows else []
                f.write(",".join(cols) + "\n")
                for r in vec_rows:
                    f.write(",".join(str(r[c]) for c in cols) + "\n")

        if default_vectors and role_vectors:
            default_mean = mean_vector(default_vectors)
            role_mean = mean_vector(role_vectors)

            dm_ln = vector_layer_norms(default_mean)
            rm_ln = vector_layer_norms(role_mean)

            report["vectors"]["default_mean_shape"] = [int(default_mean.shape[0]), int(default_mean.shape[1])]
            report["vectors"]["role_mean_shape"] = [int(role_mean.shape[0]), int(role_mean.shape[1])]
            report["vectors"]["default_mean_layer_norm_mean"] = float(dm_ln.mean().item())
            report["vectors"]["role_mean_layer_norm_mean"] = float(rm_ln.mean().item())

            cos_by_layer: List[float] = []
            for i in range(default_mean.shape[0]):
                cos_by_layer.append(_cosine(default_mean[i], role_mean[i]))
            report["vectors"]["cosine_default_mean_vs_role_mean_by_layer"] = {
                "mean": float(torch.tensor([c for c in cos_by_layer if not math.isnan(c)]).mean().item())
                if cos_by_layer
                else None,
                "min": float(min(cos_by_layer)) if cos_by_layer else None,
                "max": float(max(cos_by_layer)) if cos_by_layer else None,
            }

            if plot_or_skip(out_dir / "vector_layer_norms_default_vs_role_mean.png"):
                plt.figure()
                plt.plot(dm_ln.float().numpy(), label="default_mean layer norm")
                plt.plot(rm_ln.float().numpy(), label="role_mean layer norm")
                plt.title("Layer norms: default_mean vs role_mean")
                plt.xlabel("Layer")
                plt.ylabel("L2 norm")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "vector_layer_norms_default_vs_role_mean.png")
                plt.close()

            if plot_or_skip(out_dir / "axis_default_rolemean_cosine_by_layer.png"):
                plt.figure()
                plt.plot(cos_by_layer)
                plt.title("Cosine similarity by layer: default_mean vs role_mean")
                plt.xlabel("Layer")
                plt.ylabel("cosine(default_mean, role_mean)")
                plt.tight_layout()
                plt.savefig(out_dir / "axis_default_rolemean_cosine_by_layer.png")
                plt.close()
        else:
            report["notes"].append("Could not compute default_mean/role_mean (missing default_vectors or role_vectors).")
    else:
        report["notes"].append("vectors_dir missing or not provided")

    # ---------- Axis ----------
    axis: Optional[torch.Tensor] = None
    if axis_path and axis_path.exists():
        try:
            axis = _torch_load(axis_path)
            if isinstance(axis, dict) and "axis" in axis:
                axis = axis["axis"]
            if not torch.is_tensor(axis):
                raise ValueError("axis file did not contain a Tensor")

            axis = axis.float()

            report["axis"]["shape"] = [int(axis.shape[0]), int(axis.shape[1])]
            norms = vector_layer_norms(axis)
            report["axis"]["mean_layer_norm"] = float(norms.mean().item())
            report["axis"]["max_layer_norm"] = float(norms.max().item())
            report["axis"]["argmax_layer_norm_layer"] = int(torch.argmax(norms).item())

            rows = [{"layer": i, "axis_layer_norm": float(n.item())} for i, n in enumerate(norms)]
            if pd is not None:
                pd.DataFrame(rows).to_csv(out_dir / "axis_layer_norms.csv", index=False)
            else:
                with (out_dir / "axis_layer_norms.csv").open("w", encoding="utf-8") as f:
                    f.write("layer,axis_layer_norm\n")
                    for r in rows:
                        f.write(f"{r['layer']},{r['axis_layer_norm']}\n")

            if plot_or_skip(out_dir / "axis_layer_norms.png"):
                plt.figure()
                plt.plot(norms.float().numpy())
                plt.title("Assistant Axis: per-layer L2 norms")
                plt.xlabel("Layer")
                plt.ylabel("L2 norm")
                plt.tight_layout()
                plt.savefig(out_dir / "axis_layer_norms.png")
                plt.close()

            # If we have default_mean and role_mean, sanity check axis = default_mean - role_mean
            if vectors_dir and vectors_dir.exists():
                vectors = vectors if vectors else load_vectors_dir(vectors_dir)
                dv: List[torch.Tensor] = []
                rv: List[torch.Tensor] = []
                for role, d in vectors.items():
                    v = d["vector"]
                    vtype = str(d.get("type", "unknown"))
                    if ("default" in role.lower()) or (vtype == "mean"):
                        dv.append(v)
                    else:
                        rv.append(v)
                if dv and rv:
                    dm = mean_vector(dv)
                    rm = mean_vector(rv)
                    axis_recon = (dm - rm).float()
                    diff = torch.norm(axis - axis_recon).item()
                    rel = diff / (torch.norm(axis_recon).item() + 1e-9)
                    report["axis"]["reconstruction_check"] = {
                        "l2_diff": float(diff),
                        "relative_l2_diff": float(rel),
                        "ok_if_rel_diff_lt": 1e-6,
                        "passes": bool(rel < 1e-6),
                    }
        except Exception as e:
            report["notes"].append(f"Failed to load axis: {e}")
    else:
        report["notes"].append("axis_path missing or not provided")

    # ---------- Score plots + CSV ----------
    if scores_by_role:
        rows = []
        for r in per_role_scores:
            rows.append(
                {
                    "role": r.role,
                    "n_total": r.n_total,
                    "n0": r.n0,
                    "n1": r.n1,
                    "n2": r.n2,
                    "n3": r.n3,
                    "rate3": r.rate3,
                    "passes_min_count": int(r.n3 >= args.min_count),
                }
            )
        if pd is not None:
            pd.DataFrame(rows).sort_values("rate3").to_csv(out_dir / "role_scores_summary.csv", index=False)
        else:
            with (out_dir / "role_scores_summary.csv").open("w", encoding="utf-8") as f:
                f.write("role,n_total,n0,n1,n2,n3,rate3,passes_min_count\n")
                for rr in rows:
                    f.write(
                        f"{rr['role']},{rr['n_total']},{rr['n0']},{rr['n1']},{rr['n2']},{rr['n3']},{rr['rate3']},{rr['passes_min_count']}\n"
                    )

        if plot_or_skip(out_dir / "score_distribution_overall.png"):
            plt.figure()
            counts = [
                overall_scores.get("n0", 0),
                overall_scores.get("n1", 0),
                overall_scores.get("n2", 0),
                overall_scores.get("n3", 0),
            ]
            plt.bar([0, 1, 2, 3], counts)
            plt.title("Overall judge score distribution")
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(out_dir / "score_distribution_overall.png")
            plt.close()

        if plot_or_skip(out_dir / "score3_rate_by_role.png"):
            sorted_roles = sorted(rows, key=lambda x: x["rate3"])
            worst = sorted_roles[: min(args.top_k_roles, len(sorted_roles))]
            best = sorted_roles[-min(args.top_k_roles, len(sorted_roles)) :]

            def _plot_block(block, title, fname):
                labels = [b["role"] for b in block]
                vals = [b["rate3"] for b in block]
                plt.figure(figsize=(10, max(4, len(block) * 0.25)))
                plt.barh(labels, vals)
                plt.title(title)
                plt.xlabel("Score-3 rate")
                plt.tight_layout()
                plt.savefig(out_dir / fname)
                plt.close()

            _plot_block(worst, f"Worst {len(worst)} roles by score-3 rate", "score3_rate_by_role_worst.png")
            _plot_block(best, f"Best {len(best)} roles by score-3 rate", "score3_rate_by_role_best.png")

    # ---------- Save JSON report ----------
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Minimal stdout summary
    print("=" * 60)
    print("Assistant-Axis Pipeline Report")
    print("=" * 60)
    print(f"Saved to: {out_dir}")
    if "responses" in report["counts"]:
        print(f"Responses: {report['counts']['responses']}")
    if "scores" in report["counts"]:
        print(f"Scores files: {report['counts']['scores']['n_files']}")
        print(f"Scores overall: {report['scores'].get('overall', {})}")
        print(f"Roles pass min_count({args.min_count}): {report['scores'].get('roles_pass_min_count')}")
    if "vectors" in report["counts"]:
        print(f"Vectors files: {report['counts']['vectors']['n_files']}")
        print(
            f"Default vectors: {report['vectors'].get('n_default_vectors')}, role vectors: {report['vectors'].get('n_role_vectors')}"
        )
    if report["axis"].get("shape"):
        print(f"Axis shape: {report['axis']['shape']}")
        rc = report["axis"].get("reconstruction_check")
        if rc:
            print(f"Axis reconstruction passes: {rc.get('passes')} (rel diff {rc.get('relative_l2_diff')})")
    if report["notes"]:
        print("NOTES:")
        for n in report["notes"]:
            print(f"  - {n}")
    print("=" * 60)


if __name__ == "__main__":
    main()