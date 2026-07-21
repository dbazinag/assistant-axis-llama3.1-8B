#!/usr/bin/env python3
"""
build_llama_detection_table.py

Regenerates the consolidated Llama-3.1-8B detection/transfer table (paper Table 1)

AUCs are reported with the paper's per-cell orientation, max(auc, 1-auc), so symmetric
detectors that land below chance on a family (e.g. JBShield/PAP, Perplexity) are read the
right way up.

Run from the project root:  python3 full_trait_tools/build_llama_detection_table.py
"""
import json
import numpy as np
from pathlib import Path

OUT_DIR = Path("full_trait_output")
FAM = ["GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]
COLS = ["HarmBench"] + FAM
TRAIT_CONFIG = "logreg_l2_C10.0_raw"


def orient(v):
    """Paper convention: report max(auc, 1-auc) for symmetric detectors."""
    return None if v is None else round(max(v, 1.0 - v), 3)


def ds_auc(node):
    """Pull HarmBench..PEZ AUCs from a {dataset: {auc:{mean}}} or {dataset: scalar} node."""
    out = []
    for k in COLS:
        v = node.get(k)
        if isinstance(v, dict):
            v = v["auc"]["mean"] if isinstance(v.get("auc"), dict) else v.get("auc")
        out.append(v)
    return out


def trait_row():
    d = json.load(open(OUT_DIR / "all_traits_sweep_v2/results.json"))["summary"][TRAIT_CONFIG]
    return [d["human_test"]["auc"]["mean"]] + [d[f]["auc"]["mean"] for f in FAM]


def fast_transfer_row(mode):
    allf = json.load(open(OUT_DIR / "transfer_results_all/transfer_results_all.json"))["modes"][mode]
    hb, gcg, pair = allf["human_test"]["mean"], allf["transfer_gcg"]["mean"], allf["transfer_pair"]["mean"]
    fmap = {"PAP": "transfer_results_pap", "GPTFuzz": "transfer_results_gptfuzz", "PEZ": "transfer_results_pez"}
    fams = {}
    for fam, d in fmap.items():
        node = json.load(open(OUT_DIR / d / "transfer_results.json"))["modes"][mode]
        key = next(k for k in node if k.startswith("transfer_"))
        fams[fam] = node[key]["mean"]
    return [hb, gcg, pair, fams["PAP"], fams["GPTFuzz"], fams["PEZ"]]


def main():
    rows = {}
    rows["Trait projections"] = (trait_row(),
        f"all_traits_sweep_v2/results.json :: summary.{TRAIT_CONFIG}")
    rows["Raw activations"] = (fast_transfer_row("raw"),
        "transfer_results_all + transfer_results_{pap,pez,gptfuzz} :: modes.raw")
    rows["PCA (3)"] = (fast_transfer_row("pca"),
        "transfer_results_all + transfer_results_{pap,pez,gptfuzz} :: modes.pca")

    cg = json.load(open(OUT_DIR / "baselines_contrastive_geometry/contrastive_geometry_results.json"))
    rows["Mahalanobis contrastive"] = (ds_auc(cg["methods"]["mcd"]["datasets"]),
        "baselines_contrastive_geometry :: methods.mcd")
    jlt = json.load(open(OUT_DIR / "baselines_jlt_hidden_tensor/jlt_hidden_tensor_results.json"))
    rows["Jailbreak Leaves a Trace (SVM)"] = (ds_auc(jlt["models"]["pca_svm_rbf"]["datasets"]),
        "baselines_jlt_hidden_tensor :: pca_svm_rbf")
    rows["Jailbreak Leaves a Trace (RF)"] = (ds_auc(jlt["models"]["pca_random_forest"]["datasets"]),
        "baselines_jlt_hidden_tensor :: pca_random_forest")
    jb = json.load(open(OUT_DIR / "baselines_jbshield_fjd/jbshield_fjd_results.json"))
    rows["JBShield"] = (ds_auc(jb["results"]["jbshield_mean_direction"]["datasets"]),
        "baselines_jbshield_fjd :: jbshield_mean_direction")
    gs = json.load(open(OUT_DIR / "baselines_all_attacks/gradsafe_all_attacks.json"))["results"]
    rows["GradSafe"] = ([gs[k]["auc"] for k in COLS], "baselines_all_attacks/gradsafe_all_attacks.json")
    lg = json.load(open(OUT_DIR / "baselines_all_attacks/llama_guard_all_attacks.json"))
    rows["Llama Guard (input only)"] = ([lg[k]["input_only_auc"] for k in COLS],
        "baselines_all_attacks/llama_guard_all_attacks.json :: input_only_auc")
    se = json.load(open(OUT_DIR / "baselines_all_attacks/self_exam_all_attacks.json"))
    rows["Verbalized (no CoT)"] = ([se[k]["direct_auc"] for k in COLS],
        "baselines_all_attacks/self_exam_all_attacks.json :: direct_auc")
    rows["Verbalized (w/ CoT)"] = ([se[k]["cot_auc"] for k in COLS],
        "baselines_all_attacks/self_exam_all_attacks.json :: cot_auc")
    ppl = json.load(open(OUT_DIR / "baselines_harmbench_ppl_smoothllm/harmbench_ppl_smoothllm_results.json"))
    rows["Perplexity"] = (ds_auc(ppl["results"]["perplexity_nll"]["datasets"]),
        "baselines_harmbench_ppl_smoothllm :: perplexity_nll")

    out = {"model": "meta-llama/Llama-3.1-8B-Instruct", "layer": 16, "metric": "AUC",
           "orientation": "per-cell max(auc, 1-auc)", "families": FAM, "columns": COLS, "methods": {}}
    for name, (vals, src) in rows.items():
        v = [orient(x) for x in vals]
        out["methods"][name] = {
            "source": src,
            **{COLS[i]: v[i] for i in range(len(COLS))},
            "Avg_AUC": round(float(np.nanmean([x for x in v if x is not None])), 3),
            "Avg_Transfer": round(float(np.nanmean([x for x in v[1:] if x is not None])), 3),
        }

    dest = OUT_DIR / "llama_detection_table_consolidated.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    for name, m in out["methods"].items():
        print(f"  {name:32s} " + " ".join(f"{m[c]}" for c in COLS) +
              f"  | AvgAUC={m['Avg_AUC']} Xfer={m['Avg_Transfer']}")


if __name__ == "__main__":
    main()
