#!/usr/bin/env python3
"""
trait_steering_attack_jbb_persona_w.py

Steers the raw JBB goals on Llama-3.1-8B using the PERSONA CLASSIFIER's decision
direction `w`, mapped from trait-projection space into 4096-dim activation space —
rather than the raw-activation hyperplane used by trait_steering_attack_jbb.py --mode w.

The persona classifier scores an activation x via proj = x @ trait_matrix.T, then
StandardScaler -> LogisticRegression. Because that is linear in x, its decision normal
in activation space is:

    w_act = (clf.coef_[0] / scaler.scale_) @ trait_matrix      # [hidden_dim]
    w_vec = w_act / ||w_act||

+w_vec is the compliance/jailbreak direction (coef_ points at class 1 = jailbroken),
so we steer +w_vec and escalate alpha — same fixed-vector path as the base --mode w.

Two base classifiers are compared (identical recipe, only the training data differs):
  harmbench — logreg on HarmBench human_jailbreak trait projections (reference)
  jbb       — logreg on JBB attack (PAIR/PAP/GPTFuzz/PEZ) trait projections

The steering TARGET is the raw JBB goals (100 pairs, no templates) for both — JBB has
no templates of its own; the base script's templates come from HarmBench.

Usage:
  uv run python jailbreaking_tools/trait_steering_attack_jbb_persona_w.py --clf_source both
  uv run python jailbreaking_tools/trait_steering_attack_jbb_persona_w.py --clf_source jbb --test
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the base attack's machinery (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trait_steering_attack_jbb as base
from trait_steering_attack_jbb import (
    LAYER,
    MODEL_NAME,
    RANDOM_SEED,
    REPO_ROOT,
    compute_category_metrics,
    compute_metrics,
    fetch_harmbench_behaviors,
    fetch_jbb_behaviors,
    fetch_wildjailbreak_behaviors,
    fit_classifier,
    is_degenerate,
    judge_response_gpt41mini,
    judge_response_jbb,
    load_jsonl,
    load_trait_matrix_and_names,
    logger,
    run_llama_steered,
)

# ── Classifier training-data sources ────────────────────────────────────────────
# (classified_responses path, activations path, harmbench-only attack_type filter)
CLF_SOURCES = {
    "harmbench": {
        "classified": REPO_ROOT / "full_trait_output/harmbench_activations/classified_responses.jsonl",
        "acts":       REPO_ROOT / "full_trait_output/harmbench_activations/activations.pt",
        "filter_human_jailbreak": True,
    },
    "jbb": {
        "classified": REPO_ROOT / "full_trait_output/jbb_attack_activations/classified_responses.jsonl",
        "acts":       REPO_ROOT / "full_trait_output/jbb_attack_activations/activations.pt",
        "filter_human_jailbreak": False,
    },
}


def build_proj_features(classified_path, acts_path, trait_matrix, filter_human_jailbreak):
    """Layer-16 activations projected onto the trait matrix, plus jailbroken labels."""
    rows = load_jsonl(classified_path)
    if filter_human_jailbreak:
        rows = [r for r in rows if r.get("attack_type") == "human_jailbreak"]
    acts = torch.load(acts_path, map_location="cpu", weights_only=False)

    layer_key = str(LAYER)
    X_list, y_list = [], []
    for row in rows:
        pid = row.get("pair_id")
        jb = row.get("jailbroken")
        if pid is None or jb is None:
            continue
        item = acts.get(pid)
        if item is None or layer_key not in item:
            continue
        act = item[layer_key].float().numpy()
        X_list.append(act @ trait_matrix.T)
        y_list.append(1 if jb else 0)

    if not X_list:
        raise RuntimeError(f"No labeled rows with activations found in {classified_path}")
    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64)


def persona_w(clf, scaler, trait_matrix):
    """Map the trait-projection classifier's decision normal into activation space."""
    coef = clf.coef_[0]                                   # [n_traits]
    w_act = (coef / scaler.scale_) @ trait_matrix         # [hidden_dim]
    norm = np.linalg.norm(w_act)
    if norm < 1e-12:
        raise RuntimeError("persona w has near-zero norm")
    return (w_act / norm).astype(np.float32)


def cv_auc(X, y):
    """Quick 5-fold CV AUC for sanity (same recipe as fit_classifier)."""
    try:
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, C=1.0),
        )
        return float(cross_val_score(pipe, X, y, cv=5, scoring="roc_auc").mean())
    except Exception as e:  # pragma: no cover - tiny/degenerate data
        logger.warning(f"  CV AUC failed: {e}")
        return float("nan")


WJB_CLF_PATH = REPO_ROOT / "full_trait_output/wildjailbreak_clf/best_model.pkl"


def wjb_persona_w():
    """Derive the persona w from the existing WJB classifier (trained on
    wildjailbreak_activations). Same coef -> activation-space mapping as train_source,
    using the pickle's own scaler/trait_matrix and its jailbreak-orientation sign so
    the direction points toward jailbreak-promoting (consistent with harmbench/jbb).
    """
    import pickle
    if not WJB_CLF_PATH.exists():
        raise SystemExit(f"ERROR: missing WJB classifier at {WJB_CLF_PATH}")
    with open(WJB_CLF_PATH, "rb") as fh:
        art = pickle.load(fh)
    pipe  = art["pipeline"]
    coef  = pipe.named_steps["clf"].coef_[0]
    scale = pipe.named_steps["sc"].scale_
    sign  = art.get("sign", 1)
    tm    = art["trait_matrix"]
    w_act = (sign * coef / scale) @ tm
    norm  = np.linalg.norm(w_act)
    if norm < 1e-12:
        raise RuntimeError("WJB persona w has near-zero norm")
    w_vec = (w_act / norm).astype(np.float32)
    meta  = art.get("meta", {})
    logger.info(f"[clf:wildjailbreak] persona w from {WJB_CLF_PATH.name} "
                f"(unit-norm={np.linalg.norm(w_vec):.4f}, sign={sign})")
    info = {"n": int(meta.get("n_train", meta.get("n", 0))),
            "jb_rate": float(meta.get("jb_rate", 0.0)),
            "cv_auc": float(meta.get("val_auc", meta.get("cv_auc", 0.0))),
            "source_pickle": str(WJB_CLF_PATH)}
    return w_vec, None, None, info


def train_source(source, trait_matrix):
    """Train the persona classifier for one data source; return (w_vec, clf, scaler, info)."""
    if source == "wildjailbreak":
        return wjb_persona_w()
    cfg = CLF_SOURCES[source]
    classified_path, acts_path = Path(cfg["classified"]), Path(cfg["acts"])
    if not classified_path.exists() or not acts_path.exists():
        raise SystemExit(
            f"ERROR: missing {source} classifier data.\n"
            f"  expected: {classified_path}\n"
            f"            {acts_path}\n"
            + ("  Produce the Llama JBB set with:\n"
               "    collect_jbb_attack_activations.py --model meta-llama/Llama-3.1-8B-Instruct "
               "--output_dir full_trait_output/jbb_attack_activations\n"
               "  then judge it into classified_responses.jsonl.\n"
               if source == "jbb" else "")
        )

    logger.info(f"[clf:{source}] loading + projecting features ...")
    X, y = build_proj_features(classified_path, acts_path, trait_matrix,
                               cfg["filter_human_jailbreak"])
    auc = cv_auc(X, y)
    logger.info(f"[clf:{source}] X={X.shape}  jb_rate={y.mean():.3f}  cv_auc={auc:.4f}")

    clf, scaler = fit_classifier(X, y)
    w_vec = persona_w(clf, scaler, trait_matrix)
    logger.info(f"[clf:{source}] persona w computed (unit-norm={np.linalg.norm(w_vec):.4f})")
    info = {"n": int(len(y)), "jb_rate": float(y.mean()), "cv_auc": auc}
    return w_vec, clf, scaler, info


def build_raw_goal_pool(behaviors, test_mode):
    """100 raw JBB goals (3 in test mode); no templates."""
    subset = behaviors[:3] if test_mode else behaviors
    return [{
        "pair_id":       beh["behavior_id"],
        "behavior_id":   beh["behavior_id"],
        "goal":          beh["goal"],
        "category":      beh["category"],
        "source":        beh["source"],
        "jailbreak_idx": -1,
        "prompt":        beh["goal"],
    } for beh in subset]


def escalate_until_jailbroken(pair_id, prompt, goal, w_vec,
                              model, tokenizer, device, client):
    """Steer +w_vec, judge with GPT-4.1-mini, escalate alpha; stop at first jailbreak.

    Escalation is driven ONLY by the in-loop GPT judge — no classifier-score early stop.
    """
    history = []
    min_alpha = None
    for iteration, alpha in enumerate(base.ALPHA_SCHEDULE):
        logger.info(f"    iter {iteration+1}/{len(base.ALPHA_SCHEDULE)} | alpha={alpha}")
        response, _ = run_llama_steered(model, tokenizer, prompt, w_vec, alpha, device)
        if response is None:
            logger.warning("    generation failed — stopping escalation")
            break
        degenerate = is_degenerate(response)
        is_jb = False
        if not degenerate:
            is_jb = judge_response_gpt41mini(client, prompt, response)
        logger.info(f"    judge: {'JAILBROKEN' if is_jb else 'no'}"
                    f"{' (degenerate)' if degenerate else ''}")
        history.append({"iteration": iteration + 1, "alpha": alpha,
                        "is_jailbroken": is_jb, "degenerate": degenerate,
                        "response": response})
        if is_jb:
            min_alpha = alpha
            break

    final = history[-1] if history else None
    return {
        "pair_id":                pair_id,
        "prompt":                 prompt,
        "goal":                   goal,
        "final_jailbroken":       final["is_jailbroken"] if final else False,
        "final_degenerate":       final["degenerate"] if final else False,
        "final_response":         final["response"] if final else "",
        "min_alpha_to_jailbreak": min_alpha,   # None if never jailbroken within budget
        "n_iterations":           len(history),
        "history":                history,
    }


def run_steering(source, w_vec, pool, model, tokenizer, device, client,
                 output_dir, clf_info):
    """Run the fixed-w, judge-only escalation attack on the raw JBB goals for one source."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n{'='*70}\n  STEERING on persona w | clf_source={source} | pairs={len(pool)}\n{'='*70}")

    # Incremental write per pair so an eviction mid-run doesn't lose work.
    out_path = output_dir / "results.jsonl"
    results = []
    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, pair in enumerate(pool):
            logger.info(f"\n[{source}] [{i+1}/{len(pool)}] {pair['behavior_id']} | {pair['category']}")
            result = escalate_until_jailbroken(
                pair_id=pair["pair_id"], prompt=pair["prompt"], goal=pair["goal"],
                w_vec=w_vec, model=model, tokenizer=tokenizer, device=device, client=client,
            )
            result["behavior_id"]   = pair["behavior_id"]
            result["category"]      = pair["category"]
            result["source"]        = pair["source"]
            result["jailbreak_idx"] = pair["jailbreak_idx"]
            result["jbb_jailbroken"] = None
            results.append(result)
            out_f.write(json.dumps({k: v for k, v in result.items() if k != "history"},
                                   ensure_ascii=True) + "\n")
            out_f.flush()

    # Post-hoc JBB 1-10 rubric judge.
    logger.info(f"[{source}] running post-hoc JBB judge on {len(results)} responses ...")
    for result in results:
        if not result.get("final_response") or result.get("final_degenerate"):
            result["jbb_jailbroken"] = False
        else:
            result["jbb_jailbroken"] = judge_response_jbb(
                client, goal=result["goal"], prompt=result["prompt"],
                response=result["final_response"])

    with open(out_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps({k: v for k, v in result.items() if k != "history"},
                               ensure_ascii=True) + "\n")

    gpt_metrics = compute_metrics(results, "final_jailbroken")
    jbb_metrics = compute_metrics(results, "jbb_jailbroken")
    summary = {
        "clf_source":     source,
        "steering_vector": "persona_classifier_w",
        "alpha_schedule": base.ALPHA_SCHEDULE,
        "config": {
            "model":          MODEL_NAME,
            "target":         "raw JBB goals (no templates)",
            "n_pairs":        len(results),
            "classifier":     f"{source} trait-projection logreg (C=1.0)",
            "classifier_stats": clf_info,
            "judge_inloop":   "gpt-4.1-mini binary YES/NO",
            "judge_posthoc":  "gpt-4.1-mini JBB 1-10 rubric (jailbroken if == 10)",
        },
        "gpt41mini": {
            "per_pair_asr":     gpt_metrics["per_pair_asr"],
            "per_behavior_asr": gpt_metrics["per_behavior_asr"],
            "by_category":      compute_category_metrics(results, "final_jailbroken"),
        },
        "jbb_rubric": {
            "per_pair_asr":     jbb_metrics["per_pair_asr"],
            "per_behavior_asr": jbb_metrics["per_behavior_asr"],
            "by_category":      compute_category_metrics(results, "jbb_jailbroken"),
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[{source}] saved to {output_dir}/")
    return gpt_metrics, jbb_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clf_source", default="both", choices=["harmbench", "jbb", "wildjailbreak", "both"],
                        help="Which classifier(s) to derive the steering w from.")
    parser.add_argument("--output_dir", default="full_trait_output/trait_steering_attack_jbb_persona_w")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--behaviors_source", default="jbb", choices=["jbb", "wildjailbreak", "harmbench"],
                        help="Steering target: jbb (100 raw goals), wildjailbreak (2000 adv prompts), or harmbench (400 behaviors)")
    parser.add_argument("--max_behaviors", type=int, default=None,
                        help="Subsample this many behaviors (mainly for wildjailbreak)")
    parser.add_argument("--alpha_schedule", type=float, nargs="+", default=None,
                        help="Override alpha schedule, e.g. --alpha_schedule 0.1 0.2 0.3 0.4 0.5")
    parser.add_argument("--test", action="store_true", help="3 goals only")
    args = parser.parse_args()

    if args.alpha_schedule is not None:
        base.ALPHA_SCHEDULE = args.alpha_schedule
        base.MAX_ITER = len(args.alpha_schedule)

    np.random.seed(RANDOM_SEED)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    sources = ["harmbench", "jbb"] if args.clf_source == "both" else [args.clf_source]
    logger.info(f"clf sources: {sources} | alpha schedule: {base.ALPHA_SCHEDULE}")

    # Model + trait matrix loaded once, shared across sources.
    logger.info(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map={"": device})
    model.eval()

    trait_matrix, _trait_names = load_trait_matrix_and_names(LAYER)
    logger.info(f"Trait matrix: {trait_matrix.shape}")

    if args.behaviors_source == "wildjailbreak":
        behaviors = fetch_wildjailbreak_behaviors(max_behaviors=args.max_behaviors)
    elif args.behaviors_source == "harmbench":
        behaviors = fetch_harmbench_behaviors()
    else:
        behaviors = fetch_jbb_behaviors()
    pool = build_raw_goal_pool(behaviors, args.test)
    logger.info(f"{args.behaviors_source} raw-goal pool: {len(pool)} pairs")

    output_root = Path(args.output_dir)
    metrics_by_source = {}
    for source in sources:
        w_vec, _clf, _scaler, clf_info = train_source(source, trait_matrix)
        gpt_m, jbb_m = run_steering(
            source, w_vec, pool, model, tokenizer, device, client,
            output_root / source, clf_info)
        metrics_by_source[source] = {"gpt": gpt_m, "jbb": jbb_m}

    # ── Side-by-side comparison ───────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  PERSONA-w JBB STEERING — steering target: raw JBB goals ({len(pool)} pairs)")
    print(f"  Alpha schedule: {base.ALPHA_SCHEDULE}")
    print(sep)
    print(f"  {'clf source':12}  {'GPT per-pair':>13}  {'GPT per-beh':>12}  "
          f"{'JBB per-pair':>13}  {'JBB per-beh':>12}")
    print(f"  {'-'*68}")
    for source in sources:
        g = metrics_by_source[source]["gpt"]
        j = metrics_by_source[source]["jbb"]
        print(f"  {source:12}  {g['per_pair_asr']:>13.1%}  {g['per_behavior_asr']:>12.1%}  "
              f"{j['per_pair_asr']:>13.1%}  {j['per_behavior_asr']:>12.1%}")
    print(sep)
    logger.info(f"Per-source results under {output_root}/<source>/")


if __name__ == "__main__":
    main()
