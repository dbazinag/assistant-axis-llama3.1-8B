#!/usr/bin/env python3
"""
collect_rtv_activations.py

Forward-only activation collector for an RTV-style Mahalanobis baseline.

The RTV paper uses refusal-direction fingerprints from:
  - layers 18, 25, 32 on Llama-3-8B
  - the last five prompt-token positions

For HuggingFace Llama models with 32 transformer blocks, paper layer 32 maps to
model layer index 31, because hidden_states[32] is the output after block 31.

This script reconstructs the exact prompts used by the existing attack-family
runs, checks pair_id/behavior_id/attack_type alignment against saved rows, and
saves only prompt-side hidden states. It does not generate, judge, or rescore.
"""

import argparse
import csv
import gc
import json
import logging
import os
import random
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME_OLMO3 = "allenai/OLMo-3-7B-Instruct"
HARMBENCH_ROOT = "/dlabscratch1/bazina/HarmBench"
OUTPUT_ROOT = "full_trait_output/rtv_activations"
OUTPUT_ROOT_OLMO3 = "full_trait_output/rtv_activations_olmo3"

PAPER_LAYERS = [18, 25, 32]
MODEL_LAYERS = [18, 25, 31]
N_LAST_TOKENS = 5
RANDOM_SEED = 42
N_ALPACA_HARMLESS = 240

SKIP_JAILBREAK_INDICES = {16, 44}
JAILBREAKS_LOCAL = f"{HARMBENCH_ROOT}/baselines/human_jailbreaks/jailbreaks.py"
JAILBREAKS_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/baselines/human_jailbreaks/jailbreaks.py"
)
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

DATASET_CONFIGS = {
    "harmbench": {
        "responses": "full_trait_output/harmbench_activations/classified_responses.jsonl",
    },
    "gcg": {
        "responses": "full_trait_output/gcg_activations/responses.jsonl",
        "test_cases": f"{HARMBENCH_ROOT}/results/GCG/llama3_1_8b/test_cases/test_cases.json",
    },
    "pair": {
        "responses": "full_trait_output/pair_activations/responses.jsonl",
        "test_cases": f"{HARMBENCH_ROOT}/results/PAIR/llama3_1_8b/test_cases/test_cases.json",
    },
    "pap": {
        "responses": "full_trait_output/pap_activations/responses.jsonl",
        "test_cases": f"{HARMBENCH_ROOT}/results/PAP/llama3_1_8b/test_cases/test_cases.json",
    },
    "gptfuzz": {
        "responses": "full_trait_output/gptfuzz_activations/responses.jsonl",
        "test_cases": f"{HARMBENCH_ROOT}/results/GPTFuzz/llama3_1_8b/test_cases/test_cases.json",
    },
    "pez": {
        "responses": "full_trait_output/pez_activations/responses.jsonl",
        "test_cases": f"{HARMBENCH_ROOT}/results/PEZ/llama3_1_8b/test_cases/test_cases.json",
    },
}

# OLMo-3 uses the labelled responses produced by the OLMo attack collector. For the
# attack families the prompts are rebuilt from the same per-behavior test-case files
# that collector consumed, so pair ordering matches the saved responses exactly.
DATASET_CONFIGS_OLMO3 = {
    "harmbench": {
        "responses": "full_trait_output/harmbench_activations_olmo3/classified_responses.jsonl",
    },
    "pair":    {"responses": "full_trait_output/pair_activations_olmo3_hb/classified_responses.jsonl",    "exp": "olmo3_7b_harmbench"},
    "pap":     {"responses": "full_trait_output/pap_activations_olmo3_hb/classified_responses.jsonl",     "exp": "olmo3_7b_top5"},
    "gptfuzz": {"responses": "full_trait_output/gptfuzz_activations_olmo3_hb/classified_responses.jsonl", "exp": "olmo3_7b"},
    "pez":     {"responses": "full_trait_output/pez_activations_olmo3_hb/classified_responses.jsonl",     "exp": "olmo3_7b"},
}


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_behaviors(path: Path, standard_only: bool | None = None) -> Dict[str, dict]:
    behaviors = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            is_standard = row.get("FunctionalCategory", "").lower() == "standard"
            if standard_only is None or is_standard == standard_only:
                behaviors[row["BehaviorID"]] = row
    return behaviors


def load_jailbreak_templates() -> List[Tuple[int, str]]:
    if Path(JAILBREAKS_LOCAL).exists():
        source = Path(JAILBREAKS_LOCAL).read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(JAILBREAKS_URL) as resp:
            source = resp.read().decode("utf-8")
    ns = {}
    exec(source, ns)  # noqa: S102 - HarmBench file is a list assignment.
    templates = []
    for idx, template in enumerate(ns["JAILBREAKS"]):
        if idx in SKIP_JAILBREAK_INDICES or "{0}" not in template:
            continue
        templates.append((idx, template))
    return templates


def assert_alignment(saved_rows: List[dict], rebuilt_rows: List[dict], name: str) -> None:
    if len(saved_rows) != len(rebuilt_rows):
        raise ValueError(f"{name}: saved rows={len(saved_rows)} rebuilt rows={len(rebuilt_rows)}")
    for i, (saved, rebuilt) in enumerate(zip(saved_rows, rebuilt_rows)):
        for key in ["pair_id", "behavior_id", "attack_type"]:
            if saved.get(key) != rebuilt.get(key):
                raise ValueError(
                    f"{name}: alignment mismatch at row {i} key={key}: "
                    f"saved={saved.get(key)!r} rebuilt={rebuilt.get(key)!r}"
                )


def build_harmbench_rows(responses_path: Path) -> List[dict]:
    saved_rows = load_jsonl(responses_path)
    templates = load_jailbreak_templates()
    rng = random.Random(RANDOM_SEED)
    sampled = rng.sample(templates, min(20, len(templates)))

    behavior_rows = list(load_behaviors(
        Path(HARMBENCH_ROOT) / "data/behavior_datasets/harmbench_behaviors_text_test.csv",
        standard_only=True,
    ).values())

    rebuilt = []
    pair_id = 0
    for behavior in behavior_rows:
        behavior_text = behavior["Behavior"]
        rebuilt.append({
            "pair_id": pair_id,
            "behavior_id": behavior["BehaviorID"],
            "behavior_text": behavior_text,
            "semantic_category": behavior.get("SemanticCategory", ""),
            "attack_type": "direct_request",
            "jailbreak_idx": -1,
            "jailbroken": saved_rows[pair_id].get("jailbroken"),
            "prompt_text": behavior_text,
        })
        pair_id += 1

        for jb_idx, template in sampled:
            rebuilt.append({
                "pair_id": pair_id,
                "behavior_id": behavior["BehaviorID"],
                "behavior_text": behavior_text,
                "semantic_category": behavior.get("SemanticCategory", ""),
                "attack_type": "human_jailbreak",
                "jailbreak_idx": jb_idx,
                "jailbroken": saved_rows[pair_id].get("jailbroken"),
                "prompt_text": template.format(behavior_text),
            })
            pair_id += 1

    assert_alignment(saved_rows, rebuilt, "harmbench")
    return rebuilt


def flatten_prompts(prompts: list) -> List[str]:
    flat = []
    for prompt in prompts:
        if isinstance(prompt, list):
            flat.extend(p for p in prompt if isinstance(p, str))
        elif isinstance(prompt, str):
            flat.append(prompt)
    return [p for p in flat if p.strip()]


def build_attack_rows(name: str, responses_path: Path, test_cases_path: Path) -> List[dict]:
    saved_rows = load_jsonl(responses_path)
    test_cases = json.load(open(test_cases_path, encoding="utf-8"))
    behaviors_all = load_behaviors(
        Path(HARMBENCH_ROOT) / "data/behavior_datasets/harmbench_behaviors_text_all.csv",
        standard_only=None,
    )

    rebuilt = []
    pair_id = 0
    if name == "pair":
        for behavior_id, prompts in test_cases.items():
            if behavior_id not in behaviors_all:
                continue
            behavior = behaviors_all[behavior_id]
            behavior_text = behavior["Behavior"]
            jailbreak_prompt = prompts[0]
            for attack_type, prompt_text in [("PAIR", jailbreak_prompt), ("plain", behavior_text)]:
                rebuilt.append({
                    "pair_id": pair_id,
                    "behavior_id": behavior_id,
                    "behavior_text": behavior_text,
                    "semantic_category": behavior.get("SemanticCategory", ""),
                    "functional_category": behavior.get("FunctionalCategory", ""),
                    "attack_type": attack_type,
                    "jailbreak_idx": -1,
                    "jailbroken": saved_rows[pair_id].get("jailbroken"),
                    "prompt_text": prompt_text,
                })
                pair_id += 1
    else:
        attack_type = name.upper() if name != "gptfuzz" else "GPTFuzz"
        if name == "gcg":
            attack_type = "GCG"
        for behavior_id, prompts in test_cases.items():
            if behavior_id not in behaviors_all:
                continue
            behavior = behaviors_all[behavior_id]
            for prompt_text in flatten_prompts(prompts):
                rebuilt.append({
                    "pair_id": pair_id,
                    "behavior_id": behavior_id,
                    "behavior_text": behavior["Behavior"],
                    "semantic_category": behavior.get("SemanticCategory", ""),
                    "functional_category": behavior.get("FunctionalCategory", ""),
                    "attack_type": attack_type,
                    "jailbreak_idx": -1,
                    "jailbroken": saved_rows[pair_id].get("jailbroken"),
                    "prompt_text": prompt_text,
                })
                pair_id += 1

    assert_alignment(saved_rows, rebuilt, name)
    return rebuilt


def load_individual_test_cases_olmo3(attack_upper: str, exp: str) -> Dict[str, list]:
    """Merge per-behavior test_cases.json files exactly as the OLMo attack collector did."""
    ind_dir = (Path(HARMBENCH_ROOT) / "results" / f"HarmBench_{attack_upper}" / exp
               / "test_cases" / "test_cases_individual_behaviors")
    if not ind_dir.exists():
        raise FileNotFoundError(f"Not found: {ind_dir}")
    merged: Dict[str, list] = {}
    for bdir in sorted(ind_dir.iterdir()):
        if not bdir.is_dir():
            continue
        tc_file = bdir / "test_cases.json"
        if not tc_file.exists():
            continue
        with tc_file.open() as f:
            d = json.load(f)
        for bid, prompts in d.items():
            flat: List[str] = []
            for p in prompts:
                if isinstance(p, list):
                    flat.extend(p)
                elif isinstance(p, str):
                    flat.append(p)
            if flat:
                merged[bid] = merged.get(bid, []) + flat
    return merged


def build_attack_rows_olmo3(name: str, cfg: dict) -> List[dict]:
    saved_rows = load_jsonl(Path(cfg["responses"]))
    attack_upper = {"pair": "PAIR", "pap": "PAP", "gptfuzz": "GPTFuzz", "pez": "PEZ"}[name]
    test_cases = load_individual_test_cases_olmo3(attack_upper, cfg["exp"])
    behaviors_all = load_behaviors(
        Path(HARMBENCH_ROOT) / "data/behavior_datasets/harmbench_behaviors_text_all.csv",
        standard_only=None,
    )

    rebuilt = []
    pair_id = 0
    for behavior_id, prompts in test_cases.items():
        behavior = behaviors_all.get(behavior_id)
        if behavior is None:
            continue
        for prompt in prompts:
            if not prompt or not isinstance(prompt, str) or not prompt.strip():
                continue
            rebuilt.append({
                "pair_id": pair_id,
                "behavior_id": behavior_id,
                "behavior_text": behavior["Behavior"],
                "semantic_category": behavior.get("SemanticCategory", ""),
                "functional_category": behavior.get("FunctionalCategory", ""),
                "attack_type": name,
                "jailbreak_idx": -1,
                "jailbroken": saved_rows[pair_id].get("jailbroken"),
                "prompt_text": prompt.strip(),
            })
            pair_id += 1

    assert_alignment(saved_rows, rebuilt, name)
    return rebuilt


def format_alpaca_prompt(record: dict) -> str:
    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", "")).strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def build_alpaca_harmless_rows(n_rows: int, seed: int) -> List[dict]:
    with urllib.request.urlopen(ALPACA_URL) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    prompts = [format_alpaca_prompt(row) for row in data]
    prompts = [p for p in prompts if p]
    rng = random.Random(seed)
    selected = rng.sample(prompts, min(n_rows, len(prompts)))
    return [
        {
            "pair_id": i,
            "behavior_id": f"alpaca_{i:04d}",
            "behavior_text": "",
            "semantic_category": "harmless",
            "functional_category": "harmless",
            "attack_type": "alpaca_harmless",
            "jailbreak_idx": -1,
            "jailbroken": False,
            "prompt_text": prompt,
        }
        for i, prompt in enumerate(selected)
    ]


def build_rows_for_dataset(name: str, args) -> List[dict]:
    if name == "alpaca_harmless":
        return build_alpaca_harmless_rows(args.n_alpaca_harmless, args.seed)
    if args.olmo3:
        cfg = DATASET_CONFIGS_OLMO3[name]
        if name == "harmbench":
            return build_harmbench_rows(Path(cfg["responses"]))
        return build_attack_rows_olmo3(name, cfg)
    cfg = DATASET_CONFIGS[name]
    responses_path = Path(cfg["responses"])
    if name == "harmbench":
        return build_harmbench_rows(responses_path)
    return build_attack_rows(name, responses_path, Path(cfg["test_cases"]))


def sanitize_text(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def load_model_and_tokenizer(model_name: str, device: torch.device):
    is_olmo = "olmo" in model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=is_olmo, trust_remote_code=is_olmo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map={"": device},
        trust_remote_code=is_olmo,
    )
    model.eval()
    return model, tokenizer


def prompt_input_ids(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": sanitize_text(prompt)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def process_row(model, tokenizer, row: dict, device: torch.device):
    input_ids = prompt_input_ids(tokenizer, row["prompt_text"], device)
    attention_mask = torch.ones_like(input_ids)
    if input_ids.shape[1] < N_LAST_TOKENS:
        raise ValueError(f"prompt has only {input_ids.shape[1]} tokens")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    layer_acts = {}
    for paper_layer, model_layer in zip(PAPER_LAYERS, MODEL_LAYERS):
        hidden = outputs.hidden_states[model_layer + 1][0, -N_LAST_TOKENS:, :].detach().cpu()
        layer_acts[str(paper_layer)] = hidden
    return layer_acts


def worker_fn(worker_id: int, gpu_id: int, rows: List[dict], model_name: str, output_dir: Path, dataset: str):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    activations = {}
    ok_rows = []
    n_errors = 0

    for row in tqdm(rows, desc=f"{dataset}/worker-{worker_id}", position=worker_id):
        try:
            activations[row["pair_id"]] = process_row(model, tokenizer, row, device)
            ok_rows.append({k: v for k, v in row.items() if k != "prompt_text"})
        except Exception as exc:
            n_errors += 1
            logger.warning(f"{dataset} row {row.get('pair_id')} failed: {exc}")
        if len(activations) % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    torch.save(activations, output_dir / f"{dataset}_worker_{worker_id}_activations.pt")
    write_jsonl(output_dir / f"{dataset}_worker_{worker_id}_rows.jsonl", ok_rows)
    logger.info(f"{dataset} worker {worker_id}: ok={len(ok_rows)} errors={n_errors}")
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def merge_dataset_outputs(output_dir: Path, dataset: str, n_workers: int):
    all_rows = []
    all_acts = {}
    for wid in range(n_workers):
        row_path = output_dir / f"{dataset}_worker_{wid}_rows.jsonl"
        act_path = output_dir / f"{dataset}_worker_{wid}_activations.pt"
        if row_path.exists():
            all_rows.extend(load_jsonl(row_path))
            row_path.unlink()
        if act_path.exists():
            all_acts.update(torch.load(act_path, map_location="cpu", weights_only=False))
            act_path.unlink()
    all_rows.sort(key=lambda r: r["pair_id"])
    write_jsonl(output_dir / f"{dataset}_rows.jsonl", all_rows)
    torch.save(all_acts, output_dir / f"{dataset}_activations.pt")
    logger.info(f"{dataset}: merged rows={len(all_rows)} activations={len(all_acts)}")


def collect_dataset(dataset: str, rows: List[dict], args, output_dir: Path):
    if args.test:
        rows = rows[: min(len(rows), args.test_rows)]
    logger.info(f"{dataset}: collecting {len(rows)} rows")

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_ids = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    if not gpu_ids:
        gpu_ids = [-1]
    if args.test:
        gpu_ids = gpu_ids[:1]

    chunks = [[] for _ in gpu_ids]
    for i, row in enumerate(rows):
        chunks[i % len(gpu_ids)].append(row)

    if len(gpu_ids) == 1:
        worker_fn(0, gpu_ids[0], chunks[0], args.model, output_dir, dataset)
    else:
        mp.set_start_method("spawn", force=True)
        procs = []
        for wid, gpu_id in enumerate(gpu_ids):
            proc = mp.Process(
                target=worker_fn,
                args=(wid, gpu_id, chunks[wid], args.model, output_dir, dataset),
            )
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"{dataset}: worker exited with code {proc.exitcode}")

    merge_dataset_outputs(output_dir, dataset, len(gpu_ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="alpaca_harmless,harmbench,gcg,pair,pap,gptfuzz,pez")
    parser.add_argument("--output_dir", default=OUTPUT_ROOT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--n_alpaca_harmless", type=int, default=N_ALPACA_HARMLESS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_rows", type=int, default=8)
    parser.add_argument("--olmo3", action="store_true",
                        help="Use OLMo-3-7B-Instruct + OLMo response paths (GCG excluded; not yet collected)")
    args = parser.parse_args()

    if args.olmo3:
        if args.model == MODEL_NAME:
            args.model = MODEL_NAME_OLMO3
        if args.output_dir == OUTPUT_ROOT:
            args.output_dir = OUTPUT_ROOT_OLMO3
        if args.datasets == parser.get_default("datasets"):
            args.datasets = "alpaca_harmless,harmbench,pair,pap,gptfuzz,pez"

    output_dir = Path(args.output_dir)
    if args.test:
        output_dir = output_dir.parent / f"{output_dir.name}_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    valid = {"alpaca_harmless", *DATASET_CONFIGS.keys()}
    unknown = [d for d in datasets if d not in valid]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "paper_layers": PAPER_LAYERS,
        "model_layers": MODEL_LAYERS,
        "n_last_tokens": N_LAST_TOKENS,
        "datasets": datasets,
        "test_mode": args.test,
        "notes": "RTV-style forward-only cache; paper layer 32 maps to HF model layer 31.",
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    for dataset in datasets:
        rows = build_rows_for_dataset(dataset, args)
        collect_dataset(dataset, rows, args, output_dir)

    logger.info(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
