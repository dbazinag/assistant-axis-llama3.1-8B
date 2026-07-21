#!/usr/bin/env python3
"""
collect_jbb_attack_activations.py

Run a target model on JBB attack prompts (from comparison_attacks_jbb/) and
collect hidden states at layers 16 & 28 at the last prompt token.

Supports: PAIR, PAP, GPTFuzz, PEZ (model-agnostic attacks available in JBB).
GCG is skipped — gradient-optimized for Llama, no prompt text available here.

Usage:
  uv run full_trait_tools/collect_jbb_attack_activations.py \\
    --model allenai/OLMo-3-7B-Instruct \\
    --trust_remote_code \\
    --output_dir full_trait_output/jbb_attack_activations_olmo3

  # Test mode (10 prompts per family, single GPU)
  uv run full_trait_tools/collect_jbb_attack_activations.py --test
"""

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LAYERS_TO_SAVE = [16, 28]
MAX_NEW_TOKENS = 300
RANDOM_SEED    = 42

# Attack families available in comparison_attacks_jbb (model-agnostic)
JBB_FAMILIES = ["PAIR", "PAP", "GPTFuzz", "PEZ"]


def load_jbb_pairs(jbb_base: Path, test_mode: bool = False) -> List[dict]:
    """Load attack prompts from comparison_attacks_jbb/*/results.jsonl."""
    pairs: List[dict] = []
    pair_id = 0
    for family in JBB_FAMILIES:
        p = jbb_base / family / "results.jsonl"
        if not p.exists():
            logger.warning(f"Missing: {p}")
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if test_mode:
            rows = rows[:10]
        for row in rows:
            raw_prompt = row.get("prompt") or ""
            # GPTFuzz stores some prompts as a list; take the last element
            if isinstance(raw_prompt, list):
                raw_prompt = raw_prompt[-1] if raw_prompt else ""
            prompt = raw_prompt.strip()
            if not prompt:
                continue
            pairs.append({
                "pair_id":           pair_id,
                "behavior_id":       row.get("behavior_id", f"{family}_{pair_id:05d}"),
                "behavior_text":     row.get("goal", ""),
                "semantic_category": row.get("category", family),
                "attack_type":       family.lower(),
                "jailbreak_idx":     pair_id,
                "formatted_prompt":  prompt,
            })
            pair_id += 1
        logger.info(f"  {family}: {len(rows)} prompts")
    logger.info(f"Total pairs: {len(pairs)}")
    return pairs


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    trust_remote_code: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def sanitize_text(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def process_pair(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pair: dict,
    layers: List[int],
    max_new_tokens: int,
    device: torch.device,
) -> Optional[dict]:
    try:
        text = sanitize_text(pair["formatted_prompt"])
        conversation = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(
            formatted, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        with torch.no_grad():
            fwd_out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

        layer_acts: Dict[int, torch.Tensor] = {}
        for layer_idx in layers:
            hs = fwd_out.hidden_states[layer_idx + 1]
            layer_acts[layer_idx] = hs[0, -1, :].detach().cpu()

        del fwd_out
        torch.cuda.empty_cache()

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_ids  = gen_ids[0, input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        del gen_ids
        torch.cuda.empty_cache()

        return {"response_text": response_text, "layer_acts": layer_acts}

    except Exception as e:
        logger.warning(f"Error on pair {pair['pair_id']}: {e}")
        return None


def worker_fn(
    worker_id: int,
    gpu_id: int,
    pairs: List[dict],
    model_name: str,
    trust_remote_code: bool,
    max_new_tokens: int,
    output_dir: Path,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - Worker-{worker_id}[GPU:{gpu_id}] - %(levelname)s - %(message)s",
        force=True,
    )
    wlogger = logging.getLogger(f"worker_{worker_id}")
    wlogger.info(f"Starting — {len(pairs)} pairs")

    model, tokenizer = load_model_and_tokenizer(model_name, device, trust_remote_code)

    responses:   List[dict]      = []
    activations: Dict[int, dict] = {}
    n_errors = 0

    for pair in tqdm(pairs, desc=f"Worker-{worker_id}", position=worker_id):
        result = process_pair(model, tokenizer, pair, LAYERS_TO_SAVE, max_new_tokens, device)
        if result is None:
            n_errors += 1
            continue

        row = {k: v for k, v in pair.items() if k != "formatted_prompt"}
        row["response_text"] = result["response_text"]
        responses.append(row)

        activations[pair["pair_id"]] = {
            str(li): t for li, t in result["layer_acts"].items()
        }

        if len(responses) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    resp_path = output_dir / f"worker_{worker_id}_responses.jsonl"
    acts_path  = output_dir / f"worker_{worker_id}_activations.pt"

    with open(resp_path, "w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    torch.save(activations, acts_path)
    wlogger.info(f"Done — {len(responses)} OK, {n_errors} errors.")


def merge_worker_outputs(output_dir: Path, n_workers: int) -> None:
    all_responses:   List[dict]      = []
    all_activations: Dict[int, dict] = {}

    for wid in range(n_workers):
        rp = output_dir / f"worker_{wid}_responses.jsonl"
        ap = output_dir / f"worker_{wid}_activations.pt"

        if rp.exists():
            all_responses.extend(
                json.loads(l) for l in rp.read_text().splitlines() if l.strip()
            )
            rp.unlink()

        if ap.exists():
            all_activations.update(torch.load(ap, map_location="cpu", weights_only=False))
            ap.unlink()

    all_responses.sort(key=lambda x: x["pair_id"])

    with open(output_dir / "responses.jsonl", "w", encoding="utf-8") as f:
        for r in all_responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    torch.save(all_activations, output_dir / "activations.pt")
    logger.info(f"Merged {len(all_responses)} responses, {len(all_activations)} activation entries")

    # Save per-family splits so run_all_traits_sweep_v2.py can use them as
    # separate transfer test sets (each family gets its own responses + activations).
    from collections import defaultdict
    by_family: Dict[str, List[dict]] = defaultdict(list)
    for r in all_responses:
        by_family[r["attack_type"]].append(r)

    for family, rows in by_family.items():
        fdir = output_dir / family
        fdir.mkdir(exist_ok=True)
        pids = {r["pair_id"] for r in rows}
        with open(fdir / "responses.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        fam_acts = {pid: all_activations[pid] for pid in pids if pid in all_activations}
        torch.save(fam_acts, fdir / "activations.pt")
        logger.info(f"  {family}: {len(rows)} responses saved to {fdir}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          type=str, default="allenai/OLMo-3-7B-Instruct")
    parser.add_argument("--jbb_base",       type=str, default="full_trait_output/comparison_attacks_jbb")
    parser.add_argument("--output_dir",     type=str, default="full_trait_output/jbb_attack_activations_olmo3")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--test", action="store_true", help="10 prompts per family, single GPU")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.test:
        output_dir = output_dir.parent / (output_dir.name + "_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_jbb_pairs(Path(args.jbb_base), test_mode=args.test)
    if not pairs:
        logger.error("No pairs loaded. Check --jbb_base path.")
        sys.exit(1)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model":          args.model,
        "dataset":        "JBB comparison_attacks_jbb",
        "families":       JBB_FAMILIES,
        "n_pairs":        len(pairs),
        "layers_saved":   LAYERS_TO_SAVE,
        "test_mode":      args.test,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_ids = [int(x.strip()) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    if not gpu_ids:
        logger.error("No GPUs found.")
        sys.exit(1)

    if args.test:
        gpu_ids = gpu_ids[:1]

    n_workers = len(gpu_ids)
    chunks = [[] for _ in range(n_workers)]
    for i, pair in enumerate(pairs):
        chunks[i % n_workers].append(pair)

    kwargs = dict(
        model_name=args.model,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        output_dir=output_dir,
    )

    if n_workers == 1:
        worker_fn(0, gpu_ids[0], chunks[0], **kwargs)
    else:
        mp.set_start_method("spawn", force=True)
        procs = []
        for wid in range(n_workers):
            p = mp.Process(target=worker_fn, args=(wid, gpu_ids[wid], chunks[wid]), kwargs=kwargs)
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    if n_workers > 1:
        merge_worker_outputs(output_dir, n_workers)
    else:
        for stem in ["responses.jsonl", "activations.pt"]:
            w = output_dir / f"worker_0_{stem.split('.')[0]}.{stem.split('.')[1]}"
            if w.exists():
                w.rename(output_dir / stem)

    logger.info(f"Done. Outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
