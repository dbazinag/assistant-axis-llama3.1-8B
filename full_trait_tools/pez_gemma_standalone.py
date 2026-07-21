"""Standalone PEZ (hard-prompt gradient attack) for Gemma-4-31B in .venv_gemma.

HarmBench's PEZ harness can't run here: its env (transformers 4.45) won't load
gemma4, and pulling gemma4-capable transformers into that env collides with
ray/fastchat/sentence-transformers. PEZ's core is ~60 self-contained lines, so we
vendor it and run in the proven .venv_gemma (gemma4 loads, eager attn for clean
autograd at head_dim=512). The one sentence_transformers call (nearest-neighbour
projection) becomes a 2-line torch cosine-NN.

Faithful to baselines/pez/pez.py: optimize `num_optim_tokens` soft embeddings via
Adam + cosine schedule against the affirmative target (teacher-forced CE), with a
straight-through projection to real token embeddings each step. Top-K vocab loss
restriction kept (Gemma's vocab is ~256k). Writes HarmBench's per-behavior layout
so the existing collector (--attack_type PEZ --exp gemma4_31b) reads it directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pez_gemma")

TOP_K_VOCAB = 1000   # restrict CE to top-K logits + target tokens (256k vocab -> ~1k)


def nn_project(curr_embeds, embed_matrix_norm, embed_layer):
    """Nearest vocab embedding (cosine) for each soft embedding. [B,T,D]->proj,[B,T]."""
    B, T, D = curr_embeds.shape
    flat = F.normalize(curr_embeds.reshape(-1, D).float(), dim=-1)   # match embed_matrix_norm (float32)
    nn_idx = (flat @ embed_matrix_norm.T).argmax(dim=-1)
    proj = embed_layer(nn_idx).reshape(B, T, D)
    return proj, nn_idx.reshape(B, T)


class ProjectSoftEmbeds(torch.autograd.Function):
    """Straight-through: project to nearest hard embedding forward, identity grad."""
    embed_matrix_norm = None
    embed_layer = None

    @staticmethod
    def forward(ctx, inp):
        proj, _ = nn_project(inp, ProjectSoftEmbeds.embed_matrix_norm, ProjectSoftEmbeds.embed_layer)
        return proj

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def optimize_behavior(model, tokenizer, behavior, target, num_generate, num_optim_tokens,
                      num_steps, lr, device):
    embed_layer = model.get_input_embeddings()   # embed_matrix_norm set once in main()

    # template wrapper from Gemma's own chat template, split on {instruction}
    wrapped = tokenizer.apply_chat_template(
        [{"role": "user", "content": "{instruction}"}], tokenize=False, add_generation_prompt=True)
    before_tc, after_tc = wrapped.split("{instruction}")
    behavior = behavior + " "

    before_ids = tokenizer([before_tc], padding=False)["input_ids"]                       # keeps BOS
    rest = tokenizer([behavior, after_tc, target], padding=False, add_special_tokens=False)["input_ids"]
    ids = [torch.tensor(x, device=device).unsqueeze(0) for x in (before_ids[0], *rest)]
    before_ids, behavior_ids, after_ids, target_ids = ids
    before_e, behavior_e, after_e, target_e = [embed_layer(x) for x in ids]

    optim = torch.randint(embed_layer.weight.shape[0], (num_generate, num_optim_tokens), device=device)
    optim_embeds = torch.nn.Parameter(embed_layer(optim).float())
    optimizer = torch.optim.Adam([optim_embeds], lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

    loss_fct = CrossEntropyLoss(reduction="none")
    test_cases = None
    final_loss = None
    for _ in range(num_steps):
        projected = ProjectSoftEmbeds.apply(optim_embeds.to(model.dtype))
        input_embeds = torch.cat([
            before_e.repeat(num_generate, 1, 1), behavior_e.repeat(num_generate, 1, 1),
            projected, after_e.repeat(num_generate, 1, 1), target_e.repeat(num_generate, 1, 1)], dim=1)

        # Only compute logits for the last (target_len + 1) positions — computing them
        # over the full sequence materializes [B, seq, 256k] and OOMs on long contexts.
        tgt_len = target_e.shape[1]
        logits = model(inputs_embeds=input_embeds, use_cache=False,
                       logits_to_keep=tgt_len + 1).logits
        shift_logits = logits[..., :-1, :].contiguous()   # positions predicting the target tokens
        shift_labels = target_ids.repeat(num_generate, 1)

        top_vocab = shift_logits.view(-1, shift_logits.size(-1)).abs().mean(0).topk(TOP_K_VOCAB).indices
        top_vocab = torch.cat([top_vocab, shift_labels.view(-1).unique()]).unique()
        restricted = shift_logits[..., top_vocab]
        remap = torch.zeros(shift_logits.size(-1), dtype=torch.long, device=device)
        remap[top_vocab] = torch.arange(top_vocab.numel(), device=device)
        labels = remap[shift_labels.view(-1)]
        loss = loss_fct(restricted.view(-1, restricted.size(-1)), labels).view(num_generate, -1).mean(dim=1)

        optimizer.zero_grad()
        loss.mean().backward(inputs=[optim_embeds])
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            _, nn_idx = nn_project(optim_embeds.to(model.dtype),
                                   ProjectSoftEmbeds.embed_matrix_norm, embed_layer)
        full = torch.cat([behavior_ids.repeat(num_generate, 1), nn_idx], dim=1)
        test_cases = tokenizer.batch_decode(full, skip_special_tokens=True)
        final_loss = loss.detach().cpu().tolist()
    return test_cases, final_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--behaviors_path", required=True)
    ap.add_argument("--targets_path", required=True)
    ap.add_argument("--save_dir", required=True, help=".../results/HarmBench_PEZ/<exp>/test_cases")
    ap.add_argument("--num_generate", type=int, default=3)
    ap.add_argument("--micro_batch", type=int, default=0,
                    help="If >0, run num_generate in chunks of this size (rows are independent, "
                         "so this only caps peak attention memory). 0 = single batch (default).")
    ap.add_argument("--num_optim_tokens", type=int, default=20)
    ap.add_argument("--num_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--behavior_start_idx", type=int, default=None)
    ap.add_argument("--behavior_end_idx", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    with open(args.behaviors_path, encoding="utf-8") as f:
        behaviors = list(csv.DictReader(f))
    if args.behavior_start_idx is not None and args.behavior_end_idx is not None:
        behaviors = behaviors[args.behavior_start_idx:args.behavior_end_idx]
    targets = json.load(open(args.targets_path, encoding="utf-8"))

    out_root = Path(args.save_dir) / "test_cases_individual_behaviors"
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map="cuda:0", trust_remote_code=True)
    model.eval()
    model.requires_grad_(False)
    # Gradient checkpointing: recompute layer activations in backward instead of
    # storing all 60 layers — needed so long contextual behaviors (~2k-token
    # contexts) fit in 80GB. optim_embeds requires grad, so the graph still builds.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Precompute the normalized embedding matrix ONCE (float32, ~5.5GB) — recomputing
    # it per behavior leaked memory (transient doubling) and caused OOM mid-run.
    ProjectSoftEmbeds.embed_layer = model.get_input_embeddings()
    ProjectSoftEmbeds.embed_matrix_norm = F.normalize(
        ProjectSoftEmbeds.embed_layer.weight.float(), dim=-1)
    logger.info("Model loaded.")

    n = len(behaviors)
    for i, b in enumerate(behaviors):
        bid = b["BehaviorID"]
        out_file = out_root / bid / "test_cases.json"
        if out_file.exists():
            logger.info(f"[{i+1}/{n}] {bid} exists, skip")
            continue
        behavior = b["Behavior"]
        if b.get("ContextString"):
            behavior = f"{b['ContextString']}\n\n---\n\n{behavior}"
        target = targets[bid]
        mb = args.micro_batch if args.micro_batch > 0 else args.num_generate
        tcs, loss = [], []
        for g0 in range(0, args.num_generate, mb):
            ng = min(mb, args.num_generate - g0)
            t, l = optimize_behavior(
                model, tokenizer, behavior, target, ng,
                args.num_optim_tokens, args.num_steps, args.lr, device)
            tcs += t
            loss += l
            torch.cuda.empty_cache()   # release each micro-batch graph before the next
        out_file.parent.mkdir(parents=True, exist_ok=True)
        json.dump({bid: tcs}, open(out_file, "w"))
        logger.info(f"[{i+1}/{n}] {bid} done, final_loss={[round(x,3) for x in loss]}")
        torch.cuda.empty_cache()   # release the behavior's optimization graph before the next

    logger.info("All behaviors done.")


if __name__ == "__main__":
    main()
