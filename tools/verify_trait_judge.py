#!/usr/bin/env python3
# Verifies a random sample of trait-judge scores by re-judging them with the same trait eval_prompt.

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai not found. Run: uv add openai")


def collect_all_samples(responses_dir: Path) -> list[dict]:
    """
    Walk every .jsonl in responses_dir and collect all records,
    attaching the trait name (stem of the file).
    """
    samples: list[dict] = []
    for jsonl_path in sorted(responses_dir.glob("*.jsonl")):
        trait = jsonl_path.stem
        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rec["_trait"] = trait
                samples.append(rec)
            except json.JSONDecodeError:
                pass
    return samples


def load_trait_eval_prompt(traits_dir: Path, trait: str) -> Optional[str]:
    p = traits_dir / f"{trait}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        prompt = data.get("eval_prompt")
        return prompt if isinstance(prompt, str) and prompt.strip() else None
    except Exception:
        return None


def get_stored_score(scores_dir: Path, trait: str, label: str) -> Optional[int]:
    """
    Score file is a dict with keys equal to resp['label'], e.g.:
      positive_p0_q17
      negative_p3_q42
    """
    p = scores_dir / f"{trait}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        val = data.get(label)
        return int(val) if val is not None else None
    except Exception:
        return None


def extract_assistant_answer(rec: dict) -> str:
    conv = rec.get("conversation", [])
    for msg in conv:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    if conv and isinstance(conv[-1], dict):
        return conv[-1].get("content", "")
    return ""


def call_judge(
    client: OpenAI,
    model: str,
    eval_prompt_template: str,
    question: str,
    answer: str,
) -> tuple[int, str]:
    prompt = eval_prompt_template.format(question=question, answer=answer)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    raw = (resp.choices[0].message.content or "").strip()

    if raw.upper().startswith("REFUSAL"):
        return 0, raw

    m = re.search(r"\b(100|[0-9]{1,2})\b", raw)
    score = int(m.group(1)) if m else -1
    if not (0 <= score <= 100):
        score = -1
    return score, raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses_dir", required=True, type=Path)
    ap.add_argument("--scores_dir", required=True, type=Path)
    ap.add_argument("--traits_dir", required=True, type=Path)
    ap.add_argument("--n_samples", type=int, default=15, help="Total prompts to sample across all traits")
    ap.add_argument("--openai_model", default="gpt-4o-mini")
    ap.add_argument("--openai_api_key", default=None)
    ap.add_argument("--out_file", type=Path, default=Path("verify_judge_traits.txt"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY or pass --openai_api_key")

    client = OpenAI(api_key=api_key)
    random.seed(args.seed)

    all_samples = collect_all_samples(args.responses_dir)
    if not all_samples:
        sys.exit(f"No records found in {args.responses_dir}")

    samples = random.sample(all_samples, min(args.n_samples, len(all_samples)))
    print(
        f"Sampled {len(samples)} records from {len(all_samples)} total across "
        f"{len(set(s['_trait'] for s in samples))} traits.\n"
    )

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit(f"Judge verification  |  model: {args.openai_model}  |  n={len(samples)}")
    emit(f"Seed: {args.seed}")
    emit("=" * 88)

    matches = 0
    mismatches = 0
    no_stored = 0
    parse_errors = 0
    no_prompt = 0

    for i, rec in enumerate(samples, start=1):
        trait = rec["_trait"]
        question = rec.get("question", "")
        answer = extract_assistant_answer(rec)
        label = rec.get("label", f"missing_label_{i}")

        eval_prompt = load_trait_eval_prompt(args.traits_dir, trait)
        if eval_prompt is None:
            no_prompt += 1
            emit(f"\n[{i}/{len(samples)}]  trait={trait}  key={label}")
            emit("  ERROR  : missing eval_prompt for trait")
            continue

        new_score, raw = call_judge(
            client=client,
            model=args.openai_model,
            eval_prompt_template=eval_prompt,
            question=question,
            answer=answer,
        )
        stored = get_stored_score(args.scores_dir, trait, label)

        if new_score == -1:
            parse_errors += 1

        if stored is None:
            no_stored += 1
            verdict = "no stored score"
        elif stored == new_score:
            matches += 1
            verdict = "MATCH"
        else:
            mismatches += 1
            verdict = f"DIFFERS  (stored={stored})"

        emit(f"\n[{i}/{len(samples)}]  trait={trait}  key={label}")
        emit(f"  Q      : {question[:160]}")
        emit(f"  A      : {answer[:300]}{'...' if len(answer) > 300 else ''}")
        emit(f"  Score  : {new_score}  |  raw='{raw}'  |  {verdict}")

    compared = len(samples) - no_stored - no_prompt
    emit("\n" + "=" * 88)
    emit("SUMMARY")
    emit(f"  Sampled      : {len(samples)}")
    emit(f"  Missing eval : {no_prompt}")
    emit(f"  Parse errors : {parse_errors}")
    emit(f"  No stored    : {no_stored}")
    if compared:
        emit(f"  Matches      : {matches} / {compared}  ({100 * matches / compared:.0f}% match rate)")
    else:
        emit("  Matches      : (no stored scores to compare)")
    emit(f"  Mismatches   : {mismatches}")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text("\n".join(lines) + "\n")
    print(f"\nSaved → {args.out_file}")


if __name__ == "__main__":
    main()