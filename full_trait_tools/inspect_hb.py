#!/usr/bin/env python3
"""
inspect_harmbench.py

Quick inspection script to pull HarmBench HumanJailbreaks + behavior questions
and eyeball their quality before committing to a full run.

No HarmBench clone needed — fetches two files directly from GitHub.
"""

import csv
import io
import random
import re
import textwrap
import urllib.request

# ── URLs ──────────────────────────────────────────────────────────────────────
JAILBREAKS_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/baselines/human_jailbreaks/jailbreaks.py"
)
BEHAVIORS_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/data/behavior_datasets/harmbench_behaviors_text_test.csv"
)

# ── Config ────────────────────────────────────────────────────────────────────
N_JAILBREAK_SAMPLES = 3   # how many jailbreak templates to print in full
N_PAIR_SAMPLES      = 5   # how many (jailbreak, behavior) pairs to print
RANDOM_SEED         = 42
MAX_TEMPLATE_CHARS  = 600  # truncate long templates in the summary table


def fetch_text(url: str) -> str:
    print(f"  Fetching: {url}")
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def parse_jailbreaks(source: str) -> list[str]:
    """
    Execute the jailbreaks.py source in a sandbox namespace and extract
    the JAILBREAKS list.  The file is just a single list assignment so
    this is safe.
    """
    ns: dict = {}
    exec(source, ns)  # noqa: S102
    jailbreaks = ns.get("JAILBREAKS", [])
    assert isinstance(jailbreaks, list), "Expected JAILBREAKS to be a list"
    return jailbreaks


def load_behaviors(csv_text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(csv_text))
    return list(reader)


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def wrap(text: str, width: int = 90, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    random.seed(RANDOM_SEED)

    # 1. Fetch data
    print_section("Fetching data from GitHub")
    jb_source    = fetch_text(JAILBREAKS_URL)
    behavior_csv = fetch_text(BEHAVIORS_URL)

    # 2. Parse
    jailbreaks = parse_jailbreaks(jb_source)
    behaviors  = load_behaviors(behavior_csv)

    # 3. Stats
    print_section("Dataset statistics")
    print(f"  Jailbreak templates : {len(jailbreaks)}")
    print(f"  Behavior rows (all) : {len(behaviors)}")

    # Show column names
    if behaviors:
        print(f"  Behavior CSV columns: {list(behaviors[0].keys())}")

    # Break down by FunctionalCategory
    from collections import Counter
    func_counts = Counter(b.get("FunctionalCategory", "?") for b in behaviors)
    sem_counts  = Counter(b.get("SemanticCategory",  "?") for b in behaviors)
    print("\n  FunctionalCategory breakdown:")
    for k, v in sorted(func_counts.items()):
        print(f"    {k:30s}  {v}")
    print("\n  SemanticCategory breakdown:")
    for k, v in sorted(sem_counts.items()):
        print(f"    {k:40s}  {v}")

    # Filter to standard only (no context string needed)
    standard_behaviors = [b for b in behaviors
                          if b.get("FunctionalCategory", "").lower() == "standard"]
    print(f"\n  Standard behaviors (no context): {len(standard_behaviors)}")

    # 4. Jailbreak template previews
    print_section(f"Jailbreak template previews ({N_JAILBREAK_SAMPLES} random samples)")
    sample_jbs = random.sample(jailbreaks, min(N_JAILBREAK_SAMPLES, len(jailbreaks)))
    for i, jb in enumerate(sample_jbs, 1):
        print(f"\n  ── Template {i} ──")
        preview = jb[:MAX_TEMPLATE_CHARS]
        if len(jb) > MAX_TEMPLATE_CHARS:
            preview += f"  ... [{len(jb) - MAX_TEMPLATE_CHARS} more chars]"
        print(wrap(preview))
        # Check placeholder
        n_placeholders = jb.count("{0}")
        print(f"    [total length: {len(jb)} chars | {{0}} placeholders: {n_placeholders}]")

    # 5. Formatted (jailbreak + behavior) pair samples
    print_section(f"Formatted prompt samples ({N_PAIR_SAMPLES} random pairs)")
    print("  (These are the actual strings that would be sent to the model)\n")

    sample_jbs2 = random.choices(jailbreaks, k=N_PAIR_SAMPLES)
    sample_bvs  = random.sample(standard_behaviors,
                                min(N_PAIR_SAMPLES, len(standard_behaviors)))

    for i, (jb, bv) in enumerate(zip(sample_jbs2, sample_bvs), 1):
        behavior_text = bv.get("Behavior", bv.get("behavior", ""))
        behavior_id   = bv.get("BehaviorID", "?")
        sem_cat       = bv.get("SemanticCategory", "?")

        try:
            formatted = jb.format(behavior_text)
        except (KeyError, IndexError) as e:
            formatted = f"[FORMAT ERROR: {e}]\n{jb}"

        print(f"  ── Pair {i} ──")
        print(f"  BehaviorID : {behavior_id}")
        print(f"  Category   : {sem_cat}")
        print(f"  Behavior   : {behavior_text}")
        print(f"  Jailbreak template length: {len(jb)} chars")
        print(f"  Formatted prompt (first 800 chars):")
        print(wrap(formatted[:800]))
        if len(formatted) > 800:
            print(f"    ... [{len(formatted) - 800} more chars]")
        print()

    # 6. Check for {0} placeholder in all templates
    print_section("Placeholder audit")
    missing = [i for i, jb in enumerate(jailbreaks) if "{0}" not in jb]
    if missing:
        print(f"  WARNING: {len(missing)} templates have no {{0}} placeholder!")
        print(f"  Indices: {missing}")
    else:
        print(f"  All {len(jailbreaks)} templates contain a {{0}} placeholder. Good.")

    # 7. Length distribution
    print_section("Jailbreak template length distribution")
    lengths = sorted(len(jb) for jb in jailbreaks)
    print(f"  Min    : {lengths[0]:,} chars")
    print(f"  Median : {lengths[len(lengths)//2]:,} chars")
    print(f"  Max    : {lengths[-1]:,} chars")
    buckets = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 999999)]
    for lo, hi in buckets:
        count = sum(1 for l in lengths if lo <= l < hi)
        label = f"{lo}-{hi}" if hi < 999999 else f"{lo}+"
        print(f"  {label:12s}: {count:3d} templates")

    print("\nDone. Inspect the output above to assess prompt quality.\n")


if __name__ == "__main__":
    main()