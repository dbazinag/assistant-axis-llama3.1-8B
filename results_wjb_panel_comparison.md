# ASR Comparison — Llama-3.1-8B (LLAMA ONLY)

How each attack/induction approach changes the baseline ASR, across all three eval datasets. Lead metric: **per-pair ASR, GPT-4.1-mini judge**. "—" = never run on that dataset.

| Method | WJB (2000) | HarmBench (400) | JBB (100) |
|---|---:|---:|---:|
| **Baseline** (unsteered / direct request) | 51.2% | — | 1.0% |
| wjb-clf-guided | 57.0% | — | — |
| trait (top-k) | 72.2% | — | 16.0% |
| persona-w (HarmBench-clf) | 77.3% | 55.0% | 63.0% |
| persona-w (WJB-clf) | 65.6% | 42.5% | 41.0% |
| w_aligned | 79.2% | 39.5% | 31.0% |
| raw w | 83.6% | — | 68.0% |
| PEZ | 26.35% | — | 0.3% |
| PAP | 20.15% | — | 6.0% |
| PAIR | 22.2% | — | 9.0% |
| GCG | 37.9% | — | 31.0% |
| GPTFuzz | 54.35% | — | 77.0% |

## Notes

- **Steering rows** (wjb-clf-guided, trait, persona-w, w_aligned, raw w): direction/trait selection is driven by a fixed classifier (HarmBench-clf by default unless labeled WJB-clf). `top_k=3`, `alpha=[0.1,0.2,0.3,0.4,0.5]`, `LAYER=16`. On WJB/JBB these steer already-loaded goal pools directly (`--no_templates`); on HarmBench, `--behaviors_source harmbench` (400 behaviors: 200 standard / 100 copyright / 100 contextual — copyright reads as 0% because the GPT-4.1-mini harm judge isn't HarmBench's dedicated copyright classifier, so HarmBench-column numbers are a conservative lower bound).
- **Classic-attack rows** (PEZ, PAP, PAIR, GCG, GPTFuzz) transform the target prompt itself (suffix/rewrite/paraphrase/template-mutation) rather than steering activations. On WJB the starting prompt is already a working jailbreak (51.2% baseline), so these numbers are **per-pair** (diluted for multi-variant attacks — PAP ×5, PEZ ×3 — since per-pair divides by all variants, not just successful behaviors); per-behavior numbers (WJB: GPTFuzz 54.35%, PAP 50.25%, PEZ 41.5%, GCG 37.9%, PAIR 22.2%) are the fairer read there. On JBB the starting prompt is a raw near-zero-ASR behavior (1.0% baseline), so per-pair is already the natural metric.
- Columns are not directly comparable to each other — each has its own baseline/regime (WJB: attack applied to an already-adversarial prompt; JBB: attack applied to a raw behavior; HarmBench: steering-only, no classic-attack runs).
- persona-w has two variants because the direction was fit on two different classifiers (HarmBench-clf: trained on `harmbench_activations` human_jailbreak rows, CV AUC 0.968; WJB-clf: trained on `wildjailbreak_activations`, CV AUC 0.806). Diagonal cells (clf trained and evaluated on the same dataset) are WJB→77.3%, HarmBench→55.0% for HarmBench-clf; WJB→65.6% for WJB-clf's own diagonal is also its WJB cell. HarmBench-clf transfers off-distribution better (55.0%/63.0%) than WJB-clf (42.5%/41.0%), consistent with its higher training AUC.
- HarmBench-clf persona-w's JBB cell (63.0%) is from an earlier run (`trait_steering_attack_jbb_persona_w/harmbench/`) with a wider alpha sweep (`0.1–0.7`, 7 steps) than the rest of this table's rows (`0.1–0.5`, 5 steps). A same-alpha-range rerun this session (`personaw_hbclf_on_jbb`) gave 55.0% — the gap is entirely the extra 0.6/0.7 alpha steps finding additional jailbreaks, not a discrepancy in the method itself. Kept 63.0% here since it's the correct number for that wider sweep; flagging so the two aren't confused if alpha ranges ever get unified.
- PAP's `jbb_judge` in `summary.json` was corrected post-hoc: the resume run's log counted only newly-judged jailbroken pairs (63); the true total across the full `jbb_judgements.jsonl` is 233 pairs / 178 behaviors. The `binary_judge` (headline, GPT-4.1-mini) numbers were never affected.
- GCG was recovered without regeneration (consolidated `test_cases.json` had been merged from 100 per-behavior files; generation was already complete).
- w_aligned's HarmBench cell (39.5%) is from a wider `0.1–0.7` alpha rerun (`harmbench_waligned_a7`), up from 39.0% at `0.1–0.5` — unlike the persona-w JBB case, the extra alpha steps barely moved this number (standard 40.5% vs 39.0%, contextual 77.0% vs 78.0%). A second run with the full 400×112 (behavior×template) pool is in progress, sharded across 20 GPUs, to test w_aligned with templates instead of behavior-only.
