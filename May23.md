# Experiment Log — May 28, 2026

---

## OLMo-3 Trait Sweep Results

### Diagnostic (olmo3-diag6) — 2026-05-28
- All OLMo attack datasets properly GPT-judged (n=gpt_judged for all sets)
- Trait matrices full rank: OLMo all_traits 240 vecs erank=240, corrected 228 erank=228, Llama 229 erank=229
- Centroid AUC raw vs trait (no classifier, just mean-diff direction):
  | Dataset      | Raw AUC | Trait AUC |
  |---|---|---|
  | OLMo HarmBench | 0.9037 | 0.8112 |
  | OLMo PAIR    | 0.7978  | 0.7290  |
  | OLMo PAP     | 0.8502  | 0.7042  |
  | OLMo GPTFuzz | 0.7572  | 0.6018  |
  | Llama HarmBench | 0.9080 | 0.7310 |

### All-traits sweep L16 (olmo3-sweep-all-traits2) — Succeeded
Best by val AUC → logreg_l2_C0.01_bal:
- Human test=0.815, GCG=0.757, PAIR=0.675, PAP=0.600, GPTFuzz=0.575, PEZ=0.839, AvgXfer=**0.689**

### PCA3 sweep L16 (olmo3-sweep-pca3c) — Succeeded
Best by test → extratrees_300_5:
- Human test=0.868, GCG=0.553, PAIR=0.662, PAP=0.610, GPTFuzz=0.585, PEZ=0.817, AvgXfer=**0.645**
- Val-selected extratrees_300_2: Human=0.857, AvgXfer=0.645

### OLMo-3 Baselines
- **LlamaGuard-3** (olmo3-llama-guard3) — Input+Output AUC: HB=0.910, PAIR=0.826, PAP=0.793, GPTFuzz=0.761, PEZ=0.953
- **Self-exam** (olmo3-self-exam2) — Succeeded. Balanced AUC (all ~random — self-exam fails for OLMo-3):
  | Family | N | JB% | Direct AUC | CoT AUC |
  |---|---|---|---|---|
  | HarmBench | 3180 | 25.2% | 0.5706 | 0.5088 |
  | PAIR | 400 | 16.2% | 0.5000 | 0.5077 |
  | PAP | 2000 | 9.6% | 0.5026 | 0.5000 |
  | GPTFuzz | 400 | 58.5% | 0.5060 | 0.5030 |
  | PEZ | 1200 | 7.2% | 0.5116 | 0.5058 |

### WJB Classifier
- save-wjb-clf4: Saved `full_trait_output/wildjailbreak_clf/best_model.pkl` (3.8 MB)
- Best model: logreg_en_C1.0_l1r0.3, val_auc=0.8056, threshold=0.2943

### Corrected-traits sweep L16 (olmo3-corrected) — results.json written
228 filtered trait vectors (erank=228). GCG column still invalid (Llama data).
Best by xfer → logreg_en_C0.03_l1r0.3:
- Human test=0.808, GCG=0.760, PAIR=0.680, PAP=0.610, GPTFuzz=0.579, PEZ=0.854, AvgXfer=**0.696**
- logreg_l2_C0.01_bal: test=0.814, GCG=0.755, PAIR=0.680, PAP=0.612, GPTFuzz=0.583, PEZ=0.847, AvgXfer=**0.695**

### Combined layers 16+28 trait sweep (olmo3-l1628g) — Succeeded
7 model configs, 50 seeds. GCG still invalid (Llama data).
Best by val AUC → logreg_l2_C1_raw:
- Human test=0.756, GCG=0.567, PAIR=0.684, PAP=0.570, GPTFuzz=0.597, PEZ=0.786, AvgXfer=**0.641** (ex-GCG: 0.659)
- **Worse than L16-only trait (0.672 ex-GCG) — combining layers 28 didn't help**

### Trait sweep L28 (olmo3-l28g) — Succeeded
240 trait vectors, layer 28, 50 seeds. GCG column invalid (Llama data).
Best by xfer → linsvm_C0.003_bal:
- Human test=0.805, GCG=0.559, PAIR=0.722, PAP=0.636, GPTFuzz=0.619, PEZ=0.854, AvgXfer=**0.708**
Best tree model (extratrees_500_3): test=0.841, PAIR=0.797, PAP=0.640, GPTFuzz=0.547, PEZ=0.951, AvgXfer=**0.734** (incl. invalid GCG)
- **L28 linear (0.708) > L16 linear (0.695) — layer 28 better for OLMo**
- Val-selected: logreg_l2_C0.03_raw (test=0.834), linsvm_C0.003_bal (test=0.805)

### Raw-linear sweep L16 (olmo3-rawlin7) — Succeeded
50 seeds, raw activations. GCG column invalid (Llama data).
Best by AvgXfer → logreg_l2_C0.01_bal:
- Human test=0.793, GCG=inv, PAIR=0.728, PAP=0.582, GPTFuzz=0.628, PEZ=0.903, AvgXfer=**0.710**
Best by test → logreg_l1_C0.1_raw:
- Human test=0.808, GCG=inv, PAIR=0.692, PAP=0.585, GPTFuzz=0.626, PEZ=0.845, AvgXfer=**0.687**
Val-selected: logreg_l2_C0.01_bal (x17 seeds, test=0.793, AvgXfer=0.710), logreg_l2_C0.01_raw (x21 seeds, test=0.793, AvgXfer=0.709)
- **Raw AvgXfer (0.710) > trait AvgXfer (0.695 L16) — raw linear slightly better on OLMo transfer**

---

## Steering Attacks (Llama-3.1-8B)

### steer-wjb-wjbclf3 — WJB clf steering, top-k=3 — Succeeded
- Per-pair ASR: **44.5%** (GPT-4.1-mini) / 8.6% (JBB rubric)
- Behaviors: 2000 WildJailbreak adversarial_harmful (eval)
- Classifier: WJB clf (logreg_en_C1.0_l1r0.3)

### steer-wjb-adv2 — WJB adversarial mode, top-k=3 — Succeeded (summary valid)
- Per-pair ASR: **56.3%** (GPT-4.1-mini) / 14.25% (JBB rubric)
- Behaviors: 2000 WildJailbreak adversarial_harmful, no templates
- summary.json confirmed written correctly; results cached at `trait_steering_attack_wjb_adv_trait/trait/`

### steer-strong4 — JBB strong sweep (top-k=10,20,30)
- k=10: per-pair **72.6%** (GPT) / 2.0% (JBB), per-behavior **100%** (GPT) / 5.0% (JBB)
- k=20: per-pair **71.1%** (GPT) / **49.8%** (JBB), per-behavior **100%** (GPT) / **100%** (JBB)
- k=30: Suspended (7737/~11200 rows at suspension) — log when resumes and completes

### steer-wjb-clf-strong — WJB clf trait steering, top-k=20 — Succeeded
- Per-pair ASR (GPT-4.1-mini): **47.4%** (948/2000) — BELOW baseline 51.2%
- Per-pair ASR (JBB rubric): 10.4%
- Alpha schedule: [0.1, 0.5, 1.0, 1.5, 2.0, 3.0] | degenerate: 163/2000 (8.2%)
- Jailbreak iteration dist: 769 at α=0.1, 165 at α=0.5, 14 at α≥1.0
- **Negative result**: k=20 direction too diffuse — interferes with already-adversarial prompts
  High alphas useless (only 14 jailbreaks above α=0.5). k=3 (56.8%) > k=20 (47.4%).

### steer-wjb-clf2 — WJB clf trait steering, top-k=3 — Succeeded
- Per-pair ASR: **56.8%** (GPT-4.1-mini) / 9.6% (JBB rubric)
- Behaviors: 2000 WildJailbreak adversarial_harmful (eval)
- Baseline: 51.2% (unsteered); clf-guided steering +5.6pp
- Alpha schedule: [0.1, 0.2, 0.3, 0.4, 0.5]

---

## GCG Generation

- **gcg-jbb-gen2**: Succeeded. 464 "removals" (GCG optimization artifacts). Eval submitted as `eval-gcg-jbb`.
  - Output: `/dlabscratch1/bazina/HarmBench/results/JBB_GCG/llama3_1_8b/test_cases`
- **hb-gcg-olmo3-c0d/c1c/c2d/c3d**: All Running (2d+)
- **hb-gcg-olmo3-c3xa/b/c/d**: Submitted 2026-05-28, behaviors 300-400 split into 4×25 parallel sub-jobs (ETA ~3.6d vs ~14d for c3d alone)
- **merge script**: Updated to handle monolithic chunk3 OR sub-chunks 3a/b/c/d
  - On c0d+c1c+c2d+(c3d OR all c3x*) succeed: run `run_harmbench_gcg_olmo3_merge.sh`

---

## Job Status Summary (2026-05-29 13:30 UTC)

| Job | Status | Action |
|---|---|---|
| olmo3-sweep-all-traits2 | Succeeded | Logged |
| olmo3-sweep-pca3c | Succeeded | Logged |
| olmo3-llama-guard3 | Succeeded | Logged |
| save-wjb-clf4 | Succeeded | Confirmed |
| steer-wjb-wjbclf3 | Succeeded | Logged |
| olmo3-sweep-raw2 | Failed (script missing) | Fixed → olmo3-rawlin7 |
| steer-wjb-adv2 | Succeeded | Logged |
| olmo3-rawlin7 | Succeeded | Logged (best AvgXfer=0.710) |
| olmo3-l28g | Succeeded | Logged |
| olmo3-l1628g | Succeeded | Logged |
| olmo3-corrected | Succeeded | Logged |
| olmo3-self-exam2 | Succeeded | Logged (all AUCs ~0.50 — self-exam fails for OLMo) |
| steer-wjb-clf2 | Succeeded | Logged (56.8% GPT ASR, 9.6% JBB) |
| gcg-jbb-gen2 | Succeeded | Eval submitted as eval-gcg-jbb |
| steer-strong4 k=20 | Done | Logged |
| steer-strong4 k=30 | Suspended (7737/~11200 rows) | Log when resumes/completes |
| eval-gcg-jbb | Running | Log ASR on success |
| steer-wjb-clf-strong | Running | Log per-pair GPT ASR from summary.json when done |
| hb-gcg-olmo3-c0d/c1c/c2d | Running | — |
| hb-gcg-olmo3-c3d | Running (superseded by c3x*) | — |
| hb-gcg-olmo3-c3xa/b/c/d | Running (c3xb bottleneck ~12s/step; est. done ~June 1) | Merge when all done |
| file-bridge60 | Failed (double-sshd from bad cmd) | Fixed: file-bridge61 submitted |
| file-bridge61 | Submitted | Port-forward when Running |
| steer-wjb-adv3 | Failed (bad --behaviors_source arg) | Not resubmitting |
