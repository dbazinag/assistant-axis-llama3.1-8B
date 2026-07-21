# File Reference — `full_trait_pipeline/` and `full_trait_tools/`

One short description per file. Deletion candidates are flagged at the bottom.

---

## `full_trait_pipeline/` — the 5-step trait-vector pipeline

These run in order to produce the trait vectors and assistant axis used everywhere else.

| File | What it does |
|------|--------------|
| `1_generate_traits40.py` | Step 1. Generates trait responses by running the model on the 40 questions inside each trait JSON, and saves rich chat-template metadata for later activation extraction. |
| `2_activations_traits40.py` | Step 2. Extracts all-layer activations at multiple token positions from the step-1 generation outputs. |
| `3_judge_traits40.py` | Step 3. Scores the trait responses with each trait's `eval_prompt`, saving simple scores plus separate diagnostics. |
| `4_vectors_traits40.py` | Step 4. Builds trait vectors from the activations, saving an unfiltered set plus several paper-inspired filtered variants, each with a README. |
| `5_make_pc1_axis.py` | Step 5. Computes the PCA-based "assistant axis" (PC1 per layer) from a filtered trait-vector set, saving metadata + README. |

---

## `full_trait_tools/` — sweeps, baselines, data collection, analysis

### Data collection (activations)
| File | What it does |
|------|--------------|
| `collect_harmbench_activations.py` | Core collector: runs Llama-3.1-8B (or OLMo-3) on each (jailbreak_template, behavior) HarmBench pair, extracts layer-16/28 pre-generation activations, and generates + saves responses. Multi-GPU. |
| `collect_attack_activations.py` | Generic collector for attack families PAP/GPTFuzz/PEZ across all HarmBench behaviors. |
| `collect_gcg_activations.py` | Collects activations for GCG attack test cases (the transfer counterpart to the HarmBench collector). |
| `collect_pair_activations.py` | Collects activations for PAIR test cases; emits both the PAIR prompt and a plain-behavior pair per behavior. |
| `collect_harmbench_attack_activations_olmo3.py` | OLMo-3 version: runs OLMo-3 on HarmBench attack test cases (PAIR/PAP/GPTFuzz/PEZ) and collects layer-16/28 states. |
| `collect_jbb_attack_activations.py` | Runs any target model on JBB attack prompts (PAIR/PAP/GPTFuzz/PEZ; GCG skipped) and collects last-prompt-token states. |
| `collect_wildjailbreak_activations.py` | Collects activations + responses for the WildJailbreak eval split (adversarial + direct-harmful columns). |
| `collect_rtv_activations.py` | Forward-only collector that saves prompt-side hidden states at RTV layers (18/25/32, last 5 tokens) for the Mahalanobis baseline. |

### Judging / labelling
| File | What it does |
|------|--------------|
| `judge_harmbench_responses.py` | Re-labels responses with GPT-4.1-mini (jailbroken yes/no), preserving the old HarmBench label as `harmbench_jailbroken`. |
| `estimate_judge_cost.py` | Estimates GPT-4.1-mini judging cost for the pipeline without making any API calls. |

### Main classifier sweeps (current)
| File | What it does |
|------|--------------|
| `run_all_traits_sweep_v2.py` | **Main sweep.** Wide hyperparameter sweep over all-traits projection features with a strict train/tune-on-HarmBench, transfer-eval-only protocol. AUC-primary, threshold calibrated on the refit model. |
| `run_all_traits_model_sweep.py` | Earlier all-traits projection sweep with the same transfer protocol (precursor to v2). |
| `run_all_traits_layers16_28_sweep.py` | Small/fast transfer sweep using concatenated layer-16 + layer-28 trait projections. |
| `run_all_traits_layer_score_fusion_quick.py` | Tests whether layer 28 helps as a score-level residual instead of feature concatenation. |
| `run_all_traits_layer_score_fusion_constrained.py` | 12-line driver: a constrained mild layer-28 residual-fusion variant of the above. |
| `run_trait_pca_sweep.py` | Sweeps number of PCA components applied to the 229-dim trait projection space to find optimal k for transfer. |
| `run_trait_subspace_sweep.py` | Sweeps top-k PCA components of the trait-vector *matrix* (PC1 = assistant axis) as classifier features. |
| `run_mlp_architecture_sweep.py` | Sweeps 20 MLP architectures (simple→deep) × seeds × modes to find the bias-variance sweet spot for transfer. |

### Saved-classifier tools
| File | What it does |
|------|--------------|
| `save_best_olmo3_clf.py` | Trains the best OLMo-3 classifier on all HarmBench OLMo-3 data and saves the artefact dict (pipeline + trait matrix + threshold/sign/meta). |
| `save_best_wjb_clf.py` | Sweeps LogisticRegression variants over WildJailbreak Llama activations and saves the best-AUC model. |
| `eval_olmo3_clf.py` | Evaluates the saved OLMo-3 classifier on any new attack dataset; prints per-family AUC and writes a JSON. |

### Transfer classifiers
| File | What it does |
|------|--------------|
| `fast_transfer_classifier.py` | Fast transfer-classification runner; pre-computes all trait projections as one matmul. Supports raw/pca/all_traits/top_traits modes. |
| `run_transfer_classifier.py` | Generic CLI wrapper around the transfer logic with configurable `--transfer1`/`--transfer2` attack families. |
| `run_transfer_classifier_mlp.py` | Same regime as fast_transfer_classifier but with a 3-layer MLP instead of logistic regression. |

### Baselines (competing detectors)
| File | What it does |
|------|--------------|
| `run_baselines_llama_guard.py` | Runs Llama Guard 3 on all attack families (input-only and input+output), reporting balanced direct AUC. |
| `llama_guard_classifier.py` | Earlier Llama-Guard baseline that trains a logreg on the guard's P(unsafe) score under the transfer regime. |
| `run_baselines_self_exam.py` | Self-exam baseline: asks the model whether it *would* answer each prompt, judges the meta-response with GPT-4.1-mini. Direct + CoT variants. |
| `run_baselines_wildguard.py` | WildGuard baseline using the official interaction-classification prompt; reports direct balanced AUC across families. |
| `run_baselines_gradsafe.py` | Faithful GradSafe (Xie et al. 2024) implementation: gradient-cosine signatures vs reference jailbreak/benign gradients. |
| `gradsafe_classifier.py` | Earlier/standalone GradSafe baseline implementation. |
| `run_baselines_harmbench_ppl_smoothllm.py` | Three scalar baselines in one: HarmBench Llama-2 classifier, prompt perplexity, and SmoothLLM. |
| `perplexity_classifier.py` | Standalone perplexity baseline (log-perplexity feature; reports direct + cross-family AUC). |
| `run_baselines_fjd_paper.py` | Paper-aligned Free Jailbreak Detection (FJD) baseline using first-token transition probability. |
| `run_baselines_jbshield_fjd.py` | Two baselines: a JBShield-style representation direction and an FJD-style compliance-likelihood score. |
| `run_baseline_rtv_mahalanobis.py` | Paper-faithful RTV Mahalanobis outlier detector over refusal-direction fingerprints. |
| `run_baseline_contrastive_geometry.py` | Task-matched contrastive baselines (Mahalanobis MCD / k-NN KCD) trained on failed-vs-successful jailbreaks. |
| `run_baseline_jlt_hidden_tensor.py` | "Jailbreaking-Leaves-a-Trace"-style hidden-state tensor baseline (flatten → standardize → PCA → classify). |

### Steering experiments / attacks
| File | What it does |
|------|--------------|
| `pc1_trait_steering.py` | Steers along traits most aligned with PC1 (layer 16) to suppress/induce jailbreaks; tests cumulative multi-trait effects. |
| `w_aligned_trait_steering.py` | Tests w-aligned vs PC1-aligned traits and the protect-toward vs steer-against asymmetry hypothesis. |
| `steering_robustness_eval_v2.py` | **Current** steering eval: alpha sweep, strict test-only pairs, GPT-4.1-mini 3-way judge (jailbroken/refused/degenerate). |
| `steering_robustness_eval.py` | Earlier (v1) steering robustness eval over fixed conditions — superseded by v2. |
| `steering_magnitude_explorer.py` | Prints raw steered outputs for manual inspection to pick alpha before running the full judged eval. |
| `analyze_steering_results.py` | Reads `steering_robustness_v2_results.json` and prints summary tables + protective/jailbreak flip examples. |
| `category_steering_analysis.py` | Joins steering results to semantic categories and reports jailbreak rates by category. |

### Hyperplane / PCA interpretation & geometry
| File | What it does |
|------|--------------|
| `assistant_axis_predictor.py` | Tests the core hypothesis directly: uses a single scalar (activation projected onto the assistant axis) as a jailbreak predictor and reports AUC — no classifier trained. |
| `classify_jailbreak_logreg_pairs.py` | Logistic regression on persona-vector projections of pre-generation activations, with strict pair-level splits and a variance filter. |
| `classify_jailbreak_raw_activations.py` | Trains logreg on raw 4096-dim activations; saves the hyperplane normal `w` for later interpretation. |
| `compare_hyperplane_to_personas.py` | Interprets the jailbreak hyperplane normal `w` by cosine-ranking it against trait vectors / the assistant axis. |
| `decompose_jailbreak_direction.py` | Two-level decomposition: `w` into PCs, then each PC into trait vectors (trait → PCA → jailbreak chain). |
| `investigate_hyperplane_traits.py` | Stability test (retrain across seeds, compare normals) + point-biserial correlation of each trait projection with outcome. |
| `pc_ablation_study.py` | Ablates PCA-component subsets for classification to find which PCs are useful/necessary/interacting. |
| `pca_component_interpretation.py` | Interprets top PCA components by cosine-comparing each to all trait vectors + assistant axis. |
| `pca_sweep_stability.py` | Sweeps PCA dimensionality measuring normal stability vs AUC; finds (or rules out) a stable sweet spot. |
| `stable_hyperplane_analysis.py` | Uses the sweet-spot PCA dim to build an averaged, stable hyperplane normal and compare it to trait vectors. |
| `project_jailbreak_pre_and_answer_traits40.py` | Projects older jailbreak activations onto the new traits40 axis + trait vectors (pre-response token and answer-mean). |

### Plotting
| File | What it does |
|------|--------------|
| `plot_activation_scatter.py` | 2D scatter of layer-16 jailbreak activations colored by outcome, with trait-vector arrows overlaid. |
| `plot_attack_clusters.py` | PCA scatter of layer-16 activations across all attack families, colored by attack type and jailbreak status. |

### Older / one-off classification scripts
| File | What it does |
|------|--------------|
| `jbb_classification.py` | Large (1196-line) interpretable logreg over assistant-axis / trait projections (JBB-era). |
| `jbc_logreg.py` | Large (869-line) logreg variant adding `--attack_method_filter` to run on a single regime (e.g. JBC). |
| `attack_type_logreg.py` | Tiny baseline predicting jailbreak success from attack-method one-hot features only. |
| `transfer_classifier.py` | Original transfer classifier (train on human jailbreaks, test on GCG); the slow precursor to `fast_transfer_classifier.py`. |

### Inspection / verification (one-off debug)
| File | What it does |
|------|--------------|
| `inspect_hb.py` | Pulls HarmBench HumanJailbreaks + behavior questions from GitHub to eyeball quality before a full run. |
| `validate_classification.py` | Eyeball-checks judge labels by sampling rows, re-running the classifier, and flagging suspiciously short "jailbroken" responses. |
| `find_guard_mismatches.py` | Reruns Llama Guard on a sample and surfaces false negatives/positives vs the GPT-4.1-mini label. |
| `verify_generation.py` | Verifies trait generation output matches the expected 5×2×40 structure. |
| `verify_trait_steering_current.py` | Verifies additive steering by printing greedy generations for the assistant axis and selected traits. |
| `verify_evil_steering.py` | Focused "evil steering" test on layers 16/29 with fixed prompts and residual-norm-scaled additive steering. |

---

## Flagged for deletion / review

**Safe to delete (build artifact):**
- `full_trait_tools/__pycache__/` — compiled Python cache, not source.

**Likely superseded (confirm before deleting):**
- `steering_robustness_eval.py` — superseded by `steering_robustness_eval_v2.py`.
- `transfer_classifier.py` — superseded by `fast_transfer_classifier.py` + `run_transfer_classifier.py`. (Note: it still defines the four feature-mode helpers; check nothing imports it first.)
- `llama_guard_classifier.py` and `gradsafe_classifier.py` — older standalone baselines that appear superseded by the `run_baselines_llama_guard.py` / `run_baselines_gradsafe.py` versions.
- `jbb_classification.py`, `jbc_logreg.py`, `attack_type_logreg.py` — early JBB/JBC-era classification scripts that predate the all-traits sweep pipeline; not referenced in the current handover workflow.

**One-off / disposable (keep only if still useful):**
- `inspect_hb.py`, `verify_generation.py`, `verify_trait_steering_current.py`, `verify_evil_steering.py`, `validate_classification.py` — ad-hoc inspection/verification scripts. Cheap to keep, but not part of the production pipeline.

> Per CLAUDE.md I have only flagged these — none have been deleted. I have **not** verified import graphs, so confirm nothing imports a file before removing it.
