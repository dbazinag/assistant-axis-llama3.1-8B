# Final Presentation — Detecting and Inducing Jailbreaks with Persona-Trait Directions

**Format:** paste-ready Google Slides content · **Target:** ~10 min, lab/research-group audience
**Framing:** midterm already covered motivation + what a persona vector is + early detection/steering. This talk = the finished **two-sided causal story** + efficiency + generality.

---

## Slide 1 — Title

**Detecting *and* Inducing Jailbreaks with Persona-Trait Directions**
Dominic Bazina-Grolinger · w/ V. Moskvoretskii, R. West · EPFL DLab
*Llama-3.1-8B-Instruct · HarmBench*

> Notes: One line of setup — "Since the midterm I've finished both halves and started testing generality. The headline is that the same trait directions both detect and cause jailbreaks."

---

## Slide 2 — Recap & the question (the through-line)

- **Persona/trait vectors** (recap): behaviors ≈ linear directions in activation space; we extract one per trait and they're *steerable*.
- **The question:** are these trait directions related to jailbreaking — and is it just correlation, or **causal**?
- **The approach that answers "causal":** use the *same* trait subspace from **two sides**
  - **Detect** — can trait scores spot a jailbreak? (correlation)
  - **Induce** — does steering along them *make* the model comply? (causation)

> Notes: This slide is the spine of the whole talk. The two-sided design is the contribution, not either result alone. Detection alone = correlation; the steering half is what licenses the causal claim.

---

## Slide 3 — Setup: one subspace, two uses  *(figure slide)*

- **Trait extraction:** diff-of-means over 240 traits (positive vs negative persona prompts), layer-16 residual stream → **~229 unit-norm directions**, matrix T.
- Project any activation: **Tx = trait scores** (how strongly each trait fires).
- *[Insert `fig_pipeline.png`]* — same T feeds a **detector** (classifier on trait scores) and a **steering** direction (classifier normal mapped back to activations).

> Notes: Keep brief — they saw extraction at midterm. The one new emphasis: everything downstream comes from this single subspace, which is why detect+induce share a mechanism.

---

## Slide 4 — Detection: the hard test is *transfer*

- Train + tune the classifier **only on HarmBench** (strict behavior/template split).
- Evaluate on **5 attack families never seen in training**: GCG, PAIR, PAP, GPTFuzz, PEZ.
- Metric: AUC, mean over 50 seeds. **Transfer = the number that matters** (production faces unseen attacks).
- Labels from a GPT-4.1-mini judge.

> Notes: Stress the train-on-one / test-on-five protocol — this is what separates us from detectors that train and test on the same attack.

---

## Slide 5 — Detection result: traits transfer best  *(KEY)*

*[Visual: `presentation_figures/fig_id_vs_transfer.png` — ID-vs-transfer scatter, shows the overfit story at a glance. Or the report's `fig_detection_transfer.png`, or the table below.]*

| Method | HarmBench (ID) | Avg. Transfer |
|---|---|---|
| **Trait projections (ours)** | 0.86 | **0.76** |
| Raw 4096-d activation | 0.81 | 0.71 |
| Mahalanobis contrastive | 0.92 | 0.71 |
| JB-Leaves-a-Trace (SVM) | 0.91 | 0.70 |
| **GradSafe** | **0.95** | **0.62** |
| Llama Guard (input) | 0.69 | 0.61 |

- **Trait projections = best transfer (0.76)** of any method, and best overall.
- **The story:** the in-distribution winners *overfit*. GradSafe tops HarmBench (0.95) but **collapses to 0.62** on unseen attacks. Trait features generalize.
- It's the **trait structure**, not just layer-16 activations: trait proj (0.76) > raw (0.71) > PCA-3 (0.68).

> Notes: The GradSafe 0.95→0.62 contrast is the punchline — say it out loud. "The methods that win where they're trained are the ones that fall apart out of distribution."

---

## Slide 6 — Bonus: we can *read* which traits drive it

- Features are **named traits**, so the detector is interpretable (most internal detectors aren't).
- **Pushes toward refusal:** paranoid, confrontational, analytical, circumspect (cautious/guarded).
- **Pushes toward jailbreak:** risk-taking, blunt, impulsive, sassy (disinhibited).
- The attack's steering axis aligns with edgy personas (sardonic, wry, sarcastic) — same story from the attack side.

> Notes: Quick slide, but it's a real differentiator. Detector + attack agree on the intuitive picture: a jailbreak is the model shifting its assistant persona.

---

## Slide 7 — ...and detection is essentially *free*  *(efficiency)*

- **Framing:** in production the model already runs the forward pass — activations are a free byproduct. Count only the *added* detection compute.

*[Visual: `presentation_figures/fig_accuracy_vs_cost.png` — transfer AUC vs latency, top-left = best. Or the table below.]*

| Method | Added latency / request | Needs |
|---|---|---|
| **Trait projection (ours)** | **0.07 ms** | proj (4096→229) + logreg |
| Raw activation head | 0.004 ms | a dot product |
| Perplexity | 33 ms | 1 extra forward pass |
| Llama Guard | 48 ms | a separate 8B model |
| WildGuard | 510 ms | guard model + generation |
| FJD | 1,094 ms | extra generation |
| Self-Exam | 7,496 ms | 2 extra generations |
| GradSafe | 20,440 ms | forward **+ backward** pass |

- **Punchline:** our detector adds **<0.1 ms** — **~500–700× faster** than the cheapest baseline (perplexity / Llama Guard), **~300,000× faster** than GradSafe.
- It's just a logistic regression on activations the model already computed; every baseline needs a *second model*, *extra generation*, or a *backward pass*.
- **So: best transfer AND near-zero cost.** Trait projection beats Llama Guard on transfer *and* is ~700× cheaper.

> Notes: This is the practical selling point. Measured on OLMo-3-7B / A100 (same scale as Llama → representative). Prompt-based baselines timed on short prompts (mildly optimistic); guard methods on full responses. JBShield/JLT/Mahalanobis weren't timed but are the same cheap activation-head class. Pair this slide with slide 5: accuracy *and* cost both favor the trait head.

---

## Slide 8 — Attack: steering along the same directions  *(KEY / causal)*

- Map the classifier's decision boundary **normal (w)** back to activation space; add it to layer 16 during generation, escalating strength α.
- **Fair test = "started-refused" prompts:** evaluate only on prompts the *unsteered* model refused. Flipping a genuine refusal is the clearest causal signal.

*[Visual: `presentation_figures/fig_attack_asr.png` — HarmBench-refused vs JBB grouped bars. Or the table below.]*

| Attack | HarmBench (refused→broken) | JBB ASR |
|---|---|---|
| **Steer on w (persona)** | **0.747** | 0.63 |
| Steer on w (raw) | 0.76 | 0.68 |
| Steer-traits (top-k) | 0.666 | **0.16** |
| GPTFuzz (best standard) | 0.59 | 0.77 |
| GCG / PAIR / PAP / PEZ | 0.44 / 0.39 / 0.28 / 0.15 | — |

- Steering **flips ~75% of refusals** — more than any standard attack (GPTFuzz 0.59).
- **Transfers** to a 2nd benchmark (JBB): steer-w holds (0.63–0.68); **sparse top-k steer-traits collapses (0.16)** → dense beats sparse.

> Notes: Two messages — (1) it beats real attacks at flipping refusals; (2) the dense classifier-normal transfers but the sparse top-k overfits HarmBench. Don't oversell: GPTFuzz still wins on JBB.

---

## Slide 9 — Why this is *causal*, and the catch

- The persona normal lives in only **~229 dims**; the raw normal uses all 4096 — yet they're within a few points (**0.747 vs 0.76** refused; 0.63 vs 0.68 JBB).
- → **Almost all the refusal-flipping signal already lives in the trait subspace.** Dropping ~3900 dims costs ~nothing.
- **Two-sided takeaway:** trait directions don't just *track* the jailbroken state — moving along them helps *produce* it.
- **Honest caveat:** "some causality," not a clean single-direction edit — it uses a trained classifier + judge-in-the-loop to pick where/when to stop.

> Notes: This is the intellectual payoff slide. The 229-vs-4096 comparison is the cleanest evidence that traits carry the causal signal.

---

## Slide 10 — Generality (in progress), limits, takeaway

- **Generality:** extending to **OLMo-3** (early results weaker) and **Gemma** (running). Can't yet claim it's model-general.
- **Limits:** single model + single training benchmark; PAP is hard for *everyone* (~0.63); all rates inherit the GPT-4.1-mini judge's biases.
- **Takeaway:** *On Llama-3.1-8B, the same persona-trait subspace gives the best-transferring jailbreak detector AND the strongest refusal-flipping attack — so traits are tied to jailbreaking, partly causally.*
- **Next:** confirm generality; sweep layer / k / α; dual use — transferable detector + red-teaming probe.

> Notes: End on the one-sentence takeaway in bold. Be upfront that generality is the open question — it's also your current work, which is a good note to end a lab talk on.
