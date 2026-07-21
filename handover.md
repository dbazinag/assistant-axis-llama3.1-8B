# Handover — assistant-axis-llama3.1-8B

**Written:** 2026-05-28 | **Author:** Claude Code session

---

## 1. Project Overview

This project develops **trait-vector-based jailbreak detection** for LLMs. The core idea: extract a set of persona "trait" vectors from the model's hidden states by running the model on personas of different psychological traits, then project layer activations onto those vectors as features for a linear classifier. The classifier is trained on human-labelled HarmBench responses (jailbroken vs not), then evaluated on its ability to detect attacks it was never trained on (PAIR, PAP, GPTFuzz, PEZ, GCG).

The primary target model is **Llama-3.1-8B-Instruct**. A parallel track targets **OLMo-3-7B-Instruct** to test generality. There is also a jailbreak attack component: steering the model's trait activations toward a computed "jailbreak direction" during generation to achieve high attack success rates.

**Current status:** Multiple experiments running in parallel on the EPFL RCP cluster. Several have completed (sweeps, steering attacks) with results logged in `May23.md`. Several are still running (GCG generation, self-exam baseline, raw sweep, k=20/k=30 steering sweep).

---

## 2. Repository & Environment

### Local (your WSL2 machine)
- **Project root:** `/home/dbazinag/projects/assistant-axis-llama3.1-8B`
- This is a git repo that mirrors the cluster copy. Code edits are made here and synced (or edited directly on cluster).

### Cluster (EPFL RCP scratch)
- **Cluster path:** `/dlabscratch1/bazina/assistant-axis-llama3.1-8B`
- Inside RunAI pods the PVC is mounted at `/mnt`, so the path becomes `/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B`
- These two paths refer to the same files: cluster jobs run scripts from `/mnt/...`, SSH from outside uses `/dlabscratch1/...`

### Key subdirectories

| Dir | Purpose |
|-----|---------|
| `assistant_axis/` | Core library: axis computation, steering, PCA, generation |
| `full_trait_pipeline/` | 5-step pipeline: generate traits → activations → judge → vectors → PC1 axis |
| `full_trait_tools/` | All sweep scripts, baseline runners, attack evaluation, data collection |
| `jailbreaking_tools/` | Trait-steering attack scripts (`trait_steering_attack_jbb.py` is the main one) |
| `data/` | Static data: trait definitions, role prompts, extraction questions |
| `full_trait_output/` | **All outputs live here** (on cluster only, ~hundreds of GB) |
| `results/` | Small CSV/JSON summary files committed to git |
| `run_*.sh` | One script per experiment; these are the entry points submitted to RunAI |

### Virtual environment
- **uv** is the package manager: `/dlabscratch1/bazina/.local/bin/uv`
- Venv location: `/dlabscratch1/bazina/assistant-axis-llama3.1-8B/.venv`
- Python ≥3.10, PyTorch 2.9.1+cu128, vllm 0.15.0 (pinned via `uv.lock`)
- To install from scratch on cluster: `cd /dlabscratch1/bazina/assistant-axis-llama3.1-8B && uv sync`
- To run a script: `uv run python full_trait_tools/SCRIPT.py [args]`

### Secrets / config files
- `/dlabscratch1/bazina/assistant-axis-llama3.1-8B/.env` — contains `OPENAI_API_KEY` (needed by any script that calls GPT-4.1-mini for judging). Source it with `set -a; source .env; set +a` before running judge scripts.
- HuggingFace token: set via `HF_TOKEN` env var in scripts that pull gated models (e.g. Llama-3.1-8B). Some scripts hardcode it inline; check `run_wjb_collect.sh` for the pattern.
- HF cache: `/dlabscratch1/bazina/.cache/huggingface`

---

## 3. Infrastructure & Remote Access

### RunAI cluster
- **CLI:** `runai-rcp-prod` (installed locally, not over SSH)
- **Project:** `dlab-bazina`
- **Suppress deprecation warnings:** always prefix with `SUPPRESS_DEPRECATION_MESSAGE=true` — see Known Issues below
- **List jobs:** `SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod list jobs -p dlab-bazina`
- **Submit job:** `SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod submit <name> -p dlab-bazina --image ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt [--gpu N] -- bash /mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/run_SCRIPT.sh`
- **Pull logs:** `SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod logs <jobname> -p dlab-bazina`

### SSH access via file-bridge
Cluster nodes are not directly accessible. You SSH through a permanent interactive pod called `file-bridge61` that runs `sleep infinity`. The pod gets **suspended** by the cluster after idle time but can be **resumed** — do not submit a new one, just resume the existing job.

**Current bridge:** `file-bridge61` (use `runai resume` if Suspended)

**Resume if suspended:**
```bash
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod resume file-bridge61 -p dlab-bazina
```

**Wait for Running, then start port-forward:**
```bash
# Get pod name
kubectl get pods -n runai-dlab-bazina --field-selector=status.phase=Running \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep file-bridge61

# Port-forward (background)
kubectl port-forward -n runai-dlab-bazina file-bridge61-0-<N> 2242:22 > /tmp/pf61.log 2>&1 &
sleep 5
```

**SSH in:**
```bash
ssh -i ~/.ssh/id_ed25519 -p 2242 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -l bazina localhost
```

Wait ~10s after resume for the GASPAR user to be created before SSH-ing. Poll:
```bash
until ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.ssh/id_ed25519 -p 2242 -l bazina localhost "echo ok" 2>/dev/null; do sleep 5; done
```

**If file-bridge61 is ever permanently lost**, submit a new one with `sleep infinity` (not `sleep 7200`):
```bash
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod submit file-bridge62 -p dlab-bazina \
  --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
  --pvc dlab-scratch:/mnt --node-pools cpu \
  --interactive --cpu 4 --memory 16G -- sleep infinity
```
The image starts sshd automatically — do NOT add a custom `bash -c "useradd ... sshd"` block.

### Cluster paths
- Project root: `/dlabscratch1/bazina/assistant-axis-llama3.1-8B` (via SSH) = `/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B` (inside pods)
- HarmBench repo: `/dlabscratch1/bazina/HarmBench` — used for GCG generation; results at `/dlabscratch1/bazina/HarmBench/results/`
- All experiment outputs: `/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output/`
- Model cache: `/dlabscratch1/bazina/.cache/huggingface/`

### VPN requirement
Both the RunAI API and SSH port-forward require the EPFL VPN to be active. The WSL2 virtual network adapter sometimes drops (gateway `172.27.80.1` becomes unreachable, breaking DNS for `caas-prod.rcp.epfl.ch`). Fix: run `wsl --shutdown` from Windows PowerShell, then reopen your WSL terminal and reconnect to VPN.

---

## 4. How to Run Things

### General pattern for submitting a training job
```bash
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod submit <jobname> \
  -p dlab-bazina \
  --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
  --pvc dlab-scratch:/mnt \
  --gpu 1 \
  -- bash /mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/run_SCRIPT.sh
```

For CPU-only jobs (sweep scripts, judge scripts), omit `--gpu 1`.

### Running scripts locally on the cluster (via SSH)
```bash
ssh bazina@... "cd /dlabscratch1/bazina/assistant-axis-llama3.1-8B && uv run python full_trait_tools/SCRIPT.py [args]"
```

### Key script entry points

| Script | What it does |
|--------|-------------|
| `run_steer_trait_jbb_strong.sh` | Trait-steering jailbreak attack, k=10/20/30 sweep, alpha 0.1–0.7 |
| `run_eval_gcg_jbb.sh` | Evaluate GCG attack results (submit after gcg-jbb-gen2 succeeds) |
| `run_all_traits_sweep_olmo3_all_traits.sh` | OLMo-3 trait feature sweep (classification) |
| `run_all_traits_sweep_olmo3_pca3.sh` | OLMo-3 PCA3 feature sweep (outperforms trait) |
| `run_baselines_self_exam_olmo3.sh` | OLMo-3 self-exam baseline (asks model to self-evaluate) |
| `run_harmbench_gcg_olmo3_merge.sh` | Merge 4 GCG chunks into one test_cases.json |
| `run_save_olmo3_clf.sh` | Save best OLMo-3 classifier to pkl |
| `run_save_wjb_clf.sh` | Save best WildJailbreak classifier to pkl |

### Output locations
- Sweep results: `full_trait_output/all_traits_sweep_v2_*/results.json`
- Steering results: `full_trait_output/trait_steering_attack_jbb_strong_k*/trait/summary.json`
- Self-exam baseline: `full_trait_output/baselines_all_attacks_olmo3/self_exam_all_attacks.json`
- Saved classifiers: `full_trait_output/wildjailbreak_clf/best_model.pkl`, `full_trait_output/all_traits_sweep_v2_olmo3/best_model.pkl`
- Llama detection/transfer table (consolidated Table 1, per-method AUCs + sources): `full_trait_output/llama_detection_table_consolidated.json` (cluster) and `report/llama_detection_table_consolidated.json` (repo)
  - Regenerate from the canonical result dirs with `full_trait_tools/build_llama_detection_table.py` (documents each row's source; Trait←sweep logreg, Raw/PCA←fast_transfer, baselines←their dirs)
- **All paper-used data indexed under `full_trait_output/paper_data/`** — a symlinked, browsable tree (originals untouched, nothing duplicated): `detection_llama/` (Table 1), `detection_gemma/` (Table 2), `steering/{harmbench,jbb,wildjb}/` (Table 4), `analysis/` (Tables 5–8 + fig inputs), `building_blocks/` (persona/trait vectors, trait matrix, generation/judge, static inputs), `activations/` (HarmBench + 5 families + WJB). Every entry was verified to back a paper number before inclusion.
- Superseded/non-canonical Llama transfer runs archived (intact, not deleted) under `full_trait_output/_archive/` — see `_archive/README.md`

### Reading results
```bash
# Rank models by test AUC from a sweep
ssh bazina@... "python3 -c \"
import json
d = json.load(open('full_trait_output/SWEEP_DIR/results.json'))
rows = [(v.get('human_test',{}).get('auc',{}).get('mean',0), v.get('val_auc',{}).get('mean',0), m,
         {k: v[k]['auc']['mean'] for k in ['PAIR','PAP','GPTFuzz','PEZ'] if k in v})
        for m, v in d['summary'].items()]
rows.sort(reverse=True)
for rank, (t, val, m, tr) in enumerate(rows[:5], 1):
    print(rank, m, f'val={val:.4f}', f'test={t:.4f}', tr)
\""
```

```bash
# Read steering summary
ssh bazina@... "python3 -c \"
import json
d = json.load(open('full_trait_output/trait_steering_attack_jbb_strong_k10/trait/summary.json'))
print('per_pair_asr:', d['gpt41mini']['per_pair_asr'])
print('per_behavior_asr:', d['gpt41mini']['per_behavior_asr'])
print('jbb_rubric:', d['jbb_rubric']['per_pair_asr'])
\""
```

---

## 5. Current Task & Progress

### Completed this session
- **steer-wjb-wjbclf3** (Succeeded): WJB clf in-loop steering → 44.6% per-pair ASR (GPT-4.1-mini), 8.6% JBB rubric
- **olmo3-sweep-all-traits2** (Succeeded): 240-trait OLMo-3 sweep → best-by-test `logreg_l1_C0.01_raw` val=0.905, test=0.828; best-transfer `logreg_l2_C0.1_bal` transfer_mean=0.680
- **olmo3-sweep-pca3c** (Succeeded): PCA3 OLMo-3 sweep → best `extratrees_300_5` test=0.868 (significantly better than trait features at 0.828)
- **olmo3-llama-guard3** (Succeeded): Llama Guard AUC on OLMo-3 attacks
- **save-wjb-clf4** (Succeeded): WJB classifier saved to `full_trait_output/wildjailbreak_clf/best_model.pkl`
- **steer-strong4 k=10** (Succeeded): 72.6% per-pair ASR, 100% per-behavior ASR, 2.0% JBB rubric (alpha 0.1–0.7)
- **steer-strong4 k=20** (In progress, ~98.5% done — 11031/11200 results at last check)

### Currently running
| Job | Node | Status | Notes |
|-----|------|--------|-------|
| `gcg-jbb-gen2` | gpu002 | Running (2d) | GCG generation for JBB behaviors (Llama); saves all results at end |
| `hb-gcg-olmo3-c0d` | gpu006 | Running | OLMo-3 HarmBench GCG chunk 0 |
| `hb-gcg-olmo3-c1c` | gpu022 | Running | OLMo-3 HarmBench GCG chunk 1 |
| `hb-gcg-olmo3-c2d` | gpu024 | Running | OLMo-3 HarmBench GCG chunk 2 |
| `hb-gcg-olmo3-c3d` | gpu018 | Running | OLMo-3 HarmBench GCG chunk 3 |
| `steer-strong4` | gpu031 | Running | k=20 nearly done (~98.5%), k=30 is next in same job |
| `olmo3-sweep-raw3` | gpu206 | Running | OLMo-3 raw activation sweep (no trait projection) |
| `olmo3-self-exam2` | gpu021 | Running | OLMo-3 self-examination baseline |
| `file-bridge49` | - | Running | SSH bridge, expires ~2h after submission |

### Next actions when jobs complete

**`gcg-jbb-gen2` → Succeeded** (CRITICAL):
```bash
ssh bazina@... "bash /mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/run_eval_gcg_jbb.sh"
```
Or via RunAI: submit a job running `run_eval_gcg_jbb.sh`. Verify `JBB_GCG/` directory appeared in `/dlabscratch1/bazina/HarmBench/results/` first.

**ALL 4 of `hb-gcg-olmo3-c0d/c1c/c2d/c3d` → Succeeded** (must be all 4):
```bash
ssh bazina@... "bash /mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/run_harmbench_gcg_olmo3_merge.sh"
```

**`steer-strong4` k=20 done** (watch `results.jsonl` reach 11200 lines):
- Read `full_trait_output/trait_steering_attack_jbb_strong_k20/trait/summary.json`
- Log to `May23.md` in the k-sweep table

**`steer-strong4` k=30 done** (after k=20):
- Read `full_trait_output/trait_steering_attack_jbb_strong_k30/trait/summary.json`
- Log to `May23.md`

**`olmo3-self-exam2` → Succeeded**:
- Read `full_trait_output/baselines_all_attacks_olmo3/self_exam_all_attacks.json`
- Log Direct AUC + CoT AUC per family (HarmBench, PAIR, PAP, GPTFuzz, PEZ) to `May23.md`

**`olmo3-sweep-raw3` → Succeeded**:
- Read `full_trait_output/all_traits_sweep_v2_olmo3_raw/results.json` (same ranking snippet as above)
- Log transfer AUCs to `May23.md`

**File-bridge:**
- `file-bridge61` is the permanent bridge — resume it when suspended, don't create new ones
- `SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod resume file-bridge61 -p dlab-bazina`

---

## 6. Known Issues & Past Mistakes

### `--suppress-deprecation-message` as trailing arg — BREAKS JOB
**Symptom:** Job fails immediately with `sleep: unrecognized option '--suppress-deprecation-message'`  
**Root cause:** When passed as a trailing arg, RunAI forwards it into the container where it gets appended to the `sleep 7200` command.  
**Fix:** Always use `SUPPRESS_DEPRECATION_MESSAGE=true` as an env var *prefix* before `runai-rcp-prod`, not as a trailing `--suppress-deprecation-message` flag.

### Sign bug in `trait_steering_attack_jbb.py` line 594 — FIXED
**Symptom:** Steering attack had inverted acceptance criterion; jailbroken responses were being rejected.  
**Root cause:** `if score >= eff_threshold:` didn't account for sign.  
**Fix (applied):** `if _sign * score >= _sign * eff_threshold:`  
**Note:** All completed results post-fix are correct. `steer-strong4` was submitted after the fix.

### `olmo3-self-exam2` has NO output caching
**Symptom:** Every time the job restarts (e.g. due to eviction), it regenerates all meta-responses from scratch.  
**Root cause:** `run_baselines_self_exam.py` caches only the *input* response files, not its own meta-response output (`self_exam_harmbench_meta_responses.jsonl` etc.). 3180 HarmBench rows × ~35s each ≈ 10h just for HarmBench family.  
**Implication:** Network outages that cause job eviction restart the 10h+ computation from zero.

### WSL2 virtual network adapter drops under long VPN sessions
**Symptom:** `ping 172.27.80.1` → 100% packet loss; DNS lookup for `caas-prod.rcp.epfl.ch` fails.  
**Root cause:** WSL2 virtual network adapter becomes stale.  
**Fix:** `wsl --shutdown` from Windows PowerShell, then reopen WSL terminal and reconnect VPN.  
**Impact:** Kills all RunAI CLI calls and port-forwards until fixed.

### Port-forward conflict
**Symptom:** `Failed to create port forwarding... unable to listen on [{2242 22}]`  
**Root cause:** A stale port-forward process is still holding port 2242.  
**Fix:** `pkill -f "port-forward"` before starting a new one.

### `results.json` key structure is non-obvious
When reading sweep results, the structure is:
```python
data['summary'][model_name]['val_auc']['mean']           # validation AUC
data['summary'][model_name]['human_test']['auc']['mean'] # human-labelled test AUC
data['summary'][model_name]['PAIR']['auc']['mean']       # transfer AUC for PAIR
```
Do NOT try `val_auc = data['summary'][model]['val_auc']` directly — it's a dict, not a float. This caused a `NameError` earlier in the session.

### GCG output only appears after ALL behaviors complete
**Symptom:** `JBB_GCG/` directory does not exist even after hours of running.  
**Root cause:** HarmBench's GCG script saves everything in memory and writes at the very end; there are no per-behavior checkpoints.  
**Implication:** Cannot check intermediate progress by counting files. Use log tailing instead.

### OLMo-3 uses `trust_remote_code=True`
All scripts that load OLMo-3 need `--trust_remote_code` flag or `trust_remote_code=True` in the `AutoModel.from_pretrained` call. The full_trait_pipeline step 1 already has this, but if writing new OLMo-3 scripts, remember to add it.

### WJB label field inconsistency
`classified_responses.jsonl` from WildJailbreak uses `jailbroken` not `binary_jailbroken`. `save_best_wjb_clf.py` now falls back correctly (`.get("binary_jailbroken", r.get("jailbroken"))`), but if writing new scripts that consume WJB labels, check which field name the file actually uses.

---

## 7. Key Files & What They Do

| File | Purpose | Last modified reason |
|------|---------|---------------------|
| `May23.md` | Run log: all results, job events, network outages. Ground truth for current state. | Updated throughout this session with new results |
| `full_trait_tools/run_all_traits_sweep_v2.py` | Main sweep script: trains classifiers on trait-projected activations, reports val + test + transfer AUCs | Added balanced/raw split, PCA feature type |
| `jailbreaking_tools/trait_steering_attack_jbb.py` | Trait-steering jailbreak attack. Steers top-k trait axes during generation. | Sign fix at line 594 |
| `full_trait_tools/collect_harmbench_activations.py` | Collects Llama-3.1-8B activations + responses on HarmBench behaviours; supports OLMo-3 via `--model` + `--trust_remote_code` | OLMo-3 support added |
| `full_trait_tools/collect_harmbench_attack_activations_olmo3.py` | Collects OLMo-3 activations for attack families (PAIR/PAP/GPTFuzz/PEZ) on HarmBench | New file this session |
| `full_trait_tools/collect_wildjailbreak_activations.py` | Collects activations for WildJailbreak eval split (adversarial + direct harmful rows) | New file this session |
| `full_trait_tools/judge_harmbench_responses.py` | GPT-4.1-mini judge for classifying responses as jailbroken/not | Updated to accept `--responses_path` |
| `full_trait_tools/run_baselines_self_exam.py` | Self-exam baseline: asks the model to classify its own responses | No output caching — restarts from scratch on job restart |
| `full_trait_tools/run_baselines_llama_guard.py` | Llama Guard baseline AUC across attack families | Supports `--olmo3` flag |
| `full_trait_tools/save_best_wjb_clf.py` | Sweeps LogisticRegression variants on WJB data, saves best to pkl | Fixed WJB label field |
| `full_trait_tools/save_best_olmo3_clf.py` | Saves best OLMo-3 classifier (from all-traits sweep) to pkl | New file |
| `full_trait_tools/eval_olmo3_clf.py` | Evaluates a saved OLMo-3 clf on all attack families | New file |
| `run_steer_trait_jbb_strong.sh` | Runs steer-strong4 k=10/20/30 sweep (alpha 0.1–0.7) | This is what steer-strong4 job runs |
| `run_harmbench_gcg_olmo3_merge.sh` | Merges 4 GCG chunk outputs for OLMo-3 | Lives on cluster, not in local git |

---

## 8. Data & Checkpoints

### Training / evaluation data (all on cluster)

| Dataset | Path | Description |
|---------|------|-------------|
| HarmBench behaviours | `/dlabscratch1/bazina/HarmBench/data/behavior_datasets/` | 159/200 standard behaviours; JBB behaviours in `jbb_behaviors.csv` |
| Llama HarmBench activations | `full_trait_output/harmbench_activations/` | Layer-16 + layer-28 activations for all Llama human-request pairs |
| OLMo-3 HarmBench activations | `full_trait_output/harmbench_activations_olmo3/` | Same for OLMo-3 |
| OLMo-3 attack activations | `full_trait_output/{pair,pap,gptfuzz,pez}_activations_olmo3_hb/` | Per-family transfer test sets |
| WildJailbreak activations | `full_trait_output/wildjailbreak_activations/` | Adversarial + direct harmful pairs from WJB eval split |
| Trait vectors (Llama) | `full_trait_output/traits40_vectors/pre_generation_last_token/all_traits_no_filter/` | 229 × 4096 trait direction matrix |
| Trait vectors (OLMo-3) | `full_trait_output/traits40_vectors_olmo3_7b/pre_generation_last_token/all_traits_no_filter/` | 240 × 4096 trait direction matrix |
| Trait matrix (layer 16) | `full_trait_output/trait_matrix_layer16.npy` | Normalised trait matrix used by classifiers |
| GCG test cases (JBB) | `/dlabscratch1/bazina/HarmBench/results/JBB_GCG/llama3_1_8b/test_cases/` | Only exists after gcg-jbb-gen2 completes |
| GCG test cases (OLMo-3) | `/dlabscratch1/bazina/HarmBench/results/HarmBench_GCG/olmo3_7b/test_cases/` | Only exists after all 4 `hb-gcg-olmo3-c*d` merge |

### Saved classifiers
- Llama best model: `full_trait_output/all_traits_sweep_v2/best_model.pkl` (or `all_traits_sweep_v2_olmo3/best_model.pkl` for OLMo-3 cross-model)
- OLMo-3 best model: `full_trait_output/all_traits_sweep_v2_olmo3/best_model.pkl` — see `run_save_olmo3_clf.sh`
- WJB classifier: `full_trait_output/wildjailbreak_clf/best_model.pkl`

### Artefact pkl format
```python
{
    "pipeline":     sklearn.Pipeline (StandardScaler + LogisticRegression),
    "trait_matrix": np.ndarray [n_traits × 4096],
    "trait_names":  list[str],
    "layer":        int,  # 16
    "threshold":    float,
    "sign":         int,  # +1 or -1
    "meta":         dict  # model name, cv_auc, val_auc, etc.
}
```

---

## 9. External Dependencies & APIs

| Service | Purpose | Auth |
|---------|---------|------|
| OpenAI API (GPT-4.1-mini) | Judge for classifying model responses as jailbroken/not | `OPENAI_API_KEY` in `/dlabscratch1/bazina/assistant-axis-llama3.1-8B/.env` |
| HuggingFace Hub | Model downloads (Llama-3.1-8B-Instruct, OLMo-3-7B-Instruct, Mixtral-8x7B) | `HF_TOKEN` env var; token stored at `/dlabscratch1/bazina/.cache/huggingface/token` |
| EPFL RCP / RunAI | GPU compute; job scheduling | `runai-rcp-prod` CLI, EPFL VPN required, project `dlab-bazina` |
| Llama Guard 3 | Input-only / input+output jailbreak detection baseline | Loaded from HF (same HF_TOKEN) |
| WildJailbreak dataset | Adversarial jailbreak pairs for evaluation | `allenai/wildjailbreak` via HuggingFace `datasets` |

**GPT-4.1-mini cost note:** Judge scripts call GPT-4.1-mini per (behavior, response) pair. Large runs (HarmBench self-exam: 3180 rows, 5 families = ~15K calls) can be expensive. Always verify `.env` is loaded before submitting judge jobs.

---

## 10. Useful One-Liners

```bash
# List all jobs, filter to interesting ones
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod list jobs -p dlab-bazina 2>&1 | \
  grep -E "Running|Failed|Pending" | grep -v "file-bridge[0-9]*  Succeeded"

# Resume the permanent file-bridge (do this instead of submitting a new one)
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod resume file-bridge61 -p dlab-bazina

# Start SSH port-forward (get pod name first)
POD=$(kubectl get pods -n runai-dlab-bazina --field-selector=status.phase=Running \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep file-bridge61)
kubectl port-forward -n runai-dlab-bazina $POD 2242:22 > /tmp/pf61.log 2>&1 &
sleep 5 && ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.ssh/id_ed25519 -p 2242 -l bazina localhost "echo ok"

# Tail job logs
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod logs steer-strong4 -p dlab-bazina --tail 20

# Check steer-strong4 k=20 progress (how many results written)
ssh -i ~/.ssh/id_ed25519 -p 2242 -o StrictHostKeyChecking=no bazina@localhost \
  "wc -l full_trait_output/trait_steering_attack_jbb_strong_k20/trait/results.jsonl"
# Total expected: 11200 (100 behaviors × 7 alphas × 16 pairs)

# Read sweep results ranked by test AUC
ssh -i ~/.ssh/id_ed25519 -p 2242 -o StrictHostKeyChecking=no bazina@localhost \
"python3 -c \"
import json
d = json.load(open('full_trait_output/all_traits_sweep_v2_olmo3_all_traits/results.json'))
rows = [(v.get('human_test',{}).get('auc',{}).get('mean',0), v.get('val_auc',{}).get('mean',0), m,
         {k: v[k]['auc']['mean'] for k in ['PAIR','PAP','GPTFuzz','PEZ'] if k in v})
        for m, v in d['summary'].items()]
rows.sort(reverse=True)
for rank, (t, val, m, tr) in enumerate(rows[:5], 1):
    print(rank, m, f'val={val:.4f}', f'test={t:.4f}', tr)
\""

# Read steering attack summary
ssh -i ~/.ssh/id_ed25519 -p 2242 -o StrictHostKeyChecking=no bazina@localhost \
"python3 -c \"
import json
d = json.load(open('full_trait_output/trait_steering_attack_jbb_strong_k10/trait/summary.json'))
pp = d['gpt41mini']['per_pair_asr']
pb = d['gpt41mini']['per_behavior_asr']
jb = d['jbb_rubric']['per_pair_asr']
print(f'per_pair={pp:.4f} per_behavior={pb:.4f} jbb={jb:.4f}')
\""

# Check if GCG completed (JBB_GCG dir appears only after all 100 behaviors done)
ssh -i ~/.ssh/id_ed25519 -p 2242 -o StrictHostKeyChecking=no bazina@localhost \
  "ls /dlabscratch1/bazina/HarmBench/results/JBB_GCG/ 2>/dev/null && echo 'GCG done'"

# Run harmbench GCG OLMo-3 merge (after all 4 chunks succeed)
ssh -i ~/.ssh/id_ed25519 -p 2242 -o StrictHostKeyChecking=no bazina@localhost \
  "bash /mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/run_harmbench_gcg_olmo3_merge.sh"

# Check file-bridge61 status (resume if Suspended)
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod list jobs -p dlab-bazina 2>&1 | grep "file-bridge61"
```

---

## 11. Current Results Summary (as of 2026-05-28)

### Jailbreak detection — Llama-3.1-8B (test AUC / transfer AUC mean)
| Approach | Val AUC | Test AUC | PAIR | PAP | GPTFuzz | PEZ |
|----------|---------|----------|------|-----|---------|-----|
| Trait sweep (best-by-test) | 0.902 | 0.760 | — | — | — | — |
| PCA3 sweep | — | ~0.74 | — | — | — | — |

### Jailbreak detection — OLMo-3-7B (test AUC / transfer AUC)
| Approach | Val AUC | Test AUC | PAIR | PAP | GPTFuzz | PEZ |
|----------|---------|----------|------|-----|---------|-----|
| All-traits sweep (best-by-test) | 0.905 | 0.828 | 0.668 | 0.545 | 0.592 | 0.808 |
| All-traits sweep (best-transfer) | 0.956 | 0.812 | 0.681 | 0.597 | 0.579 | 0.863 |
| PCA3 sweep (best-by-test) | 0.942 | **0.868** | 0.662 | 0.610 | 0.585 | 0.817 |
| Llama Guard (input+output) | — | 0.910 | 0.826 | 0.793 | 0.761 | 0.953 |
| Self-exam | — | TBD | — | — | — | — |

GCG AUC is invalid for OLMo-3 (falls back to Llama paths). Will be valid after hb-gcg-olmo3 jobs complete.

### Steering attack — Llama-3.1-8B JBB (alpha 0.1–0.7, per-pair ASR via GPT-4.1-mini)
| Mode / Config | GPT-4.1-mini per-pair | GPT-4.1-mini per-behavior | JBB rubric |
|---------------|----------------------|--------------------------|------------|
| steer-wjb-vanilla2 (trait, no template) | 56.5% | — | 14.7% |
| steer-wjb-adv2 (adv trait) | 56.3% | — | 14.3% |
| steer-wjb-strong (alpha 0.5–3.0) | 36.9% | — | 9.2% |
| steer-wjb-wjbclf3 (WJB clf in-loop) | 44.6% | — | 8.6% |
| steer-strong4 k=10 (JBB) | **72.6%** | **100%** | 2.0% |
| steer-strong4 k=20 (JBB) | TBD | — | — |
| steer-strong4 k=30 (JBB) | TBD | — | — |

### OLMo-3 attack ASR on HarmBench (via GPT-4.1-mini)
PAIR: 16.2% | PAP: 9.6% | GPTFuzz: 58.5% | PEZ: 7.2%
