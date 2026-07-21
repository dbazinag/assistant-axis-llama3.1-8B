# RCP Cluster Usage Guide

This document explains how to use the EPFL RCP (Research Computing Platform) cluster from this
environment. Written for LLM assistants operating in this workspace.

---

## Environment

- **Platform**: EPFL RCP cluster (RunAI scheduler)
- **Project**: `dlab-bazina`
- **User**: `dominic.bazina-grolinger@epfl.ch` (GASPAR: `bazina`)
- **Local machine**: WSL2 on Windows (this workspace)
- **RunAI CLI**: `runai-rcp-prod` (always prefix with `SUPPRESS_DEPRECATION_MESSAGE=true`)

---

## CLI Basics

All RunAI commands require the deprecation env var:

```bash
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod <command> -p dlab-bazina
```

Common commands:

```bash
# List all jobs
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod list jobs -p dlab-bazina

# Describe a job (get submission command, status, node)
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod describe job <name> -p dlab-bazina

# Tail logs
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod logs <name> -p dlab-bazina --tail 20

# Delete a job
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod delete job <name> -p dlab-bazina

# Resume a suspended job
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod resume job <name> -p dlab-bazina
```

---

## SSH Access via File-Bridge

The cluster nodes are not directly SSH-accessible. Use an **interactive file-bridge pod** as a
jump host via `kubectl port-forward`.

### Starting a file-bridge pod

```bash
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod submit file-bridge<N> \
  --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
  --pvc dlab-scratch:/mnt \
  --node-pools cpu \
  -p dlab-bazina \
  --interactive \
  --cpu 4 --memory 16G \
  -- sleep infinity
```

**Key points:**
- Use `-- sleep infinity` — the image already starts sshd and creates the GASPAR user automatically.
  Do NOT add a custom `bash -c "useradd ... sshd ..."` block — this double-wraps the command and
  breaks sshd (the image entrypoint does `/bin/bash -c "<your command>"`, so inner bash -c breaks quoting).
- `--node-pools cpu` — file-bridge runs on CPU nodes (no GPU needed).
- `--interactive` flag is required for interactive jobs.
- The pod will get **suspended** by the cluster after idle time. Resume it instead of creating a new one:
  ```bash
  SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod resume job file-bridge<N> -p dlab-bazina
  ```

### Setting up port-forward

Once the pod is Running:

```bash
# Get the pod name
kubectl get pods -n runai-dlab-bazina --field-selector=status.phase=Running \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep file-bridge<N>

# Start port-forward (background)
kubectl port-forward -n runai-dlab-bazina file-bridge<N>-0-0 2242:22 > /tmp/pf<N>.log 2>&1 &

# Verify
sleep 3 && cat /tmp/pf<N>.log
```

### SSH to the cluster

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i ~/.ssh/id_ed25519 -p 2242 -l bazina localhost "echo ok"
```

SSH config shortcut (in `~/.ssh/config`):
```
Host runai
    HostName localhost
    User bazina
    ForwardAgent yes
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
    Port 2222
```

Wait for the GASPAR user to be created before connecting — poll until ready:
```bash
until ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i ~/.ssh/id_ed25519 -p 2242 -l bazina localhost "echo ok" 2>/dev/null; do
  sleep 5
done
```

---

## Filesystem / Paths

The NFS scratch is mounted at `/mnt` inside pods (`--pvc dlab-scratch:/mnt`).
Inside pods, `/dlabscratch1` is a symlink to `/mnt/dlabscratch1`.

| Context | Base path |
|---|---|
| Inside pod | `/dlabscratch1/bazina/` or `/mnt/dlabscratch1/bazina/` |
| Via SSH (file-bridge) | `/dlabscratch1/bazina/` |
| This workspace (local) | `/home/dbazinag/projects/` (local, NOT on cluster) |

Project root on cluster: `/dlabscratch1/bazina/assistant-axis-llama3.1-8B/`

**Important**: Local files (this workspace) must be explicitly copied to the cluster before a job
can use them. Copy via SSH:
```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i ~/.ssh/id_ed25519 -p 2242 -l bazina localhost \
    "cat > /dlabscratch1/bazina/assistant-axis-llama3.1-8B/my_script.sh" < /local/path/my_script.sh
```

---

## Submitting GPU Training Jobs

```bash
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod submit <job-name> \
  --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
  --pvc dlab-scratch:/mnt \
  --gpu 1 \
  -p dlab-bazina \
  -- bash /dlabscratch1/bazina/assistant-axis-llama3.1-8B/run_my_script.sh
```

**Critical**: Always pass the script as an **absolute path directly** after `--`. Do NOT use
`bash -c "cd /path && bash script.sh"` — the image entrypoint wraps your command in
`/bin/bash -c "..."`, so an inner `bash -c` produces double-wrapping that breaks argument parsing.

Correct:
```
-- bash /dlabscratch1/bazina/assistant-axis-llama3.1-8B/run_script.sh
```

Wrong (double-wraps):
```
-- bash -c "cd /dlabscratch1/... && bash run_script.sh"
```

### Run scripts

Run scripts on the cluster should follow this pattern:
```bash
#!/usr/bin/env bash
set -euo pipefail
set -a
source /dlabscratch1/bazina/assistant-axis-llama3.1-8B/.env   # loads OPENAI_API_KEY etc.
set +a
cd /dlabscratch1/bazina/assistant-axis-llama3.1-8B
uv run python my_script.py --arg value
```

The `.env` file contains API keys (`OPENAI_API_KEY`, `HF_TOKEN`, etc.) — always source it.
Use `uv run python` (not `python` directly) to get the project's virtualenv.

---

## Image

All jobs use: `ghcr.io/jkminder/dlab-runai-images/pytorch:master`

This image:
- Starts sshd on boot (before running your command)
- Creates the GASPAR user (`bazina`) from LDAP, home at `/dlabscratch1/bazina`
- Wraps your `-- command` in `/bin/bash -c "command"` automatically
- Has CUDA available for GPU nodes

---

## Monitoring Jobs

```bash
# Quick status of all jobs
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod list jobs -p dlab-bazina

# Tail logs of a running job
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod logs <name> -p dlab-bazina --tail 30

# Check if a job completed successfully by grepping logs
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod logs <name> -p dlab-bazina 2>/dev/null | grep -E "Saved|Finished|Error|Traceback"
```

Job statuses:
- **Running** — active, consuming resources
- **Succeeded** — completed normally
- **Failed/Error** — exited non-zero; check logs
- **Suspended** — paused by scheduler (quota/preemption); can be resumed
- **Pending** — waiting for resources

---

## Reading Cluster Files (without SSH)

If no file-bridge is available, read output from job logs directly:
```bash
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod logs <job> -p dlab-bazina 2>/dev/null | tail -50
```

For structured output (JSON results etc.), use SSH to the file-bridge and run Python inline:
```bash
ssh ... "python3 -c \"import json; d=json.load(open('/path/results.json')); print(d['key'])\""
```

Use heredoc for multi-line Python (avoids quoting issues):
```bash
ssh ... "python3 << 'EOF'
import json
d = json.load(open('/dlabscratch1/bazina/.../results.json'))
print(d['summary']['model']['family']['auc']['mean'])
EOF"
```

---

## Cleanup

Delete finished/failed jobs regularly to keep the job list clean:
```bash
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod delete job <name> -p dlab-bazina
```

Batch delete:
```bash
for job in job1 job2 job3; do
  SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod delete job $job -p dlab-bazina 2>/dev/null \
    && echo "deleted $job" || echo "skip $job"
done
```

---

## Common Gotchas

1. **Double bash-c wrapping**: The image adds `/bin/bash -c` around your `-- command`.
   Never add an inner `bash -c`. Use absolute script paths directly.

2. **File-bridge lifetime**: Pods get suspended after cluster idle timeout. Use `resume` not
   a new submission. Use `sleep infinity` as the command.

3. **Port-forward on wrong pod**: After restarting/resuming, get the new pod name via
   `kubectl get pods` and restart port-forward.

4. **SSH auth before user creation**: The GASPAR user takes ~10s to be created on pod start.
   Poll with a loop before SSH-ing to a fresh pod.

5. **NFS path consistency**: Both `/dlabscratch1/bazina/` and `/mnt/dlabscratch1/bazina/`
   resolve to the same NFS location inside pods.

6. **No `--no-verify` or `--force`**: Don't bypass cluster safety flags.

7. **GCG writes output at the end**: HarmBench GCG (`generate_test_cases.py`) batch-saves all
   outputs when the full chunk completes — no intermediate files. Track progress via logs.
