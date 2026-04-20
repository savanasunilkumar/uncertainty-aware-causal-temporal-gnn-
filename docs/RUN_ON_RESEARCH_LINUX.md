# Running UACT-GNN on ISU ECpE research-linux servers

Target hosts: `research-linux1…N.ece.iastate.edu` (see
<https://etg.ece.iastate.edu/remote/>). These are shared Linux servers;
pick **one** host and stick to it for your whole session — ETG will
terminate sessions that fan out across servers.

> Do **not** use `class-linux1` for research (it reboots every Sunday 3 AM
> and is explicitly scoped to coursework).

---

## 1. Get on the network

Off-campus: connect to ISU VPN first.

- Installer + docs: <https://it.iastate.edu/services/vpn>
- CLI client: Cisco AnyConnect / Cisco Secure Client
  ```
  sudo /opt/cisco/secureclient/bin/vpn connect vpn.iastate.edu
  ```
  Log in with **Net-ID** + password, approve the Okta/Duo push.

On-campus (ISU wired or `eduroam`): VPN is not required.

---

## 2. SSH in

```bash
ssh <netid>@research-linux3.ece.iastate.edu
```

Replace `3` with whichever server is least loaded — check
`uptime` / `top` right after login.

If this is your first time on that host, accept the host key fingerprint
shown by SSH. The password is your Net-ID password.

---

## 3. One-shot setup + smoke test

From your SSH session on the research-linux host:

```bash
curl -fsSL https://raw.githubusercontent.com/savanasunilkumar/uncertainty-aware-causal-temporal-gnn-/main/scripts/run_on_research_linux.sh | bash
```

What this does (idempotent, ~5–10 min first run, under a minute on reruns):

1. Creates `~/uactgnn/` as workspace (override with `UACTGNN_HOME=...`).
2. Clones the repo into `~/uactgnn/uncertainty-aware-causal-temporal-gnn-`.
3. Finds a Python 3.9–3.12; if nothing is on `PATH`, tries `module load python/…`.
4. Builds a venv at `~/uactgnn/.venv`.
5. Installs CPU-only `torch`, `torch-geometric`, and the rest of `requirements.txt`.
6. Runs `example_usage.py` on synthetic data as a smoke test.

If `module load python` fails in step 3, do it manually first:

```bash
module avail python            # see what's installed
module load python/3.11        # or whichever version is listed
```

and rerun the script.

---

## 4. Run the real benchmarks

Activate the venv once per shell:

```bash
source ~/uactgnn/.venv/bin/activate
cd ~/uactgnn/uncertainty-aware-causal-temporal-gnn-

# Be a good citizen on a shared box:
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# MovieLens-100K, all 5 models (UACT-GNN + 4 baselines):
python benchmark_m3.py

# Single model end-to-end training on MovieLens-100K:
python train_movielens.py --dataset ml-100k

# Ablation across 5 variants:
python run_ablation_study.py
```

MovieLens-100K is auto-downloaded on first run (`~5 MB`). Larger
MovieLens variants are not recommended on CPU — they will take hours.

Outputs land under:
- `outputs/` — metrics JSON, plots
- `checkpoints/` — best-k model checkpoints
- `logs/` — training logs

---

## 5. GPU on these servers

`research-linux*` nodes are **not guaranteed to have GPUs**. Confirm with:

```bash
nvidia-smi
```

If you do see a GPU and want to use it, reinstall PyTorch with the
matching CUDA wheel into the existing venv **before** you run anything:

```bash
source ~/uactgnn/.venv/bin/activate
pip uninstall -y torch
pip install --index-url https://download.pytorch.org/whl/cu121 'torch>=2.0,<2.8'
```

then pass `--device cuda` (or set `config.device='cuda'` in your script).

For serious GPU work, ETG recommends VDI or a lab GPU box — ask
<etg@iastate.edu> for a VM if needed.

---

## 6. Disk / quota tips

Home directory quotas are small. If `pip install` fails with
`No space left on device`, redirect the workspace to a larger volume:

```bash
export UACTGNN_HOME=/work/<netid>/uactgnn    # or /scratch/<netid>/uactgnn
bash ~/uactgnn/uncertainty-aware-causal-temporal-gnn-/scripts/run_on_research_linux.sh
```

(Check what's actually mounted with `df -h ~ /work /scratch` before
picking one.)

Also bias `pip` to a cache on the same volume:

```bash
export PIP_CACHE_DIR="$UACTGNN_HOME/.pip-cache"
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ssh: Connection refused` / `timed out` | Not on ISU VPN or wrong hostname. Verify with `ping research-linux3.ece.iastate.edu`. |
| `No Python 3.9+` | `module avail python` → `module load python/3.11` → rerun. |
| `ModuleNotFoundError: torch_geometric` | Venv not activated. `source ~/uactgnn/.venv/bin/activate`. |
| `OOM` / killed by cgroup | Shrink `config.batch_size` and/or `config.embedding_dim`; drop back to `ml-100k`. |
| `Weights only load failed` | Already fixed in `main` (`torch.load(..., weights_only=False)`). Pull latest. |
| ImportError for `causal-learn` when `causal_method='pc'` | This is intentional (see PR #1). Use `causal_method='advanced'` or `pip install causal-learn`. |
| Script silently does nothing | It's `set -euo pipefail`; re-run with `bash -x scripts/run_on_research_linux.sh` to see the failing line. |

---

## 8. Cleanup

```bash
deactivate 2>/dev/null || true
rm -rf ~/uactgnn             # nukes venv, repo clone, and outputs
```

Do this when you're done — home directories on shared servers fill up fast.
