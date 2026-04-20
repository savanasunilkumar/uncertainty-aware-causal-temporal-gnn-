#!/usr/bin/env bash
# One-shot setup + smoke-test script for ISU ECpE research-linux*.ece.iastate.edu.
#
# What it does (idempotent):
#   1. Creates ~/uactgnn as workspace (override with UACTGNN_HOME=...)
#   2. Clones or fast-forwards the repo there
#   3. Builds a Python 3.10/3.11/3.12 venv at ~/uactgnn/.venv
#   4. Installs CPU-only PyTorch + torch-geometric + project deps
#   5. Runs example_usage.py as a smoke test
#
# Usage (after `ssh netid@research-linux3.ece.iastate.edu`):
#     curl -fsSL https://raw.githubusercontent.com/savanasunilkumar/uncertainty-aware-causal-temporal-gnn-/main/scripts/run_on_research_linux.sh | bash
#   OR, if the repo is already cloned:
#     bash ~/uactgnn/uncertainty-aware-causal-temporal-gnn-/scripts/run_on_research_linux.sh
#
# The script is deliberately read-only outside of $UACTGNN_HOME and never uses sudo.

set -euo pipefail

REPO_URL="https://github.com/savanasunilkumar/uncertainty-aware-causal-temporal-gnn-.git"
REPO_DIR_NAME="uncertainty-aware-causal-temporal-gnn-"
UACTGNN_HOME="${UACTGNN_HOME:-$HOME/uactgnn}"
VENV="$UACTGNN_HOME/.venv"
REPO="$UACTGNN_HOME/$REPO_DIR_NAME"

log()  { printf '\033[1;34m[uactgnn]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[uactgnn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[uactgnn]\033[0m %s\n' "$*" >&2; exit 1; }

# --- Host check (warn only; script still runs anywhere) -------------------------
case "$(hostname -f 2>/dev/null || hostname)" in
  research-linux*.ece.iastate.edu|class-linux*.ece.iastate.edu) ;;
  *) warn "Not on an ECpE research/class linux host; continuing anyway." ;;
esac

# --- Find a usable Python -------------------------------------------------------
find_python() {
  local candidates=(python3.12 python3.11 python3.10 python3.9 python3.13 python3)
  for p in "${candidates[@]}"; do
    if command -v "$p" >/dev/null 2>&1; then
      # Accept any Python >= 3.9. 3.12 is preferred (ordered first) because
      # PyTorch stable wheels target it; 3.13 works too on recent torch.
      if "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
        echo "$p"
        return 0
      fi
    fi
  done
  return 1
}

PY="$(find_python || true)"
if [[ -z "${PY:-}" ]]; then
  warn "No Python 3.9+ on PATH. Trying the 'module' system..."
  if command -v module >/dev/null 2>&1; then
    # Best-effort: try common module names. The user can pre-load their own.
    for m in python/3.11 python/3.10 python/3.12 python3 anaconda3 miniconda3; do
      module load "$m" 2>/dev/null && break || true
    done
    PY="$(find_python || true)"
  fi
fi

# Fallback 1: reuse an already-installed Miniconda in the user's home.
if [[ -z "${PY:-}" && -x "$HOME/miniconda3/bin/python" ]]; then
  log "Activating existing Miniconda at $HOME/miniconda3"
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/bin/activate"
  PY="$(find_python || true)"
fi

# Fallback 2: auto-install Miniconda into $HOME/miniconda3 (no sudo).
# Skip with UACTGNN_NO_MINICONDA=1.
if [[ -z "${PY:-}" && -z "${UACTGNN_NO_MINICONDA:-}" ]]; then
  log "No system Python 3.9+; installing Miniconda into \$HOME/miniconda3"
  MC_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
  MC_SH="$(mktemp --suffix=.sh)"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$MC_URL" -o "$MC_SH"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "$MC_URL" -O "$MC_SH"
  else
    die "Neither curl nor wget available; cannot install Miniconda."
  fi
  bash "$MC_SH" -b -p "$HOME/miniconda3"
  rm -f "$MC_SH"
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/bin/activate"
  PY="$(find_python || true)"
fi

[[ -n "${PY:-}" ]] || die "Could not find Python 3.9+. Either install one, 'module load <python>', or re-run without UACTGNN_NO_MINICONDA to auto-install Miniconda."
log "Using Python: $PY ($($PY -V))"

# --- Workspace + clone ----------------------------------------------------------
mkdir -p "$UACTGNN_HOME"
if [[ -d "$REPO/.git" ]]; then
  log "Updating existing clone at $REPO"
  git -C "$REPO" fetch --all --prune
  git -C "$REPO" checkout main
  git -C "$REPO" pull --ff-only
else
  log "Cloning $REPO_URL into $REPO"
  git clone "$REPO_URL" "$REPO"
fi

# --- Virtualenv -----------------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
  log "Creating venv at $VENV"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel >/dev/null

# --- PyTorch (CPU wheel) + torch-geometric -------------------------------------
if ! python -c 'import torch' 2>/dev/null; then
  log "Installing torch (CPU wheel)"
  python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.0,<2.8'
fi
if ! python -c 'import torch_geometric' 2>/dev/null; then
  log "Installing torch_geometric"
  python -m pip install 'torch-geometric>=2.3'
fi

# --- Project requirements -------------------------------------------------------
log "Installing project requirements"
# Filter torch/torch-geometric out of requirements.txt so we don't clobber the
# CPU wheel above with a CUDA one.
grep -viE '^(torch|torch-geometric)\b' "$REPO/requirements.txt" > /tmp/uactgnn_reqs.txt
python -m pip install -r /tmp/uactgnn_reqs.txt

# --- Sanity: imports ------------------------------------------------------------
log "Verifying imports"
python - <<'PY'
import torch, torch_geometric, numpy, pandas, scipy, sklearn
from causal_gnn.config import Config
from causal_gnn.training.trainer import RecommendationSystem
print(f"torch={torch.__version__} pyg={torch_geometric.__version__} cuda={torch.cuda.is_available()}")
PY

# --- Smoke test: example_usage.py on synthetic data ----------------------------
log "Running example_usage.py (synthetic data, CPU, ~1 min)"
( cd "$REPO" && python example_usage.py ) | tail -n 40

cat <<EOF

[uactgnn] Setup complete.

Workspace : $UACTGNN_HOME
Repo      : $REPO
Venv      : $VENV

Next steps (activate the venv first):
  source "$VENV/bin/activate"
  cd "$REPO"

  # Fast benchmark on MovieLens-100K (~10-30 min on CPU):
  python benchmark_m3.py

  # Full MovieLens-100K training run:
  python train_movielens.py --dataset ml-100k

  # Ablation study:
  python run_ablation_study.py

Tip: the servers are a shared resource. Keep CPU-heavy runs below ~8 threads:
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
EOF
