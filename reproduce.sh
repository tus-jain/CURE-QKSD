#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"
export PYTHONPATH="$root/src:$root"
export MPLBACKEND=Agg
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cure-qksd-matplotlib"

bash scripts/figures/render_overview.sh
python scripts/figures/conditioning.py
python scripts/figures/threshold_trap.py
python scripts/figures/threshold_scan.py
python scripts/figures/phase_landscape.py
python scripts/figures/incommensurate.py
python scripts/figures/noiseless.py
python scripts/figures/noisy.py
python scripts/figures/hf_shift.py
python scripts/figures/m_sweep.py
python scripts/figures/table_i.py
python scripts/analysis/cure_assumptions.py
python scripts/analysis/hf_shift.py
python scripts/analysis/root_weights.py
