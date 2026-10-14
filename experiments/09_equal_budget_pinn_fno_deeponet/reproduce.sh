#!/usr/bin/env bash
# Experiment 09, one command, through the library's benchmark (pp.benchmark):
#   prepare: reference data with its measured cost (the library's solvers), the airfoil OpenFOAM campaign and OOD set,
#            hashes, solver verification, data checks and the validation stack -> results/prepared/, results/validation/
#   run:     correctness gates, search, final runs at 3 budget levels, equal-FLOPs arm, airfoil budget study and setup
#            audit, determinism spot-check, report -> results/table.md, table.csv, summary.json, figures
#
#   bash reproduce.sh                 # full run (~10 h on a 4-core CPU without a GPU, plus ~2 h of OpenFOAM)
#   bash reproduce.sh smoke           # tiny sizes and budgets, a pipeline check in minutes (results_smoke/)
#
# Needs: the PINNeAPPle library checked out next to this repository (../PINNeAPPle relative to the benchmark root) at
# the commit in env/PINNEAPPLE_COMMIT, the Python packages in env/requirements.lock, and OpenFOAM v1912 (Ubuntu 24.04:
# `apt-get install openfoam`) for the airfoil. Finished jobs are cached in results/runs/: re-running resumes.
set -euo pipefail
cd "$(dirname "$0")/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
if [ "${1:-}" = "smoke" ]; then
  python -c "import common, pinneapple as pp; from bench09 import CONFIG; pp.benchmark.run(CONFIG, common.ROOT / 'results_smoke', smoke=True, force=True)"
else
  python bench09.py prepare
  python bench09.py run
fi
