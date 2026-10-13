#!/usr/bin/env bash
# Experiment 09, one command: data (with its cost), verification, search, final runs, equal-FLOPs arm, tables, figures.
#
#   bash reproduce.sh            # full run (~7 h on a 4-core CPU, plus ~2 h for the OpenFOAM campaign)
#   EXP09_SMOKE=1 bash reproduce.sh   # tiny sizes and budgets, a few minutes: checks the pipeline only
#
# Needs: the PINNeAPPle library checked out next to this repository (../PINNeAPPle relative to the benchmark root,
# at the commit in env/PINNEAPPLE_COMMIT), the Python packages in env/requirements.lock, and OpenFOAM v1912
# (Ubuntu 24.04: `apt-get install openfoam`, which provides /usr/share/openfoam/etc/bashrc) for the airfoil runs.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
LIB="$HERE/../../../PINNeAPPle"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
DATA=data; [ "${EXP09_SMOKE:-0}" = "1" ] && DATA=data_smoke
mkdir -p "$DATA"

# 1. airfoil: the aero app's OpenFOAM campaign (436 runs; re-running skips finished runs)
if [ "${EXP09_SMOKE:-0}" = "1" ]; then
  [ -e "$DATA/airfoil" ] || ln -s ../data/airfoil "$DATA/airfoil"   # smoke reuses (part of) the real campaign
else
  python "$LIB/apps/aero_optimizer/tools/generate_dataset.py" "$DATA/airfoil" --shapes 100 --alphas 4 --workers 4 \
    | tee "$DATA/airfoil/campaign.log"
fi

cd src
# 2. reference data for the standard problems and the inverse heat surrogate, with measured generation cost
[ "${EXP09_SMOKE:-0}" = "1" ] || python gen_data.py verify
for p in burgers darcy navier_stokes; do python gen_data.py "$p"; done
python inverse_heat.py
# 3. search, final runs at three budget levels, equal-FLOPs arm
python run_all.py all
# 4. airfoil additions: OOD OpenFOAM cases, RANS PINN, POD+MLP fields, OOD evaluation, PINN budget study, audit
if [ "${EXP09_SMOKE:-0}" = "1" ]; then python run_all.py airfoil2; else bash ../airfoil2.sh; fi
# 5. the library's benchmark on the same artifacts: prepare (hashes, solver verification, data checks, validation
#    stack), correctness gates, determinism spot-check, tables, figures, diagnostics (runs are reused from the cache)
#    (full mode only: the smoke sizes are not the library's; the library's own CI runs its smoke benchmark)
if [ "${EXP09_SMOKE:-0}" != "1" ]; then python bench09.py prepare; python bench09.py run; fi
# 6. this experiment's own figures and airfoil tables
python report.py
python report_airfoil.py
