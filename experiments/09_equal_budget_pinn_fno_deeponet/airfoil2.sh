#!/usr/bin/env bash
# Stage 4 (airfoil additions): OOD OpenFOAM runs, then RANS PINN, POD+MLP fields, OOD evaluation, budget study, audit.
set -euo pipefail
cd "$(dirname "$0")/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python gen_airfoil_ood.py > ../results/gen_airfoil_ood.log 2>&1
# the first MLP / MeshGraphNet full-budget airfoil runs were evaluated before the OOD set existed: kept, superseded
# (once: a re-run after an interruption must not move the new runs)
if [ ! -e ../results/superseded_no_ood ]; then
mkdir -p ../results/superseded_no_ood
for f in ../results/runs/final__airfoil__mlp__L1.000__s*.json ../results/runs/final__airfoil__mgn__L1.000__s*.json; do
  [ -e "$f" ] && mv "$f" ../results/superseded_no_ood/
done
fi
python run_all.py airfoil2 > ../results/run_airfoil2.log 2>&1
touch ../results/AIRFOIL2_DONE
