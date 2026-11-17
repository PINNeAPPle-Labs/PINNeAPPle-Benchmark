#!/usr/bin/env bash
# Experiment 10, one command: the multi-fidelity learning curve on the capsule-aeroheating corpus.
#   run: every scalar target and the heat-flux field, at the budgets and draws fixed in src/run.py
#        -> results/curve.json (medians and spreads over the draws, plus the protocol)
#
#   bash reproduce.sh            # downloads 77 MB, then ~2 min on a laptop CPU
#   bash reproduce.sh smoke      # 16 expensive runs, 2 draws: a pipeline check in seconds
#
# Needs: the PINNeAPPle checkout at the commit in env/PINNEAPPLE_COMMIT, next to this repository (or in
# $PINNEAPPLE_REPO), and `pip install bijective-gamut` for the .gamut reader. The 77 MB file is fetched once into data/.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
if [ "${1:-}" = "smoke" ]; then
  python src/run.py --smoke --out results/curve_smoke.json
else
  python src/run.py
fi
