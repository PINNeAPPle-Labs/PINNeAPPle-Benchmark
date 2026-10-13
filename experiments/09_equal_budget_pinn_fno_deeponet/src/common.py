"""Shared paths, problem definitions and helpers of experiment 09 (equal-budget PINN vs FNO vs DeepONet vs GNN)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT.parents[2] / "PINNeAPPle"
sys.path.insert(0, str(LIB))

SMOKE = os.environ.get("EXP09_SMOKE") == "1"
DATA = ROOT / ("data_smoke" if SMOKE else "data")
RES = ROOT / ("results_smoke" if SMOKE else "results")
RUNS = RES / "runs"
for d in (DATA, RES, RUNS):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------- the protocol
# Every number below is fixed before any run (see README "Protocol"). Budgets are wall-clock seconds on ONE CPU
# core (torch.set_num_threads(1)); four such runs execute concurrently on the 4-core machine, for every method alike.
SEEDS = (0, 1, 2)
N_TRIALS = 4                      # hyper-parameter configurations tried per method (same for every method)
SEARCH_FRACTION = 1 / 3           # each search trial gets this fraction of the full budget
BUDGET_LEVELS = (1 / 9, 1 / 3, 1.0)

PROBLEMS = {
    "burgers": dict(budget_s=300.0, n_train=1000, n_val=100, n_test=200, n_ood=100),
    "darcy": dict(budget_s=300.0, n_train=1000, n_val=100, n_test=200, n_ood=100),
    "navier_stokes": dict(budget_s=300.0, n_train=1000, n_val=100, n_test=200, n_ood=100),
}
if SMOKE:
    for p in PROBLEMS.values():
        p.update(budget_s=6.0, n_train=40, n_val=8, n_test=8, n_ood=4)

# per-instance PINNs: which test instances they are run on (the operators are evaluated on all of them; the
# table also reports the operators on exactly these instances, so the comparison is like for like)
PINN_TEST_IDS = (0, 1, 2)
PINN_OOD_IDS = (0, 1)
PINN_VAL_ID = 0                   # validation instance used by the PINN's hyper-parameter search


def save_json(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_default))
    tmp.replace(path)


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def single_thread():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
