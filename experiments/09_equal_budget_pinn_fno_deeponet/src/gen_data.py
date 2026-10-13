"""Reference data for the three standard problems, with timing (the data-generation cost) and verification.

    python gen_data.py verify            -> results/verification.json
    python gen_data.py burgers|darcy|navier_stokes   -> data/<problem>.npz + results/datagen_<problem>.json

Solvers: pinneapple_simulation.numerical_solvers.operator_benchmarks (burgers_1d, darcy_2d,
navier_stokes_2d_vorticity, periodic_grf). Each sample is solved single-threaded; samples are spread over 4 worker
processes; the cost reported is the sum of per-sample solve times (core-seconds), measured, not estimated.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import DATA, PROBLEMS, RES, SMOKE, log, save_json, single_thread
from pinneapple_simulation.numerical_solvers.operator_benchmarks import (burgers_1d, darcy_2d, li_forcing,
                                                                         navier_stokes_2d_vorticity, periodic_grf,
                                                                         threshold_coefficient)

# --------------------------------------------------------------------------------------------- problem settings
BURGERS = dict(nu=0.01, t_end=1.0, n_solve=1024, n_out=256, dt=1e-4, alpha=2.0, tau=5.0, sigma=25.0, sigma_ood=50.0)
DARCY = dict(n_solve=128, n_out=64, alpha=2.0, tau=3.0, tau_ood=6.0, high=12.0, low=3.0)
NS = dict(nu=1e-3, t_end=10.0, n=64, dt=5e-3, alpha=2.5, tau=7.0, tau_ood=14.0)
SPLIT_SEEDS = {"train": 11, "val": 12, "test": 13, "ood": 14}
CHUNK = 25


def _burgers_chunk(args):
    single_thread()
    u0 = args
    t0 = time.process_time()
    _, u = burgers_1d(u0, BURGERS["nu"], BURGERS["t_end"], dt=BURGERS["dt"])
    dt = time.process_time() - t0
    s = BURGERS["n_solve"] // BURGERS["n_out"]
    return u[:, 0, ::s].astype(np.float32), u[:, -1, ::s].astype(np.float32), dt


def _darcy_chunk(args):
    single_thread()
    fields = args
    a = threshold_coefficient(fields, DARCY["high"], DARCY["low"])
    t0 = time.process_time()
    u = np.stack([darcy_2d(ai) for ai in a])
    dt = time.process_time() - t0
    m = DARCY["n_solve"] // DARCY["n_out"]
    pool = lambda z: z.reshape(len(z), DARCY["n_out"], m, DARCY["n_out"], m).mean(axis=(2, 4))  # noqa: E731
    return pool(a).astype(np.float32), pool(u).astype(np.float32), dt


def _ns_chunk(args):
    single_thread()
    w0 = args
    t0 = time.process_time()
    _, w = navier_stokes_2d_vorticity(w0, NS["nu"], NS["t_end"], forcing=li_forcing(NS["n"]), dt=NS["dt"])
    dt = time.process_time() - t0
    return w[:, 0].astype(np.float32), w[:, -1].astype(np.float32), dt


def _inputs(problem: str, split: str, n: int) -> np.ndarray:
    seed = SPLIT_SEEDS[split]
    if problem == "burgers":
        sig = BURGERS["sigma_ood"] if split == "ood" else BURGERS["sigma"]
        return periodic_grf(n, (BURGERS["n_solve"],), alpha=BURGERS["alpha"], tau=BURGERS["tau"], sigma=sig, seed=seed)
    if problem == "darcy":
        tau = DARCY["tau_ood"] if split == "ood" else DARCY["tau"]
        return periodic_grf(n, (DARCY["n_solve"],) * 2, alpha=DARCY["alpha"], tau=tau, seed=seed)
    tau = NS["tau_ood"] if split == "ood" else NS["tau"]
    return periodic_grf(n, (NS["n"],) * 2, alpha=NS["alpha"], tau=tau, seed=seed)


def generate(problem: str) -> None:
    cfg = PROBLEMS[problem]
    fn = {"burgers": _burgers_chunk, "darcy": _darcy_chunk, "navier_stokes": _ns_chunk}[problem]
    out, cost = {}, {}
    wall0 = time.time()
    with ProcessPoolExecutor(4) as ex:
        for split in ("train", "val", "test", "ood"):
            n = cfg[f"n_{split}"]
            X = _inputs(problem, split, n)
            chunks = [X[i:i + CHUNK] for i in range(0, n, CHUNK)]
            res = list(ex.map(fn, chunks))
            out[f"{split}_x"] = np.concatenate([r[0] for r in res])
            out[f"{split}_y"] = np.concatenate([r[1] for r in res])
            cost[split] = {"n": n, "solver_core_s": float(sum(r[2] for r in res))}
            log(problem, split, n, f"{cost[split]['solver_core_s']:.1f} core-s")
    np.savez_compressed(DATA / f"{problem}.npz", **out)
    total = sum(c["solver_core_s"] for c in cost.values())
    n_all = sum(c["n"] for c in cost.values())
    save_json(RES / f"datagen_{problem}.json", {
        "problem": problem, "settings": {"burgers": BURGERS, "darcy": DARCY, "navier_stokes": NS}[problem],
        "splits": cost, "solver_core_s_total": total, "solver_core_s_per_sample": total / n_all,
        "train_set_core_s": cost["train"]["solver_core_s"] + cost["val"]["solver_core_s"],
        "wall_s_4_workers": time.time() - wall0,
        "timing": "time.process_time() of the solver call, summed over samples (CPU core-seconds, one thread)"})


# --------------------------------------------------------------------------------------------- verification
def verify() -> None:
    single_thread()
    out = {}
    # Burgers: time step and resolution convergence on 3 samples; and why nu = 0.01, not the FNO paper's 0.1
    u0 = periodic_grf(3, (1024,), alpha=2.0, tau=5.0, sigma=25.0, seed=99)
    _, ua = burgers_1d(u0, 0.01, 1.0, dt=1e-4)
    _, ub = burgers_1d(u0, 0.01, 1.0, dt=5e-5)
    u0f = np.fft.irfft(np.fft.rfft(u0), n=2048) * 2
    _, uc = burgers_1d(u0f, 0.01, 1.0, dt=5e-5)
    _, ud = burgers_1d(u0, 0.1, 1.0, dt=1e-4)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))  # noqa: E731
    out["burgers"] = {"rel_dt_halved": rel(ua[:, -1], ub[:, -1]), "rel_grid_doubled": rel(ub[:, -1], uc[:, -1, ::2]),
                      "max_abs_u0": float(np.abs(u0).max()), "max_abs_u1_nu0.01": float(np.abs(ua[:, -1]).max()),
                      "max_abs_u1_nu0.1": float(np.abs(ud[:, -1]).max())}
    log("burgers", out["burgers"])
    # Darcy: 128 vs 256 solve, both averaged to 64
    f = periodic_grf(3, (256, 256), alpha=2.0, tau=3.0, seed=98)
    a256 = threshold_coefficient(f)
    a128 = a256.reshape(3, 128, 2, 128, 2).mean(axis=(2, 4))
    u256 = np.stack([darcy_2d(a) for a in a256]).reshape(3, 64, 4, 64, 4).mean(axis=(2, 4))
    u128 = np.stack([darcy_2d(a) for a in threshold_coefficient(periodic_grf(3, (128, 128), alpha=2.0, tau=3.0, seed=98))])
    u128c = np.stack([darcy_2d(a) for a in a128]).reshape(3, 64, 2, 64, 2).mean(axis=(2, 4))
    out["darcy"] = {"rel_grid_doubled_same_coefficient_pooled": rel(u128c, u256),
                    "note": "a on 256^2 thresholded, pooled to 128^2 for the coarse solve; both solutions pooled to 64^2",
                    "u_max": float(u128.max())}
    log("darcy", out["darcy"])
    # Navier-Stokes: 64 vs 128 grid, dt halved
    w0 = periodic_grf(3, (128, 128), alpha=2.5, tau=7.0, seed=97)
    _, w128 = navier_stokes_2d_vorticity(w0, 1e-3, 10.0, forcing=li_forcing(128), dt=2.5e-3)
    _, w64 = navier_stokes_2d_vorticity(w0[:, ::2, ::2], 1e-3, 10.0, forcing=li_forcing(64), dt=5e-3)
    _, w64h = navier_stokes_2d_vorticity(w0[:, ::2, ::2], 1e-3, 10.0, forcing=li_forcing(64), dt=2.5e-3)
    out["navier_stokes"] = {"rel_grid_doubled": rel(w64[:, -1], w128[:, -1, ::2, ::2]),
                            "rel_dt_halved": rel(w64[:, -1], w64h[:, -1])}
    log("navier_stokes", out["navier_stokes"])
    save_json(RES / "verification.json", out)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "verify":
        verify()
    else:
        generate(what)
