"""One benchmark job = one training run under one ComputeBudget, run in its own single-threaded process.

A job spec is a dict: {"id", "problem", "method", "cfg", "seed", "budget": {wall_s|flops|steps}, "stage", ...}.
The result (or the traceback, if it failed) goes to results/runs/<id>.json with the spec and the hardware line.
"""
from __future__ import annotations

import time
import traceback

from common import RUNS, load_json, save_json, single_thread

STD = ("burgers", "darcy", "navier_stokes")


def run_job(spec: dict) -> str:
    single_thread()
    from pinneapple_analysis.cost import ComputeBudget, hardware_record
    path = RUNS / f"{spec['id']}.json"
    if path.exists():
        return spec["id"] + " (cached)"
    budget = ComputeBudget(**spec["budget"])
    t0 = time.time()
    try:
        p, m, cfg, seed = spec["problem"], spec["method"], spec["cfg"], spec["seed"]
        final = spec["stage"] != "search"
        if p in STD:
            import methods_std as M
            if m == "pinn":
                res = M.train_pinn(p, cfg, seed, budget, split=spec["split"], instance=spec["instance"])
            else:
                res = M.train_operator(p, m, cfg, seed, budget,
                                       eval_splits=("val", "test", "ood") if final else ("val",))
        elif p == "airfoil":
            import airfoil as A
            if m in ("pinn", "pinn_rans"):
                res = A.train_pinn(cfg, seed, budget, case=spec["case"], kind="rans" if m == "pinn_rans" else "potential",
                                   ckpt=str(RUNS / f"{spec['id']}.ckpt.pt"))
            elif m == "mlp":
                res = A.train_mlp(cfg, seed, budget, final=final)
            elif m == "mlp_fields":
                res = A.train_mlp_fields(cfg, seed, budget, final=final)
            else:
                res = A.train_mgn(cfg, seed, budget, final=final)
        elif p == "inverse_heat":
            import inverse_heat as H
            if m == "pinn":
                res = H.pinn_job(cfg, seed, budget, kind=spec["kind"], case=spec["case"])
            elif m == "classical":
                res = H.classical_job(seed)
            else:
                res = H.surrogate_job(m, cfg, seed, budget, final=final)
        else:
            raise ValueError(p)
        out = {"spec": spec, "result": res, "ok": True}
    except Exception:                                             # noqa: BLE001 - failures are reported, not hidden
        out = {"spec": spec, "ok": False, "error": traceback.format_exc()}
    out.update(hardware=hardware_record(), job_wall_s=time.time() - t0, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    save_json(path, out)
    return spec["id"] + (" ok" if out["ok"] else " FAILED")


def result(job_id: str):
    p = RUNS / f"{job_id}.json"
    return load_json(p) if p.exists() else None
