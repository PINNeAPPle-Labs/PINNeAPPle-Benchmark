"""Experiment 09 driver: hyper-parameter search, final runs at three budget levels, equal-FLOPs arm.

    python run_all.py [search|final|flops|all]

Stage 1 (search): every method gets N_TRIALS configurations drawn by pinneapple_analysis.cost.equal_budget_search
from its grid, each trained for SEARCH_FRACTION of the full budget, scored on validation data only.
Stage 2 (final): the best configuration, seeds 0-2, at BUDGET_LEVELS x the full budget; evaluated on the test sets.
Stage 3 (flops, the three standard problems): every method trained to the same number of FLOPs, the smallest
FLOP count any method used in its full wall-clock budget (median over seeds), with a wall-clock cap of 3x.

Every job runs single-threaded; 4 run at a time (4 cores), whatever the method. Finished jobs are cached in
results/runs/, so the driver can be re-run after an interruption.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import (BUDGET_LEVELS, N_TRIALS, PINN_OOD_IDS, PINN_TEST_IDS, PINN_VAL_ID, PROBLEMS, RES, SEARCH_FRACTION,
                    SEEDS, SMOKE, log, save_json)
from jobs import STD, result, run_job
from pinneapple_analysis.cost import equal_budget_search

AIRFOIL_BUDGET = 20.0 if SMOKE else 900.0
HEAT_BUDGET = 6.0 if SMOKE else 120.0
METHODS = {**{p: ("fno", "deeponet", "pinn") for p in STD}, "airfoil": ("mlp", "mgn", "pinn"),
           "inverse_heat": ("mlp", "deeponet", "pinn")}


def budget_s(problem):
    return PROBLEMS[problem]["budget_s"] if problem in STD else AIRFOIL_BUDGET if problem == "airfoil" else HEAT_BUDGET


def spaces(problem, method):
    if problem in STD:
        import methods_std as M
        return M.space(problem, method)
    if problem == "airfoil":
        import airfoil as A
        return A.SPACES[method]
    import inverse_heat as H
    return H.SPACES[method]


def airfoil_cases():
    import airfoil as A
    S = A.splits()
    test_shapes = sorted({r["shape_id"] for r in S["test"]})[:2]
    test = [f"{s}_a{b}" for s in test_shapes for b in (1, 2)]
    val = sorted(r["id"] for r in S["val"] if r["id"].endswith("_a1"))[0]
    return {"val": val, "test": test, "naca": "NACA2412_a+04"}


def search_configs(problem, method):
    sp = spaces(problem, method)
    seen = []
    equal_budget_search(sp, lambda c: seen.append(dict(c)) or 0.0, n_trials=N_TRIALS, seed=0)
    return seen


def search_jobs():
    jobs = []
    for p, ms in METHODS.items():
        b = {"wall_s": budget_s(p) * SEARCH_FRACTION}
        for m in ms:
            for i, cfg in enumerate(search_configs(p, m)):
                spec = {"id": f"search__{p}__{m}__{i}", "problem": p, "method": m, "cfg": cfg, "seed": 0,
                        "budget": b, "stage": "search"}
                if m == "pinn":
                    if p in STD:
                        spec.update(split="val", instance=PINN_VAL_ID)
                    elif p == "airfoil":
                        spec.update(case=airfoil_cases()["val"])
                    else:
                        spec.update(kind="val", case=0)
                jobs.append(spec)
    return jobs


def search_score(res: dict, problem: str, method: str) -> float:
    if not res or not res.get("ok"):
        return float("nan")
    r = res["result"]
    if problem in STD:
        return r["val"]["rel_l2_mean"]
    if problem == "airfoil":
        v = r["result"] if method == "pinn" else r["val"]
        return v["cl_abs_err_mean"] + v["cd_rel_err_mean"]
    return r["h_rel_err"] if method == "pinn" else r["val_rel_l2"]


def best_config(problem, method):
    cfgs = search_configs(problem, method)
    scores = {i: search_score(result(f"search__{problem}__{method}__{i}"), problem, method) for i in range(len(cfgs))}
    lookup = {tuple(sorted(c.items())): scores[i] for i, c in enumerate(cfgs)}
    sr = equal_budget_search(spaces(problem, method), lambda c: lookup[tuple(sorted(c.items()))], n_trials=N_TRIALS, seed=0)
    return sr


def final_jobs():
    jobs = []
    for p, ms in METHODS.items():
        B = budget_s(p)
        for m in ms:
            cfg = best_config(p, m).best_config
            base = {"problem": p, "method": m, "cfg": cfg, "stage": "final"}
            for lv in BUDGET_LEVELS:
                b = {"wall_s": B * lv}
                for s in SEEDS:
                    tag = f"final__{p}__{m}__L{lv:.3f}__s{s}"
                    if m != "pinn":
                        jobs.append({**base, "id": tag, "seed": s, "budget": b, "level": lv})
                    elif p in STD:
                        for i in PINN_TEST_IDS:
                            jobs.append({**base, "id": f"{tag}__test{i}", "seed": s, "budget": b, "level": lv,
                                         "split": "test", "instance": i})
                        if lv == 1.0:
                            for i in PINN_OOD_IDS:
                                jobs.append({**base, "id": f"{tag}__ood{i}", "seed": s, "budget": b, "level": lv,
                                             "split": "ood", "instance": i})
                    elif p == "airfoil":
                        cases = airfoil_cases()
                        sel = cases["test"] + [cases["naca"]] if lv == 1.0 else cases["test"][:1]
                        for c in sel:
                            jobs.append({**base, "id": f"{tag}__{c}", "seed": s, "budget": b, "level": lv, "case": c})
                    else:
                        import inverse_heat as H
                        for kind, lst in H.CASES.items():
                            for i in range(len(lst)):
                                jobs.append({**base, "id": f"{tag}__{kind}{i}", "seed": s, "budget": b, "level": lv,
                                             "kind": kind, "case": i})
    for s in SEEDS:
        jobs.append({"id": f"final__inverse_heat__classical__s{s}", "problem": "inverse_heat", "method": "classical",
                     "cfg": {}, "seed": s, "budget": {"steps": 1}, "stage": "final", "level": 1.0})
    return jobs


def flops_target(problem):
    used = {}
    for m in METHODS[problem]:
        f = []
        for s in SEEDS:
            tag = f"final__{problem}__{m}__L1.000__s{s}" + ("__test0" if m == "pinn" else "")
            r = result(tag)
            if r and r.get("ok"):
                f.append(r["result"]["train"]["flops"])
        if f:
            used[m] = float(np.median(f))
    return min(used.values()), used


def flops_jobs():
    jobs, targets = [], {}
    for p in STD:
        F, used = flops_target(p)
        targets[p] = {"flops": F, "median_flops_used_in_full_wall_budget": used}
        for m in METHODS[p]:
            cfg = best_config(p, m).best_config
            for s in SEEDS:
                spec = {"id": f"flops__{p}__{m}__s{s}", "problem": p, "method": m, "cfg": cfg, "seed": s,
                        "budget": {"flops": F, "wall_s": 3 * budget_s(p)}, "stage": "flops"}
                if m == "pinn":
                    spec.update(split="test", instance=0)
                jobs.append(spec)
    save_json(RES / "flops_targets.json", targets)
    return jobs


def execute(jobs, workers=4):
    todo = [j for j in jobs if result(j["id"]) is None]
    log(f"{len(jobs)} jobs, {len(todo)} to run")
    # long jobs first so the four workers finish together
    todo.sort(key=lambda j: -max(j["budget"].get("wall_s", 0), 0))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, max_tasks_per_child=1) as ex:
        futs = [ex.submit(run_job, j) for j in todo]
        for i, f in enumerate(as_completed(futs)):
            log(f"[{i + 1}/{len(todo)}]", f.result())


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("search", "all"):
        execute(search_jobs())
        save_json(RES / "search.json", {f"{p}/{m}": best_config(p, m).to_dict() for p, ms in METHODS.items() for m in ms})
    if what in ("final", "all"):
        execute(final_jobs())
    if what in ("flops", "all"):
        execute(flops_jobs())


if __name__ == "__main__":
    main()
