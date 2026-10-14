"""Experiment 09 as a configuration of the library's benchmark (``pp.benchmark``, pinneapple_analysis.benchmark).

    python bench09.py prepare     # artifacts into results/prepared/: the reference data with its measured cost, the
                                  # airfoil OpenFOAM campaign and OOD set, hashes, solver verification, validation
    python bench09.py run         # gates, search, final runs, equal-FLOPs arm, airfoil budget study and setup audit
                                  # (finished jobs in results/runs/ are reused), report
    python bench09.py report      # results/table.md, table.csv, summary.json, figures/ (and ../figures/)

Everything that is not this experiment's configuration lives in the library: the problems (`burgers`, `darcy`,
`navier_stokes`, `airfoil`, `inverse_heat`), the methods, the solvers, the gates, the validation stack and the report.
``migrate`` adopts the artifacts of the first runs of this experiment (data/, made with the same library solvers and
seeds) instead of generating them again.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from common import DATA, RES, ROOT  # noqa: F401  (common also puts the library on the path)

import pinneapple as pp  # noqa: E402

CONFIG = pp.benchmark.BenchmarkConfig(
    problems=["burgers", "darcy", "navier_stokes", "airfoil", "inverse_heat"],
    methods=["pinn", "fno", "deeponet", "mgn", "mlp", "mlp_fields", "pinn_rans"],
    budget={"wall_s": 300.0},
    problem_budgets={"airfoil": {"wall_s": 900.0}, "inverse_heat": {"wall_s": 120.0}},
    budget_levels=[1 / 9, 1 / 3, 1.0],
    seeds=3, n_trials=4, search_fraction=1 / 3,
    equal_flops=["burgers", "darcy", "navier_stokes"],
    n_workers=4, threads_per_run=1,
    problem_options={"airfoil": {"gci_study": True}},
    # the airfoil PINNs: are they budget-limited (1 h, 4 h), and is their setup the problem (learning rate, loss weights,
    # Adam -> L-BFGS for the last 30 % of the budget)?
    budget_study={"airfoil": {"pinn_rans": [3600.0, 14400.0], "pinn": [3600.0]}},
    ablations={"airfoil": {
        "pinn": {"lr1e-3": {"lr": 1e-3}, "lbfgs": {"lbfgs": 0.3},
                 "weights_wall100_kutta100": {"wall_w": 100.0, "kutta_w": 100.0},
                 "weights_wall1_kutta1": {"wall_w": 1.0, "kutta_w": 1.0}},
        "pinn_rans": {"lbfgs": {"lbfgs": 0.3}, "lr1e-3": {"lr": 1e-3}}}},
)

FIGURES = {"summary_table.png": "fig0_summary_table.png", "error_vs_budget.png": "fig1_error_vs_budget.png",
           "pareto_error_vs_inference.png": "fig2_pareto_error_vs_inference.png",
           "budget_study.png": "fig3_airfoil_pinn_budget_study.png"}


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or not src.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def migrate() -> None:
    """Adopt the artifacts of this experiment's first runs (if data/ holds them) where the prepare stage expects them."""
    P = RES / "prepared"
    for pid in ("burgers", "darcy", "navier_stokes"):
        if not (DATA / f"{pid}.npz").exists():
            continue
        _link(DATA / f"{pid}.npz", P / pid / "data.npz")
        dg = json.loads((RES / f"datagen_{pid}.json").read_text())
        (P / pid / "cost.json").write_text(json.dumps({**dg, "adopted_from": f"results/datagen_{pid}.json"}, indent=1))
    if (DATA / "inverse_heat.npz").exists():
        _link(DATA / "inverse_heat.npz", P / "inverse_heat" / "inverse_heat.npz")
        dg = json.loads((RES / "datagen_inverse_heat.json").read_text())
        (P / "inverse_heat" / "cost.json").write_text(json.dumps({**dg, "adopted_from": "results/datagen_inverse_heat.json"},
                                                                 indent=1))
        refs = [json.loads((RES / "runs" / f"final__inverse_heat__classical__s{s}.json").read_text())["result"]
                for s in range(3)]
        (P / "inverse_heat" / "reference.json").write_text(json.dumps(refs, indent=1))
    if (DATA / "airfoil" / "runs").exists():
        for src_dir, dst_dir in ((DATA / "airfoil" / "runs", P / "airfoil" / "runs"),
                                 (DATA / "airfoil_ood" / "runs", P / "airfoil" / "ood" / "runs")):
            for f in sorted(src_dir.glob("*")):
                _link(f, dst_dir / f.name)
        recs = [json.loads(f.read_text()) for f in sorted((P / "airfoil" / "runs").glob("*.json"))]
        sec = [r.get("seconds") or 0.0 for r in recs]
        clean = json.loads((RES / "openfoam_clean_timing.json").read_text())
        (P / "airfoil" / "cost.json").write_text(json.dumps({
            "n_runs": len(recs), "solver_s_total_recorded": sum(sec), "solver_s_per_run_recorded": sum(sec) / len(sec),
            "solver_s_per_run": clean["mean_wall_s"], "train_set_core_s": clean["mean_wall_s"] * len(recs),
            "n_ood_runs": len(list((P / "airfoil" / "ood" / "runs").glob("*.json"))),
            "timing": "data cost = runs x the clean per-run time (openfoam_clean_timing.json, 4 runs re-timed alone); "
                      "the campaign's recorded wall times are inflated because it shared the CPU with other jobs"}, indent=1))
        _link(RES / "openfoam_clean_timing.json", P / "airfoil" / "openfoam_clean_timing.json")


def figures() -> None:
    for src, dst in FIGURES.items():
        if (RES / "figures" / src).exists():
            shutil.copy2(RES / "figures" / src, ROOT / "figures" / dst)


def main(what: str) -> None:
    if what == "prepare":
        migrate()
        for pid, m in pp.benchmark.prepare(CONFIG, RES, adopt=True).items():
            print(pid, m["status"], m.get("blocking"))
    elif what == "run":
        res = pp.benchmark.run(CONFIG, RES, prepare_missing=False)
        for d in res.diagnostics:
            print("-", d.line())
        figures()
    elif what == "report":
        pp.benchmark.report(RES)
        figures()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run")
