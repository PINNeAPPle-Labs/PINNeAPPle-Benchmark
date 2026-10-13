"""Tables (Markdown + CSV) and figures from results/runs/*.json. Every number printed comes from a run file.

    python report.py   -> results/table.md, results/table.csv, results/summary.json, figures/*.png
"""
from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import BUDGET_LEVELS, PROBLEMS, PINN_OOD_IDS, PINN_TEST_IDS, RES, ROOT, RUNS, SEEDS, load_json, save_json  # noqa: E402
from pinneapple_analysis.cost import AmortizedCost  # noqa: E402

FIG = ROOT / ("figures_smoke" if RES.name.endswith("smoke") else "figures")
FIG.mkdir(exist_ok=True)
STD = ("burgers", "darcy", "navier_stokes")
TITLE = {"burgers": "Burgers 1-D", "darcy": "Darcy 2-D", "navier_stokes": "Navier-Stokes 2-D",
         "airfoil": "Airfoil 2-D (RANS)", "inverse_heat": "Inverse heat (h)"}
NAME = {"pinn": "PINN", "fno": "FNO", "deeponet": "DeepONet", "mgn": "MeshGraphNet", "mlp": "MLP",
        "classical": "FV + calibration", "solver": "Reference solver"}
# fixed colour per method (reference categorical palette, slots in order), and a marker as a second encoding
COLOR = {"pinn": "#2a78d6", "fno": "#eb6834", "deeponet": "#1baf7a", "mgn": "#eda100", "mlp": "#e87ba4"}
MARK = {"pinn": "o", "fno": "s", "deeponet": "^", "mgn": "D", "mlp": "v"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

RUNS_ALL = {}
for f in glob.glob(str(RUNS / "*.json")):
    r = load_json(f)
    RUNS_ALL[r["spec"]["id"]] = r


def get(job_id):
    r = RUNS_ALL.get(job_id)
    return r["result"] if r and r.get("ok") else None


def failed():
    return sorted(k for k, r in RUNS_ALL.items() if not r.get("ok"))


def ms(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None, None, 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def lvl(x):
    return f"L{x:.3f}"


def fmt(m, s=None, digits=3, pct=False):
    if m is None:
        return "-"
    k = 100.0 if pct else 1.0
    if s is None:
        return f"{m * k:.{digits}g}" + ("%" if pct else "")
    return f"{m * k:.{digits}g} ± {s * k:.2g}" + ("%" if pct else "")


def fmt_s(x):
    if x is None:
        return "-"
    if x < 1e-3:
        return f"{x * 1e6:.0f} µs"
    if x < 1:
        return f"{x * 1e3:.1f} ms"
    return f"{x:.0f} s" if x < 1e4 else f"{x / 3600:.1f} h"


def datagen(problem):
    p = RES / f"datagen_{problem}.json"
    return load_json(p) if p.exists() else None


# ------------------------------------------------------------------------------------------------ standard problems
def std_seed_metrics(problem, method, level=1.0, prefix="final"):
    """Per seed: test error on the PINN instances, on the whole test set, OOD, QoI, train wall/FLOPs, latency."""
    out = []
    for s in SEEDS:
        tag = f"{prefix}__{problem}__{method}__{lvl(level)}__s{s}" if prefix == "final" else f"flops__{problem}__{method}__s{s}"
        if method != "pinn":
            r = get(tag)
            if not r:
                continue
            out.append({"sub": np.mean([r["test"]["rel_l2"][i] for i in PINN_TEST_IDS]), "full": r["test"]["rel_l2_mean"],
                        "ood_sub": np.mean([r["ood"]["rel_l2"][i] for i in PINN_OOD_IDS]) if "ood" in r else None,
                        "ood_full": r["ood"]["rel_l2_mean"] if "ood" in r else None,
                        "qoi": np.mean([r["test"]["qoi_rel_err"][i] for i in PINN_TEST_IDS]),
                        "qoi_full": r["test"]["qoi_rel_err_mean"],
                        "wall": r["train"]["wall_s"], "flops": r["train"]["flops"],
                        "infer": r["infer_batch1"]["median_s"], "steps": r["train"]["steps"]})
        else:
            if prefix == "final":
                te = [get(f"{tag}__test{i}") for i in PINN_TEST_IDS]
                od = [get(f"{tag}__ood{i}") for i in PINN_OOD_IDS]
            else:
                te, od = [get(tag)], []
            te = [r for r in te if r]
            if not te:
                continue
            od = [r for r in od if r]
            out.append({"sub": np.mean([r["test"]["rel_l2_mean"] for r in te]), "full": None,
                        "ood_sub": np.mean([r["ood"]["rel_l2_mean"] for r in od]) if od else None, "ood_full": None,
                        "qoi": np.mean([r["test"]["qoi_rel_err_mean"] for r in te]), "qoi_full": None,
                        "wall": np.mean([r["train"]["wall_s"] for r in te]),
                        "flops": np.mean([r["train"]["flops"] for r in te]),
                        "infer": np.mean([r["train"]["wall_s"] + r["infer_eval_grid"]["median_s"] for r in te]),
                        "steps": np.mean([r["train"]["steps"] for r in te])})
    return out


def agg(rows, key):
    return ms([r[key] for r in rows])


def std_table(problem):
    dg = datagen(problem)
    data_s = dg["train_set_core_s"] if dg else None
    rows = []
    pinn = std_seed_metrics(problem, "pinn")
    pinn_cost = AmortizedCost(per_query_train_s=agg(pinn, "infer")[0] or 0.0)
    for m in ("pinn", "fno", "deeponet"):
        R = std_seed_metrics(problem, m)
        if not R:
            continue
        infer = agg(R, "infer")[0]
        if m == "pinn":
            cost, be = pinn_cost, None
        else:
            cost = AmortizedCost(data_s=data_s or 0.0, train_s=agg(R, "wall")[0], infer_s=infer)
            be = cost.break_even(pinn_cost)
        rows.append({"problem": problem, "method": m, "n_seeds": len(R),
                     "rel_l2_sub": agg(R, "sub"), "rel_l2_full": agg(R, "full"), "rel_l2_ood": agg(R, "ood_sub"),
                     "rel_l2_ood_full": agg(R, "ood_full"), "qoi": agg(R, "qoi"), "train_s": agg(R, "wall"),
                     "flops": agg(R, "flops"), "data_core_s": None if m == "pinn" else data_s,
                     "time_per_instance_s": infer, "break_even_queries": be,
                     "per_query_at_100": cost.per_query(100), "per_query_at_10000": cost.per_query(10000)})
    if dg:
        rows.append({"problem": problem, "method": "solver", "time_per_instance_s": dg["solver_core_s_per_sample"]})
    return rows


# ------------------------------------------------------------------------------------------------ airfoil
def airfoil_cases():
    ids = sorted({r["spec"]["case"] for r in RUNS_ALL.values() if r["spec"].get("problem") == "airfoil"
                  and r["spec"]["method"] == "pinn" and r["spec"]["stage"] == "final" and r["spec"]["level"] == 1.0})
    naca = [c for c in ids if c.startswith("NACA")]
    return [c for c in ids if not c.startswith("NACA")], naca


def airfoil_seed_metrics(method, level=1.0, cases=None):
    test_cases, naca_cases = airfoil_cases()
    cases = cases if cases is not None else test_cases
    out = []
    for s in SEEDS:
        tag = f"final__airfoil__{method}__{lvl(level)}__s{s}"
        if method != "pinn":
            r = get(tag)
            if not r:
                continue
            te, na = r["test"], r["naca"]
            pick = [te["ids"].index(c) for c in cases if c in te["ids"]]
            row = {"cl_sub": np.mean([te["cl_abs_err"][i] for i in pick]) if pick else None,
                   "cd_sub": np.mean([te["cd_rel_err"][i] for i in pick]) if pick else None,
                   "cl_full": te["cl_abs_err_mean"], "cd_full": te["cd_rel_err_mean"],
                   "cl_naca": na["cl_abs_err_mean"], "cd_naca": na["cd_rel_err_mean"],
                   "u_sub": np.mean([te["u_rel_l2"][i] for i in pick]) if "u_rel_l2" in te and pick else None,
                   "p_sub": np.mean([te["p_rel_l2"][i] for i in pick]) if "p_rel_l2" in te and pick else None,
                   "wall": r["train"]["wall_s"], "flops": r["train"]["flops"], "infer": r["infer_batch1"]["median_s"]}
            nid = [na["ids"].index(c) for c in naca_cases if c in na["ids"]]
            row["cl_naca_case"] = np.mean([na["cl_abs_err"][i] for i in nid]) if nid else None
            row["cd_naca_case"] = np.mean([na["cd_rel_err"][i] for i in nid]) if nid else None
            out.append(row)
        else:
            te = [get(f"{tag}__{c}") for c in cases]
            te = [r for r in te if r]
            if not te:
                continue
            na = [get(f"{tag}__{c}") for c in naca_cases]
            na = [r for r in na if r]
            out.append({"cl_sub": np.mean([r["result"]["cl_abs_err_mean"] for r in te]),
                        "cd_sub": np.mean([r["result"]["cd_rel_err_mean"] for r in te]),
                        "u_sub": np.mean([r["result"]["u_rel_l2_mean"] for r in te]),
                        "p_sub": np.mean([r["result"]["p_rel_l2_mean"] for r in te]),
                        "cl_full": None, "cd_full": None, "cl_naca": None, "cd_naca": None,
                        "cl_naca_case": np.mean([r["result"]["cl_abs_err_mean"] for r in na]) if na else None,
                        "cd_naca_case": np.mean([r["result"]["cd_rel_err_mean"] for r in na]) if na else None,
                        "wall": np.mean([r["train"]["wall_s"] for r in te]),
                        "flops": np.mean([r["train"]["flops"] for r in te]),
                        "infer": np.mean([r["train"]["wall_s"] + r["infer_eval_fields"]["median_s"] for r in te])})
    return out


def airfoil_campaign():
    recs = [json.load(open(f)) for f in glob.glob(str(ROOT / "data" / "airfoil" / "runs" / "*.json"))]
    sec = [r["seconds"] for r in recs if r.get("seconds")]
    clean_p = RES / "openfoam_clean_timing.json"
    clean = load_json(clean_p)["mean_wall_s"] if clean_p.exists() else None
    # the campaign shared the CPU with other jobs, so its recorded wall times are inflated; the per-design cost and
    # the data cost used for amortisation come from the clean re-timing (4 runs alone on the machine)
    return {"n_runs": len(recs), "recorded_mean_s_per_run": float(np.mean(sec)) if sec else None,
            "recorded_total_s": float(np.sum(sec)) if sec else None, "clean_s_per_run": clean,
            "mean_s_per_run": clean if clean else (float(np.mean(sec)) if sec else None),
            "total_core_s": len(recs) * clean if clean else (float(np.sum(sec)) if sec else None),
            "status": {s: sum(r.get("status") == s for r in recs) for s in sorted({r.get("status") for r in recs})}}


def airfoil_table():
    camp = airfoil_campaign()
    rows = []
    P = airfoil_seed_metrics("pinn")
    pinn_cost = AmortizedCost(per_query_train_s=agg(P, "infer")[0] or 0.0)
    for m in ("pinn", "mgn", "mlp"):
        R = airfoil_seed_metrics(m)
        if not R:
            continue
        infer = agg(R, "infer")[0]
        cost = pinn_cost if m == "pinn" else AmortizedCost(data_s=camp["total_core_s"] or 0, train_s=agg(R, "wall")[0],
                                                           infer_s=infer)
        rows.append({"problem": "airfoil", "method": m, "n_seeds": len(R),
                     **{k: agg(R, k) for k in ("cl_sub", "cd_sub", "u_sub", "p_sub", "cl_full", "cd_full", "cl_naca",
                                               "cd_naca", "cl_naca_case", "cd_naca_case", "wall", "flops")},
                     "data_core_s": None if m == "pinn" else camp["total_core_s"], "time_per_instance_s": infer,
                     "break_even_queries": None if m == "pinn" else cost.break_even(pinn_cost)})
    rows.append({"problem": "airfoil", "method": "solver", "time_per_instance_s": camp["mean_s_per_run"]})
    return rows, camp


# ------------------------------------------------------------------------------------------------ inverse heat
def heat_seed_metrics(method, level=1.0):
    out = []
    import inverse_heat as H
    for s in SEEDS:
        tag = f"final__inverse_heat__{method}__{lvl(level)}__s{s}"
        if method == "classical":
            r = get(f"final__inverse_heat__classical__s{s}")
            cases = r["cases"] if r else None
            per_query = np.mean([c["calibration_core_s"] for c in cases]) if r else None
            wall = 0.0
        elif method == "pinn":
            cases = [get(f"{tag}__{k}{i}") for k, lst in H.CASES.items() for i in range(len(lst))]
            cases = [c for c in cases if c]
            per_query = np.mean([c["train"]["wall_s"] for c in cases]) if cases else None
            wall = per_query
        else:
            r = get(tag)
            cases = r["cases"] if r else None
            per_query = np.mean([c["calibration_s"] for c in cases]) if r else None
            wall = r["train"]["wall_s"] if r else None
        if not cases:
            continue
        idc = [c for c in cases if c["kind"] == "id"]
        odc = [c for c in cases if c["kind"] == "ood"]
        out.append({"h_id": np.mean([c["h_rel_err"] for c in idc]), "h_ood": np.mean([c["h_rel_err"] for c in odc]),
                    "f_id": np.mean([c["max_field_err_C"] for c in idc]), "f_ood": np.mean([c["max_field_err_C"] for c in odc]),
                    "hot_id": np.mean([c["hot_spot_err_C"] for c in idc]), "wall": wall, "infer": per_query})
    return out


def heat_table():
    dg = datagen("inverse_heat")
    rows = []
    P = heat_seed_metrics("pinn")
    pinn_cost = AmortizedCost(per_query_train_s=agg(P, "infer")[0] or 0.0)
    for m in ("pinn", "mlp", "deeponet", "classical"):
        R = heat_seed_metrics(m)
        if not R:
            continue
        infer = agg(R, "infer")[0]
        if m in ("pinn", "classical"):
            be = None
        else:
            be = AmortizedCost(data_s=dg["train_set_core_s"], train_s=agg(R, "wall")[0], infer_s=infer).break_even(pinn_cost)
        rows.append({"problem": "inverse_heat", "method": m, "n_seeds": len(R),
                     **{k: agg(R, k) for k in ("h_id", "h_ood", "f_id", "f_ood", "hot_id", "wall")},
                     "data_core_s": dg["train_set_core_s"] if m in ("mlp", "deeponet") and dg else None,
                     "time_per_instance_s": infer, "break_even_queries": be})
    return rows


# ------------------------------------------------------------------------------------------------ flops arm
def flops_table():
    p = RES / "flops_targets.json"
    if not p.exists():
        return []
    T = load_json(p)
    rows = []
    for prob in STD:
        for m in ("pinn", "fno", "deeponet"):
            R = std_seed_metrics(prob, m, prefix="flops")
            if R:
                rows.append({"problem": prob, "method": m, "flops_budget": T[prob]["flops"], "n_seeds": len(R),
                             "rel_l2_sub": agg(R, "sub"), "train_s": agg(R, "wall"), "steps": agg(R, "steps")})
    return rows


# ------------------------------------------------------------------------------------------------ writing
def write_tables(std, air, camp, heat, flo, hw):
    L = []
    hw_line = (f"Hardware: {hw.get('cpu')} ({hw.get('logical_cpus')} logical CPUs, {hw.get('mem_gb')} GB), no GPU; "
               f"torch {hw.get('torch')}. **Every training run uses ONE CPU core** (torch.set_num_threads(1)); four "
               f"runs execute concurrently, for every method alike. Budgets are small because there is no GPU.")
    L += ["# Experiment 09: equal-budget results", "", hw_line, "",
          "Mean ± standard deviation over 3 seeds (sample standard deviation). Test errors marked *3 inst.* are on the "
          "three test instances every PINN was trained on (the operators on exactly the same instances); *all* is the "
          "operators' whole test set (200). Time per new instance: operator = one forward pass (batch 1); PINN = its "
          "training on that instance (it has to be retrained for every new input) plus evaluation; reference solver = "
          "one solve. Break-even: number of new queries above which the operator, including its simulation data and "
          "training, costs less than per-instance PINNs (all core-seconds; '0' = always cheaper).", ""]
    for prob in STD:
        rows = [r for r in std if r["problem"] == prob]
        B = next((r for r in rows if r["method"] != "solver"), None)
        L += [f"## {TITLE[prob]} (budget {PROBLEMS[prob]['budget_s']:g} s on one core per training run)", "",
              "| Method | Rel. L2, test (3 inst.) | Rel. L2, test (all) | Rel. L2, OOD (2 inst.) | Integral QoI rel. err (3 inst.) "
              "| Training (s) | Training FLOPs | Simulation data (core-s) | Time per new instance | Break-even (queries) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["method"] == "solver":
                L.append(f"| {NAME['solver']} | reference | reference | reference | reference | - | - | - | "
                         f"{fmt_s(r['time_per_instance_s'])} | - |")
                continue
            be = r["break_even_queries"]
            L.append(f"| {NAME[r['method']]} | {fmt(*r['rel_l2_sub'][:2])} | {fmt(*r['rel_l2_full'][:2])} | "
                     f"{fmt(*r['rel_l2_ood'][:2])} | {fmt(*r['qoi'][:2])} | {fmt_s(r['train_s'][0])} | "
                     f"{r['flops'][0]:.2e} | {fmt_s(r['data_core_s']) if r['data_core_s'] else '-'} | "
                     f"{fmt_s(r['time_per_instance_s'])} | {'-' if be is None else f'{be:.1f}'} |")
        L.append("")
    L += [f"## {TITLE['airfoil']} (budget 900 s on one core per training run)", "",
          f"OpenFOAM campaign: {camp['n_runs']} runs (status {camp['status']}). One run alone on one core: "
          f"{fmt_s(camp['clean_s_per_run'])} (`openfoam_clean_timing.json`, 4 runs re-timed on the idle machine), so "
          f"the data cost used below is {camp['n_runs']} x that = {camp['total_core_s'] / 3600:.1f} core-hours. The "
          f"campaign's own recorded wall times sum to {camp['recorded_total_s'] / 3600:.1f} h of runs "
          f"({fmt_s(camp['recorded_mean_s_per_run'])} per run) because it shared the CPU with other jobs.",
          "",
          "| Method | Cl abs. err (4 test runs) | Cd rel. err (4 test runs) | Velocity rel. L2 (4 test runs) | Pressure rel. L2 "
          "(4 test runs) | Cl abs. err (all test) | Cd rel. err (all test) | Cl abs. err, NACA 2412 α=4° | Cd rel. err, NACA 2412 α=4° "
          "| Training (s) | Time per new design | Break-even (queries) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in air:
        if r["method"] == "solver":
            L.append(f"| OpenFOAM (reference) | reference | reference | reference | reference | reference | reference "
                     f"| reference | reference | - | {fmt_s(r['time_per_instance_s'])} | - |")
            continue
        be = r["break_even_queries"]
        L.append(f"| {NAME[r['method']]} | {fmt(*r['cl_sub'][:2])} | {fmt(*r['cd_sub'][:2], pct=True)} | "
                 f"{fmt(*r['u_sub'][:2])} | {fmt(*r['p_sub'][:2])} | {fmt(*r['cl_full'][:2])} | "
                 f"{fmt(*r['cd_full'][:2], pct=True)} | {fmt(*r['cl_naca_case'][:2])} | {fmt(*r['cd_naca_case'][:2], pct=True)} | "
                 f"{fmt_s(r['wall'][0])} | {fmt_s(r['time_per_instance_s'])} | {'-' if be is None else f'{be:.1f}'} |")
    L += ["", f"## {TITLE['inverse_heat']} (budget 120 s on one core per training run)", "",
          "| Method | h rel. err, in range (3 cases) | h rel. err, out of range (2 cases) | Max field err °C, in range | "
          "Max field err °C, out of range | Hot-spot err °C, in range | Training (s) | Simulation data (core-s) | "
          "Time per new inverse problem | Break-even (queries) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in heat:
        be = r["break_even_queries"]
        L.append(f"| {NAME[r['method']]} | {fmt(*r['h_id'][:2], pct=True)} | {fmt(*r['h_ood'][:2], pct=True)} | "
                 f"{fmt(*r['f_id'][:2])} | {fmt(*r['f_ood'][:2])} | {fmt(*r['hot_id'][:2])} | "
                 f"{fmt_s(r['wall'][0]) if r['method'] not in ('classical', 'pinn') else '-'} | "
                 f"{fmt_s(r['data_core_s']) if r['data_core_s'] else '-'} | {fmt_s(r['time_per_instance_s'])} | "
                 f"{'-' if be is None else f'{be:.1f}'} |")
    if flo:
        L += ["", "## Equal-FLOPs arm (standard problems)", "",
              "Every method trained to the same FLOP count: the smallest any method used in its full wall-clock budget "
              "(median over seeds); wall-clock cap 3x the budget. PINN on test instance 0 only.", "",
              "| Problem | Method | FLOP budget | Rel. L2 (PINN: instance 0; operators: 3 inst.) | Training (s) | Steps |",
              "|---|---|---|---|---|---|"]
        for r in flo:
            L.append(f"| {TITLE[r['problem']]} | {NAME[r['method']]} | {r['flops_budget']:.2e} | {fmt(*r['rel_l2_sub'][:2])} | "
                     f"{fmt_s(r['train_s'][0])} | {r['steps'][0]:.0f} |")
    bad = failed()
    L += ["", f"Failed runs: {len(bad)}" + (" — " + ", ".join(bad) if bad else ".")]
    (RES / "table.md").write_text("\n".join(L) + "\n")

    with open(RES / "table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem", "method", "metric", "mean", "std", "n_seeds"])
        for r in std + air + heat + [dict(x, method=x["method"] + "@equal_flops") for x in flo]:
            for k, v in r.items():
                if k in ("problem", "method", "n_seeds"):
                    continue
                if isinstance(v, tuple):
                    w.writerow([r["problem"], r["method"], k, v[0], v[1], v[2]])
                elif isinstance(v, (int, float)) and v is not None:
                    w.writerow([r["problem"], r["method"], k, v, "", r.get("n_seeds", "")])


def style(ax):
    from matplotlib.ticker import LogFormatterSciNotation, NullFormatter
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def fig_budget():
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    for ax, prob in zip(axes, STD + ("airfoil", "inverse_heat")):
        for m in (("pinn", "fno", "deeponet") if prob in STD else ("pinn", "mgn", "mlp") if prob == "airfoil"
                  else ("pinn", "mlp", "deeponet")):
            xs, ys, es = [], [], []
            for lv in BUDGET_LEVELS:
                if prob in STD:
                    R = std_seed_metrics(prob, m, lv)
                    vals, x = [r["sub"] for r in R], [r["wall"] for r in R]
                elif prob == "airfoil":
                    test_cases, _ = airfoil_cases()
                    R = airfoil_seed_metrics(m, lv, cases=test_cases[:1])
                    vals, x = [r["cd_sub"] for r in R], [r["wall"] for r in R]
                else:
                    R = heat_seed_metrics(m, lv)
                    vals, x = [r["h_id"] for r in R], [r["wall"] for r in R]
                mu, sd, n = ms(vals)
                if n:
                    xs.append(np.mean(x)), ys.append(mu), es.append(sd)
            if xs:
                ax.errorbar(xs, ys, yerr=es, color=COLOR[m], marker=MARK[m], ms=6, lw=2, capsize=3, label=NAME[m])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(TITLE[prob], fontsize=10, color=INK)
        ax.set_xlabel("training budget (s, one core)", fontsize=8, color=INK2)
        style(ax)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("relative L2 error (test)", fontsize=8, color=INK2)
    axes[3].set_ylabel("Cd relative error (test run 1)", fontsize=8, color=INK2)
    axes[4].set_ylabel("h relative error (in range)", fontsize=8, color=INK2)
    fig.suptitle("Error vs training budget (mean ± std over 3 seeds; same budgets for every method)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_error_vs_budget.png", dpi=150)
    plt.close(fig)


def fig_pareto(std, air, heat):
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    keys = {**{p: "rel_l2_sub" for p in STD}, "airfoil": "cd_sub", "inverse_heat": "h_id"}
    for ax, prob in zip(axes, STD + ("airfoil", "inverse_heat")):
        rows = [r for r in std + air + heat if r["problem"] == prob]
        for r in rows:
            m = r["method"]
            if m == "solver":
                ax.axvline(r["time_per_instance_s"], color=INK2, ls="--", lw=1)
                ax.text(r["time_per_instance_s"], 0.98, " solver", transform=ax.get_xaxis_transform(), fontsize=7,
                        color=INK2, va="top")
                continue
            if m == "classical":
                ax.axvline(r["time_per_instance_s"], color=INK2, ls=":", lw=1)
                ax.text(r["time_per_instance_s"], 0.98, " FV+calib.", transform=ax.get_xaxis_transform(), fontsize=7,
                        color=INK2, va="top")
                continue
            mu, sd, _ = r[keys[prob]]
            if mu is None:
                continue
            ax.errorbar([r["time_per_instance_s"]], [mu], yerr=[sd], color=COLOR[m], marker=MARK[m], ms=8, capsize=3,
                        lw=0, elinewidth=1.5, label=NAME[m])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(TITLE[prob], fontsize=10, color=INK)
        ax.set_xlabel("time per new instance (s, one core)", fontsize=8, color=INK2)
        style(ax)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    axes[0].set_ylabel("error (as in Fig. 1)", fontsize=8, color=INK2)
    fig.suptitle("Error vs time per new instance (lower left is better). PINN time = training per instance.", fontsize=11,
                 color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_pareto_error_vs_inference.png", dpi=150)
    plt.close(fig)


def fig_summary(std, air, heat, hw):
    lines = []
    for prob in STD:
        for r in std:
            if r["problem"] == prob and r["method"] != "solver":
                lines.append([TITLE[prob], NAME[r["method"]], fmt(r["rel_l2_sub"][0], digits=2),
                              fmt(r["rel_l2_ood"][0], digits=2), fmt_s(r["time_per_instance_s"])])
    for r in air:
        if r["method"] != "solver":
            lines.append([TITLE["airfoil"], NAME[r["method"]], "Cd " + fmt(r["cd_sub"][0], digits=2, pct=True),
                          "NACA Cd " + fmt(r["cd_naca_case"][0], digits=2, pct=True), fmt_s(r["time_per_instance_s"])])
    for r in heat:
        lines.append([TITLE["inverse_heat"], NAME[r["method"]], "h " + fmt(r["h_id"][0], digits=3, pct=True),
                      "h " + fmt(r["h_ood"][0], digits=3, pct=True), fmt_s(r["time_per_instance_s"])])
    h = 0.26 * (len(lines) + 1) + 1.1
    fig = plt.figure(figsize=(10, h))
    ax = fig.add_axes([0.02, 0.45 / h, 0.96, 1 - 1.0 / h])
    ax.axis("off")
    tab = ax.table(cellText=lines, colLabels=["Problem", "Method", "Error, in distribution", "Error, out of distribution",
                                              "Time per new instance"], cellLoc="left", colLoc="left",
                   bbox=[0, 0, 1, 1], colWidths=[0.22, 0.2, 0.2, 0.22, 0.16])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    prev = None
    for (i, j), c in tab.get_celld().items():
        c.set_edgecolor(GRID)
        if i == 0:
            c.set_text_props(weight="bold", color=INK)
        elif j == 0 and lines[i - 1][0] == (lines[i - 2][0] if i > 1 else None):
            c.get_text().set_text("")
    fig.text(0.02, 1 - 0.12 / h, "PINN vs FNO vs DeepONet vs MeshGraphNet under the same compute budget", fontsize=11,
             weight="bold", color=INK, va="top")
    fig.text(0.02, 1 - 0.4 / h, f"PINNeAPPle-Benchmark exp. 09 · 3 seeds (means) · one CPU core per run, no GPU · "
                         f"{hw.get('cpu', '')}", fontsize=7.5, color=INK2, va="top")
    fig.text(0.02, 0.12 / h, "Errors: relative L2 of the field (standard problems), Cd relative error (airfoil), h relative "
                         "error (inverse heat). PINN time per instance = its training.", fontsize=7, color=INK2)
    fig.savefig(FIG / "fig0_summary_table.png", dpi=200)
    plt.close(fig)


def main():
    std = [r for p in STD for r in std_table(p)]
    air, camp = airfoil_table()
    heat = heat_table()
    flo = flops_table()
    hw = next((r["hardware"] for r in RUNS_ALL.values()), {})
    write_tables(std, air, camp, heat, flo, hw)
    fig_budget()
    fig_pareto(std, air, heat)
    fig_summary(std, air, heat, hw)
    save_json(RES / "summary.json", {"standard": std, "airfoil": air, "airfoil_campaign": camp, "inverse_heat": heat,
                                     "equal_flops": flo, "hardware": hw, "failed_runs": failed(),
                                     "n_runs": len(RUNS_ALL)})
    print((RES / "table.md").read_text())


if __name__ == "__main__":
    main()
