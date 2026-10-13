"""Airfoil part of the report: integral and field tables (n in every header), the panel-method reference, the PINN
budget study and the PINN setup audit. Every number is read from results/runs/*.json or computed here from the
OpenFOAM runs and the library's panel method (pinneapple_design.aero.panel)."""
from __future__ import annotations

import glob
import json
import math
import time

import numpy as np

from common import DATA, RES, SEEDS, load_json

SETS = ("test4", "test", "naca", "ood4", "ood")
SET_LABEL = {"test4": "held-out test, PINN cases", "test": "held-out test, all", "naca": "NACA polars",
             "ood4": "OOD, PINN cases", "ood": "OOD, all"}
PINNS = ("pinn", "pinn_rans")
DATA_DRIVEN = ("mgn", "mlp", "mlp_fields")
NAME = {"pinn": "PINN (potential flow)", "pinn_rans": "PINN (RANS, mixing length)", "mgn": "MeshGraphNet",
        "mlp": "MLP (Cl, Cd only)", "mlp_fields": "POD + MLP (fields only)", "panel": "Panel method (inviscid, no training)",
        "openfoam": "OpenFOAM (reference)"}


def _runs(dirname):
    out = {}
    for f in glob.glob(str(DATA / dirname / "runs" / "*.json")):
        r = json.load(open(f))
        r["npz"] = f[:-5] + ".npz"
        out[r["id"]] = r
    return out


def pinn_cases(R):
    ids = {}
    for r in R.values():
        sp = r["spec"]
        if sp.get("problem") == "airfoil" and sp["method"] in PINNS and sp["stage"] == "final" and sp["level"] == 1.0:
            ids[sp["case"]] = True
    test4 = sorted(c for c in ids if c.startswith("s"))
    ood4 = sorted(c for c in ids if c.startswith("NACA") and c != "NACA2412_a+04")
    return test4, ood4


def _mean(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def _ms(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n_seeds": len(v)}


def seed_metrics(R, method, level=1.0):
    """{set: {metric: [per-seed mean]}} plus n (instances per set) for one method."""
    test4, ood4 = pinn_cases(R)
    res = {s: {k: [] for k in ("cl", "cd", "u", "p")} for s in SETS}
    n = {}
    cost = {"wall": [], "infer": []}
    for s in SEEDS:
        tag = f"final__airfoil__{method}__L{level:.3f}__s{s}"
        if method in PINNS:
            groups = {"test4": test4, "naca": ["NACA2412_a+04"], "ood4": ood4}
            for set_, cases in groups.items():
                rr = [R[f"{tag}__{c}"]["result"] for c in cases if R.get(f"{tag}__{c}", {}).get("ok")]
                if not rr:
                    continue
                n[set_] = len(rr)
                res[set_]["cl"].append(_mean([x["result"]["cl_abs_err_mean"] for x in rr]))
                res[set_]["cd"].append(_mean([x["result"]["cd_rel_err_mean"] for x in rr]))
                res[set_]["u"].append(_mean([x["result"]["u_rel_l2_mean"] for x in rr]))
                res[set_]["p"].append(_mean([x["result"]["p_rel_l2_mean"] for x in rr]))
                if set_ == "test4":
                    cost["wall"].append(_mean([x["train"]["wall_s"] for x in rr]))
                    cost["infer"].append(_mean([x["train"]["wall_s"] + x["infer_eval_fields"]["median_s"] for x in rr]))
            continue
        r = R.get(tag)
        if not r or not r.get("ok"):
            continue
        r = r["result"]
        for set_, src, pick in (("test4", "test", test4), ("test", "test", None), ("naca", "naca", None),
                                ("ood4", "ood", ood4), ("ood", "ood", None)):
            if src not in r:
                continue
            d = r[src]
            idx = list(range(len(d["ids"]))) if pick is None else [d["ids"].index(c) for c in pick if c in d["ids"]]
            if not idx:
                continue
            n[set_] = len(idx)
            for key, metric in (("cl", "cl_abs_err"), ("cd", "cd_rel_err"), ("u", "u_rel_l2"), ("p", "p_rel_l2")):
                if metric in d:
                    res[set_][key].append(_mean([d[metric][i] for i in idx]))
        cost["wall"].append(r["train"]["wall_s"])
        cost["infer"].append(r["infer_batch1"]["median_s"])
    out = {s: {k: _ms(v) for k, v in m.items()} for s, m in res.items()}
    return out, n, {k: _mean(v) for k, v in cost.items()}, len(cost["wall"])


def panel_reference(R):
    """The panel method on every evaluation run: inviscid Cl (Cd = 0), and the velocity/pressure on the graph nodes."""
    from pinneapple_design.aero.panel import panel_solve
    from pinneapple_design.aero.surrogate import Layout
    test4, ood4 = pinn_cases(R)
    main, ood = _runs("airfoil"), _runs("airfoil_ood")
    lay = Layout()
    sets = {"test4": test4, "naca": [k for k, r in main.items() if r.get("set") == "reference"],
            "ood4": ood4, "ood": sorted(ood)}
    # the held-out test runs: the same split as everyone else (airfoil.splits)
    import airfoil as A
    S = A.splits()
    sets["test"] = [r["id"] for r in S["test"]]
    sets["naca"] = [r["id"] for r in S["naca"]]
    sets["ood"] = [r["id"] for r in S.get("ood", [])]
    allruns = {**main, **ood}
    out, n, times = {}, {}, []
    for set_, ids in sets.items():
        cl, cd, uu, pp = [], [], [], []
        for i in ids:
            r = allruns[i]
            shape = np.array(r["shape"])
            t0 = time.perf_counter()
            sol = panel_solve(shape, r["alpha"])
            times.append(time.perf_counter() - t0)
            cl.append(abs(sol.cl_kutta_joukowski - r["Cl"]))
            cd.append(1.0)                                         # d'Alembert: zero drag, 100 % error
            pos = lay.positions(shape)
            u, v = sol.velocity(pos[:, 0], pos[:, 1])
            z = np.load(r["npz"])
            t = np.stack([z["p"], z["Ux"], z["Uy"]], 1)[lay.cell_index]
            pr = np.stack([0.5 * (1 - u ** 2 - v ** 2), u, v], 1)
            uu.append(np.linalg.norm(pr[:, 1:] - t[:, 1:]) / np.linalg.norm(t[:, 1:]))
            pp.append(np.linalg.norm(pr[:, 0] - t[:, 0]) / np.linalg.norm(t[:, 0]))
        if not ids:
            continue
        n[set_] = len(ids)
        out[set_] = {k: {"mean": float(np.mean(v)), "std": 0.0, "n_seeds": 1} for k, v in
                     (("cl", cl), ("cd", cd), ("u", uu), ("p", pp))}
    return out, n, float(np.median(times))


def pinn_vs_panel(R):
    """Did the potential-flow PINN solve its own equations? Its Kutta-Joukowski Cl against the panel method's."""
    from pinneapple_design.aero.panel import panel_solve
    rows = []
    allruns = {**_runs("airfoil"), **_runs("airfoil_ood")}
    for k, r in sorted(R.items()):
        sp = r["spec"]
        if sp.get("problem") != "airfoil" or sp["method"] != "pinn" or not r.get("ok") or "case" not in sp:
            continue
        run = allruns[sp["case"]]
        cl_panel = panel_solve(np.array(run["shape"]), run["alpha"]).cl_kutta_joukowski
        rows.append({"job": k, "case": sp["case"], "stage": sp["stage"], "budget_s": sp["budget"]["wall_s"],
                     "cl_pinn": r["result"]["cl_kutta_joukowski"], "cl_panel": cl_panel, "cl_openfoam": run["Cl"]})
    return rows


def budget_study(R):
    rows = []
    for k, r in sorted(R.items()):
        sp = r["spec"]
        if sp.get("problem") != "airfoil" or sp["method"] not in PINNS or not r.get("ok") or sp.get("seed") != 0:
            continue
        if not (sp["stage"] == "study" or (sp["stage"] == "final" and sp["level"] == 1.0)):
            continue
        res = r["result"]
        rows.append({"method": sp["method"], "case": sp["case"], "budget_s": sp["budget"]["wall_s"],
                     "steps": res["train"]["steps"], "cl_abs_err": res["result"]["cl_abs_err_mean"],
                     "cd_rel_err": res["result"]["cd_rel_err_mean"], "u_rel_l2": res["result"]["u_rel_l2_mean"],
                     "p_rel_l2": res["result"]["p_rel_l2_mean"], "final_loss": res["train"]["history"][-1]["loss"],
                     "loss_components": res.get("loss_components")})
    return [r for r in rows if r["case"] in {x["case"] for x in rows if x["budget_s"] > 900}]


def audit(R):
    rows = []
    for k, r in sorted(R.items()):
        sp = r["spec"]
        if sp.get("stage") != "diag" or not r.get("ok"):
            continue
        res = r["result"]
        rows.append({"method": sp["method"], "variant": sp["variant"], "case": sp["case"], "cfg": sp["cfg"],
                     "steps": res["train"]["steps"], "cl_abs_err": res["result"]["cl_abs_err_mean"],
                     "cd_rel_err": res["result"]["cd_rel_err_mean"], "u_rel_l2": res["result"]["u_rel_l2_mean"],
                     "loss_components": res.get("loss_components")})
    return rows


def collocation_stats():
    """How the PINN collocation pool is distributed (O-grid cell centres): near the wall and the leading edge."""
    import airfoil as A
    from pinneapple_design.aero.geometry import REFERENCE
    from pinneapple_design.aero.panel import airfoil_nodes
    out = {}
    shape = REFERENCE["NACA 2412"]
    wall = airfoil_nodes(shape, 200)
    for kind, dmin in (("potential", 1e-3), ("rans", 0.0)):
        pts = A.collocation_pool(shape, dmin).numpy()
        d = np.min(np.hypot(pts[:, None, 0] - wall[None, :, 0], pts[:, None, 1] - wall[None, :, 1]), axis=1)
        out[kind] = {"n_points": int(len(pts)), "frac_within_0.01c_of_wall": float((d < 0.01).mean()),
                     "frac_within_1e-4c_of_wall": float((d < 1e-4).mean()),
                     "frac_within_0.05c_of_leading_edge": float((np.hypot(pts[:, 0], pts[:, 1]) < 0.05).mean()),
                     "min_wall_distance": float(d.min())}
    return out


def build(R):
    rows = {}
    for m in PINNS + DATA_DRIVEN:
        sm, n, cost, nseed = seed_metrics(R, m)
        if nseed or any(n.values()):
            rows[m] = {"sets": sm, "n": n, "cost": cost, "n_seeds": nseed}
    panel, pn, pt = panel_reference(R)
    rows["panel"] = {"sets": panel, "n": pn, "cost": {"wall": 0.0, "infer": pt}, "n_seeds": 1}
    clean = load_json(RES / "openfoam_clean_timing.json")["mean_wall_s"]
    rows["openfoam"] = {"cost": {"wall": None, "infer": clean}}
    return {"rows": rows, "pinn_vs_panel": pinn_vs_panel(R), "budget_study": budget_study(R), "audit": audit(R),
            "collocation": collocation_stats()}


def _cell(sets, set_, keys, n_ref, n_have, pct_second=True):
    d = sets.get(set_) if sets else None
    if not d or all(d.get(k) is None for k in keys):
        return "-"
    parts = []
    for j, k in enumerate(keys):
        v = d.get(k)
        if v is None:
            parts.append("-")
            continue
        pct = (k == "cd") and pct_second
        f = (lambda x: f"{100 * x:.3g}%") if pct else (lambda x: f"{x:.3g}")
        parts.append(f(v["mean"]) + (f" ± {f(v['std'])}" if v["n_seeds"] > 1 else ""))
    s = " / ".join(parts)
    if n_have is not None and n_have != n_ref:
        s += f" [n={n_have}]"
    return s


def markdown(A, fmt_s):
    rows = A["rows"]
    ref = rows.get("mgn", rows.get("mlp"))
    nref = {s: max((r["n"].get(s, 0) for r in rows.values() if "n" in r), default=0) for s in SETS}
    order = [m for m in ("pinn", "pinn_rans", "mgn", "mlp", "mlp_fields", "panel") if m in rows]
    L = ["### Integral quantities: Cl absolute error / Cd relative error", "",
         "Mean ± std over seeds of the per-seed mean over the n runs of each set. A cell marked [n=...] has fewer runs "
         "than the column (the per-geometry PINNs: 1 NACA case, 4 OOD cases, seed 0 only for OOD). "
         "A scalar regressor (the MLP) is trained on exactly these scalars and is expected to beat field models on them.",
         "", "| Method | " + " | ".join(f"{SET_LABEL[s]} (n = {nref[s]})" for s in SETS) + " | Time per new design |",
         "|---|" + "---|" * (len(SETS) + 1)]
    for m in order:
        if m == "mlp_fields":
            L.append(f"| {NAME[m]} | " + " | ".join("fields only" for _ in SETS) + f" | {fmt_s(rows[m]['cost']['infer'])} |")
            continue
        r = rows[m]
        L.append(f"| {NAME[m]} | " + " | ".join(_cell(r["sets"], s, ("cl", "cd"), nref[s], r["n"].get(s)) for s in SETS)
                 + f" | {fmt_s(r['cost']['infer'])} |")
    L.append(f"| {NAME['openfoam']} | reference | reference | reference | reference | reference | "
             f"{fmt_s(rows['openfoam']['cost']['infer'])} |")
    L += ["", "### Fields on the near-field graph nodes: velocity rel. L2 / pressure rel. L2", "",
          "The same 2,850 nodes for every method (the MeshGraphNet's graph, ~5 chords, boundary layer included). The "
          "MLP predicts only Cl and Cd. The potential-flow PINN and the panel method have no boundary layer, which "
          "bounds their velocity error from below.", "",
          "| Method | " + " | ".join(f"{SET_LABEL[s]} (n = {nref[s]})" for s in SETS) + " |",
          "|---|" + "---|" * len(SETS)]
    for m in order:
        if m == "mlp":
            L.append(f"| {NAME[m]} | " + " | ".join("integrals only" for _ in SETS) + " |")
            continue
        r = rows[m]
        L.append(f"| {NAME[m]} | " + " | ".join(_cell(r["sets"], s, ("u", "p"), nref[s], r["n"].get(s), False)
                                                for s in SETS) + " |")
    return "\n".join(L), nref


def markdown_study(A, fmt_s):
    L = ["### PINN budget study (seed 0, one core)", "",
         "| PINN | Case | Budget | Steps | Cl abs. err | Cd rel. err | Velocity rel. L2 | Pressure rel. L2 | Final loss |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(A["budget_study"], key=lambda r: (r["method"], r["case"], r["budget_s"])):
        L.append(f"| {NAME[r['method']]} | {r['case']} | {fmt_s(r['budget_s'])} | {r['steps']} | {r['cl_abs_err']:.3g} | "
                 f"{100 * r['cd_rel_err']:.3g}% | {r['u_rel_l2']:.3g} | {r['p_rel_l2']:.3g} | {r['final_loss']:.3g} |")
    L += ["", "### PINN setup audit (NACA 2412 at 4°, 900 s, seed 0)", "",
          "| PINN | Variant | Steps | Cl abs. err | Cd rel. err | Velocity rel. L2 | Loss terms at the end |",
          "|---|---|---|---|---|---|---|"]
    for r in A["audit"]:
        lc = ", ".join(f"{k} {v:.2g}" for k, v in (r["loss_components"] or {}).items() if k != "total")
        L.append(f"| {NAME[r['method']]} | {r['variant']} | {r['steps']} | {r['cl_abs_err']:.3g} | {100 * r['cd_rel_err']:.3g}% "
                 f"| {r['u_rel_l2']:.3g} | {lc} |")
    pv = A["pinn_vs_panel"]
    if pv:
        L += ["", "### Did the potential-flow PINN solve its own equations?", "",
              "Its Kutta-Joukowski Cl against the exact potential-flow (panel method) Cl of the same airfoil and angle, "
              "and OpenFOAM's (RANS) Cl.", "",
              "| Runs | Budget | n | Mean abs. Cl(PINN) - Cl(panel) | Mean abs. Cl(panel) - Cl(OpenFOAM) |", "|---|---|---|---|---|"]
        groups = {}
        for r in pv:
            groups.setdefault((r["stage"], r["budget_s"]), []).append(r)
        for (st, b), rs in sorted(groups.items(), key=lambda kv: kv[0][1]):
            L.append(f"| {st} | {fmt_s(b)} | {len(rs)} | {np.mean([abs(r['cl_pinn'] - r['cl_panel']) for r in rs]):.3g} | "
                     f"{np.mean([abs(r['cl_panel'] - r['cl_openfoam']) for r in rs]):.3g} |")
    c = A["collocation"]
    L += ["", "Collocation (O-grid cell centres, NACA 2412): " + "; ".join(
        f"{k}: {v['n_points']} points, {100 * v['frac_within_0.01c_of_wall']:.0f} % within 0.01 c of the wall, "
        f"{100 * v['frac_within_0.05c_of_leading_edge']:.0f} % within 0.05 c of the leading edge, nearest "
        f"{v['min_wall_distance']:.1e} c" for k, v in c.items()) + "."]
    return "\n".join(L)


def fig_study(A, path, COLOR, MARK, style):
    import matplotlib.pyplot as plt
    rows = A["budget_study"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, label) in zip(axes, (("cl_abs_err", "Cl absolute error"), ("cd_rel_err", "Cd relative error"),
                                       ("u_rel_l2", "velocity rel. L2 (near field)"))):
        for m, color in (("pinn", COLOR["pinn"]), ("pinn_rans", "#6250d6")):
            for case, ls in zip(sorted({r["case"] for r in rows}), ("-", "--")):
                pts = sorted((r["budget_s"], r[key]) for r in rows if r["method"] == m and r["case"] == case)
                if pts:
                    ax.plot([p[0] / 3600 for p in pts], [p[1] for p in pts], ls, marker="o" if m == "pinn" else "s",
                            color=color, lw=2, ms=6, label=f"{NAME[m]}, {case}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training budget (hours, one core)", fontsize=8)
        ax.set_title(label, fontsize=10)
        style(ax)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.suptitle("Airfoil PINNs: error vs budget (seed 0). Does more compute fix them?", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
