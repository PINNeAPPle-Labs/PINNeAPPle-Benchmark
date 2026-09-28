"""Supersonic duct with a fixed-pressure outlet (shock train) -- paper."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT.parent / "paperkit"))
sys.path.insert(0, str(ROOT.parents[2] / "PINNeAPPle"))
from paperkit import CAT, INK2, SEQ, Paper, fmt, savefig, setup_mpl  # noqa: E402
from pinneapple_analysis.verification.convergence import richardson_extrapolate  # noqa: E402

setup_mpl()
G, L, H = 1.4, 12.0, 1.0
V = json.loads((RES / "verify.json").read_text())
SW = json.loads((RES / "sweep.json").read_text()) if (RES / "sweep.json").exists() else \
    [json.loads(p.read_text()) for p in sorted(RES.glob("case_sweep_*.json"))]
GR = json.loads((RES / "grid.json").read_text()) if (RES / "grid.json").exists() else None
D3S = sorted([json.loads(f.read_text()) for f in RES.glob("duct3d*.json")], key=lambda o: o["pb_ratio"])   # one or more 3-D cases
D3 = D3S[0] if D3S else None
SW = sorted(SW, key=lambda o: o["pb_ratio"])
nx = SW[0]["shape"][0]
x = (np.arange(nx) + 0.5) * L / nx

# secondary metric, fixed before the sweep results were seen: first x (> 0.3) where the wall pressure exceeds
# 1.10 x the wall pressure of the lowest-back-pressure case (verified to have no shock train: core supersonic to the exit)
REF = SW[0]
assert REF["no_train"], "the reference case must have no shock train"
p_ref = np.array(REF["p_wall"])


def wall_rise(o, thr=1.10):
    pw = np.array(o["p_wall"])
    i = np.where((x > 0.3) & (pw > thr * p_ref))[0]
    return float(x[i[0]]) if len(i) else float("nan")


for o in SW:
    o["x_wall"] = wall_rise(o)
pbs = [o["pb_ratio"] for o in SW]

# ---------------- figures
figs = {}
fig, axs = plt.subplots(len(SW), 1, figsize=(8.2, 0.75 * len(SW) + 0.6), sharex=True)
for ax, o in zip(np.atleast_1d(axs), SW):
    tag = f"field_sweep_pb{o['pb_ratio']:.2f}_{o['shape'][0]}x{o['shape'][1]}.npz"
    M = np.load(RES / tag)["Mach"]
    im = ax.imshow(M.T, origin="lower", extent=[0, L, 0, H], aspect="auto", cmap="viridis", vmin=0, vmax=2.1)
    ax.set_yticks([]); ax.set_ylabel(f"{o['pb_ratio']:.2f}", rotation=0, labelpad=14, fontsize=7)
    if np.isfinite(o["x_sonic_final"]):
        ax.axvline(o["x_sonic_final"], color="w", lw=0.7, ls="--")
np.atleast_1d(axs)[-1].set_xlabel("x / H")
fig.colorbar(im, ax=axs, shrink=0.6, label="Mach")
fig.text(0.01, 0.5, "p_b / p_in", rotation=90, va="center", fontsize=8)
figs["fields"] = savefig(fig, HERE / "fig_mach_fields.png")

fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
for j, o in enumerate(SW):
    c = plt.cm.viridis(j / max(len(SW) - 1, 1))
    ax[0].plot(x, np.array(o["p_wall"]) * G, color=c, lw=1, label=f"{o['pb_ratio']:.2f}")
    ax[1].plot(x, o["M_center"], color=c, lw=1)
ax[1].axhline(1, color=INK2, lw=0.7, ls=":")
ax[0].set_xlabel("x / H"); ax[0].set_ylabel("wall pressure p / p_in"); ax[0].legend(title="p_b/p_in", fontsize=6, ncol=2)
ax[1].set_xlabel("x / H"); ax[1].set_ylabel("centre-line Mach")
figs["profiles"] = savefig(fig, HERE / "fig_profiles.png")

fig, ax = plt.subplots(figsize=(4.6, 3.0))
_st = [o for o in SW if o["x_sonic_last10_range"] is not None and o["x_sonic_last10_range"] < 0.5]
_un = [o for o in SW if o["x_sonic_last10_range"] is not None and o["x_sonic_last10_range"] >= 0.5]
ax.plot([o["pb_ratio"] for o in _st], [o["x_sonic_final"] for o in _st], "o-", color=CAT[0], label="sonic front (centre-line M < 1)")
if _un:
    ax.plot([o["pb_ratio"] for o in _un], [o["x_sonic_final"] for o in _un], "o", mfc="none", color=CAT[0], label="sonic front, not settled")
ax.plot(pbs, [o["x_wall"] for o in SW], "s--", color=CAT[1], ms=4, label="wall-pressure rise (> 1.10 × no-train)")
ax.set_xlabel("back-pressure ratio p_b / p_in"); ax.set_ylabel("shock-train leading edge x / H"); ax.legend(fontsize=6.8)
figs["xs"] = savefig(fig, HERE / "fig_leading_edge.png")

if D3S:
    fig, ax = plt.subplots(2 * len(D3S), 1, figsize=(8.2, 1.45 * 2 * len(D3S) + 0.3), sharex=True)
    for j, d in enumerate(D3S):
        z = np.load(RES / f"field3d_3d_pb{d['pb_ratio']:.2f}.npz")
        for i, (k, t) in enumerate((("Mach_top", "top view (mid-height plane)"), ("Mach_side", "side view (mid-span plane)"))):
            a = ax[2 * j + i]
            im = a.imshow(z[k].T, origin="lower", extent=[0, L, 0, H], aspect="auto", cmap="viridis", vmin=0, vmax=2.1)
            a.set_yticks([]); a.set_title(f"p_b/p_in = {d['pb_ratio']:.2f}: {t}", fontsize=8)
    ax[-1].set_xlabel("x / H"); fig.colorbar(im, ax=ax, shrink=0.8, label="Mach")
    figs["3d"] = savefig(fig, HERE / "fig_duct3d.png")

# ---------------- grid study (Richardson / GCI on the sonic-front position)
gci = None
if GR:
    GR = sorted(GR, key=lambda o: o["shape"][0])
    xs3 = [o["x_sonic_final"] for o in GR]
    if all(np.isfinite(xs3)):
        gci = richardson_extrapolate(xs3[0], xs3[1], xs3[2], r=GR[1]["shape"][0] / GR[0]["shape"][0])

# ---------------- paper
STEADY_TOL = 0.5          # a case is steady when the sonic front moved less than 0.5 H over the last 5 time units
for o in SW:
    o["steady"] = o["x_sonic_last10_range"] is not None and o["x_sonic_last10_range"] < STEADY_TOL
trains = [o for o in SW if not o["no_train"]]
unst = [o for o in SW if o["unstarted"]]
started = [o for o in trains if o["steady"] and not o["unstarted"]]
unsteady = [o for o in trains if not o["steady"]]
PB_UNST = ", ".join(f"{o['pb_ratio']:.2f}" for o in unst)
PB_UNSTEADY = ", ".join(f"{o['pb_ratio']:.2f}" for o in unsteady)
P = Paper("A Shock Train in a Supersonic Duct with a Fixed-Pressure Outlet, Reproduced with PINNeAPPle",
          "Viscous compressible finite-volume simulations (HLLC/MUSCL) of Mach-2 duct flow under varying back pressure",
          report_id="Experiment 08 · Supersonic shock train")
P.abstract(
    "A commercial-code visualisation (CONVERGE) of Mach-2 flow in a channel with a fixed-pressure outlet shows a shock train: a sequence of "
    "shocks through which the core decelerates while interacting with the wall boundary layers. We reproduce this configuration with a new "
    "compressible finite-volume solver added to PINNeAPPle (HLLC fluxes, MUSCL reconstruction, viscous terms, SSP-RK2). The solver reproduces "
    f"the Rankine–Hugoniot jump of a standing normal shock at M = 2 (downstream Mach {V['M2']:.4f} vs {V['theory']['M2']:.4f}; pressure ratio "
    f"{V['p_ratio']:.3f} vs {V['theory']['p_ratio']:.1f}). In a 2-D duct at Re = 2×10<super>4</super>, the lowest back pressure "
    f"(p_b/p_in = {SW[0]['pb_ratio']:.2f}) produces no shock train; "
    + (f"in the steady cases the leading edge of the train (first centre-line sonic point) moves upstream from x/H = "
       f"{started[0]['x_sonic_final']:.2f} to {started[-1]['x_sonic_final']:.2f} as p_b/p_in rises from {started[0]['pb_ratio']:.2f} to "
       f"{started[-1]['pb_ratio']:.2f}" if started else "no steady shock train formed")
    + (f"; at {PB_UNST} the train reaches the inlet (unstart)" if unst else "")
    + (f"; at {PB_UNSTEADY} it had not settled by the end of the run" if unsteady else "") + "."
    + (f" The trend is robust but the absolute position is not grid-converged: a three-grid study at p_b/p_in = 3 gives x/H = "
       f"{', '.join(f'{v:.2f}' for v in xs3)} (coarse to fine), a grid-convergence index of {gci.gci_fine * 100:.0f} % and an extrapolated "
       f"x/H ≈ {float(gci.extrapolated_value):.1f}." if gci else "")
    + (" In a 3-D square duct at Re = 10<super>4</super>, " + "; ".join(
        f"p_b/p_in = {d['pb_ratio']:.2f} puts the leading edge at x/H = {d['x_sonic_final']:.2f}" if np.isfinite(d["x_sonic_final"]) else
        f"p_b/p_in = {d['pb_ratio']:.2f} produces no train" for d in D3S)
       + " (four walls add friction and blockage, so a given back pressure pushes the train further upstream than in 2-D)." if D3S else ""),
    ["shock train", "pseudo-shock", "supersonic duct", "back pressure", "HLLC", "finite volume", "PINNeAPPle"])
P.box("What this study shows — and what it does not", [
    "Shows: a verified solver reproducing the qualitative structure of the target picture and the dependence of the shock-train position on back pressure.",
    "Does not reproduce the CONVERGE case quantitatively: its geometry, Reynolds number, turbulence model and mesh are not known to us; our flow is "
    "laminar at Re = 2×10<super>4</super>, where the boundary layers are thicker than in a turbulent high-Re duct.",
    "The first shock-train position metric we used was wrong (see Methods); it is kept in the tables so the correction is visible."])
P.section("Background")
P.p("In an inviscid constant-area duct a normal shock at M = 2 raises the pressure by 4.5, and a steady shock can only sit in the duct at exactly that "
    "back pressure. Viscosity changes this: the boundary layer separates under the adverse pressure gradient, the single shock becomes a train of "
    "weaker, bifurcated shocks (a pseudo-shock), and the compression is spread over several duct heights. The train then settles at a position set by "
    "the back pressure, and moves upstream as it rises until it reaches the inlet (unstart) [1–3].")
P.section("Methods")
P.bullets([
    "Solver (new in PINNeAPPle: pinneapple_simulation.numerical_solvers.compressible_fv): conservative Navier–Stokes, γ = 1.4, Pr = 0.72, power-law "
    "viscosity (ω = 0.76), HLLC Riemann solver, MUSCL with minmod limiter, SSP-RK2, CFL 0.4. Boundary conditions: supersonic inflow, no-slip "
    "adiabatic walls, and a pressure outlet that fixes p where the outflow is subsonic.",
    f"Verification: Sod shock tube (first-order L1 convergence at the discontinuities) and a standing normal shock at M = 2 initialised at the exact "
    f"jump; after t = 30 the shock stayed at x = {V['shock_x_after_t30']:.2f} (initial 2.0).",
    f"Duct: L/H = 12, M_in = 2, Re = 2×10<super>4</super> (based on H and inflow), {SW[0]['shape'][0]} × {SW[0]['shape'][1]} cells, run to t = 45 "
    "flow-through units H/u_in; float32 on Apple MPS.",
    "Leading-edge position. First attempt: first x where the centre-line pressure exceeds 1.15 p_in. It fired at x ≈ 2.5 in the p_b/p_in = 2 case "
    "whose core stays supersonic to the exit, because the growing boundary layers compress the core gradually; it therefore does not locate a "
    "shock. Used instead (fixed before the other cases were examined): (i) the first centre-line point with M < 1, and (ii) the first point where "
    "the wall pressure exceeds 1.10 times that of the no-train case on the same grid. A train whose core stays supersonic (oblique train) is "
    "detected by (ii) but not (i).",
    "Grid study: three grids with refinement ratio 1.5 at p_b/p_in = 3.0; observed order, Richardson extrapolation and GCI from PINNeAPPle "
    "(pinneapple_analysis.verification.convergence)."])
P.section("Results")
P.figure(figs["fields"], "Mach number for each back-pressure ratio (label on the left). Dashed line: centre-line sonic point.", 16)
P.figure(figs["profiles"], "Left: wall pressure. Right: centre-line Mach number (dotted: M = 1).", 15.5)
P.table([["p_b / p_in", "sonic front x/H", "last-10 range", "wall-rise x/H", "first attempt x/H", "state", "wall time (min)"]] +
        [[f"{o['pb_ratio']:.2f}", fmt(o["x_sonic_final"]) if np.isfinite(o["x_sonic_final"]) else "—",
          fmt(o["x_sonic_last10_range"]) if o["x_sonic_last10_range"] is not None else "—",
          fmt(o["x_wall"]) if np.isfinite(o["x_wall"]) else "—", fmt(o["x_s_final"]),
          "unstart" if o["unstarted"] else ("no train" if o["no_train"] else ("train" if o["steady"] else "unsteady")), f"{o['seconds'] / 60:.0f}"] for o in SW],
        f"Back-pressure sweep. \"last-10 range\": spread of the sonic-front position over the last 5 time units; above {STEADY_TOL} H the case is "
        "marked unsteady. Wall time depends on the load of the shared machine at the time (up to 10× between identical-cost cases).",
        [2, 2.4, 2.2, 2.2, 2.4, 1.8, 2.2])
P.figure(figs["xs"], "Shock-train leading edge versus back pressure, with both definitions.", 9.5)
if GR:
    rows = [["grid", "sonic front x/H", "first attempt x/H"]] + [[f"{o['shape'][0]} × {o['shape'][1]}", fmt(o["x_sonic_final"]), fmt(o["x_s_final"])] for o in GR]
    if gci:
        rows += [["observed order", fmt(gci.observed_order), ""], ["extrapolated x/H", fmt(float(gci.extrapolated_value)), ""],
                 ["GCI (fine grid)", f"{gci.gci_fine * 100:.1f} %", ""], ["in asymptotic range", "yes" if gci.is_asymptotic else "no", ""]]
    P.table(rows, "Grid study at p_b/p_in = 3.0.", [4, 3.5, 3.5])
    if gci and not gci.is_asymptotic:
        P.p("The three grids are not in the asymptotic range, so the GCI is an indication of grid sensitivity rather than a calibrated error bar.")
    if gci:
        P.p(f"The leading-edge position moves by {xs3[0] - xs3[2]:.1f} duct heights between the coarsest and finest grid. The train starts where "
            "the laminar wall boundary layer separates, and the separation point depends on how well the thin layer is resolved (20 to 45 cells "
            "across the full height here). The sweep ran at an intermediate grid, so its absolute positions carry an uncertainty of the order of "
            "one duct height; the ordering of the cases, the no-train and unstart limits, and the upstream movement with back pressure are the "
            "results that do not depend on the grid in this study.")
if D3S:
    P.figure(figs["3d"], "3-D square duct, Re = 10<super>4</super>, 360 × 30 × 30 cells: top and side views, the two views of the target picture.", 16)
    d30 = next((d for d in D3S if abs(d["pb_ratio"] - 3.0) < 1e-9), None)
    lower = [d for d in D3S if d["pb_ratio"] < 3.0]
    if d30:
        P.p(f"At p_b/p_in = 3.0 the 3-D train sits almost at the inlet (x/H = {d30['x_sonic_final']:.2f}), so that case shows a nearly "
            "unstarted duct with subsonic flow downstream, not the developed train of the target picture. The four walls add friction and "
            "blockage, and at Re = 10<super>4</super> the boundary layers are thicker than in the 2-D sweep, so the same back pressure acts more "
            "strongly."
            + "".join(f" At p_b/p_in = {d['pb_ratio']:.2f} the leading edge is at x/H = {d['x_sonic_final']:.2f}"
                      + (", a train inside the duct as in the target picture." if np.isfinite(d["x_sonic_final"]) and 1 < d["x_sonic_final"] < 11 else
                         ", which still does not show a developed train." if np.isfinite(d["x_sonic_final"]) else ": no train formed.")
                      for d in lower))
P.section("Discussion")
P.p("The simulations reproduce what the picture shows: a supersonic core that decelerates through a series of shocks, with separated wall regions, "
    "whose position is controlled by the fixed outlet pressure. The metric correction is the main methodological lesson. A pressure threshold "
    "measured against the inflow value mistakes viscous compression for a shock; any automated reading of shock position from CFD (or from a "
    "surrogate trained on it) should reference the no-shock flow or a sonic condition. A surrogate of the leading-edge position over back "
    "pressure and Reynolds number was not attempted: eight 2-D cases at one Reynolds number are too few to fit and validate one honestly.")
P.references([
    "Matsuo, K., Miyazato, Y., & Kim, H.-D. (1999). Shock train and pseudo-shock phenomena in internal gas flows. <i>Progress in Aerospace Sciences</i> 35, 33–100.",
    "Waltrup, P. J., & Billig, F. S. (1973). Structure of shock waves in cylindrical ducts. <i>AIAA Journal</i> 11, 1404–1408.",
    "Gnani, F., Zare-Behtash, H., & Kontis, K. (2016). Pseudo-shock waves and their interactions in high-speed intakes. <i>Progress in Aerospace Sciences</i> 82, 36–56.",
    "Toro, E. F. (2009). <i>Riemann Solvers and Numerical Methods for Fluid Dynamics</i>, 3rd ed. Springer.",
    "Roache, P. J. (1994). Perspective: a method for uniform reporting of grid refinement studies. <i>Journal of Fluids Engineering</i> 116, 405–413.",
])
print(P.build(str(HERE / "Supersonic_ShockTrain_PINNeAPPle.pdf")))
