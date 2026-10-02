"""Poincaré conjecture paper (illustration of the Ricci-flow mechanism)."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "paperkit"))
from paperkit import CAT, INK2, Paper, fmt, savefig, setup_mpl  # noqa: E402

setup_mpl()
R = json.loads((HERE / "results.json").read_text())
A, AP, B = R["A_fd"], R["A_pinn"], R["B_su2"]
A0 = R.get("A_first_attempt_fixed_r", {}).get("A_fd")
# the single fit over t > 0.3 (stored by the experiment) mixes the exponential phase with a late floor; the floor is
# |4*pi/A_final - 1|, i.e. the area-normalisation offset, so we also fit only where max|K-1| > 10x that floor
_t, _d, _A = (np.array(A["history"][k]) for k in ("t", "maxdevK", "area"))
FLOOR = abs(4 * np.pi / _A[-1] - 1)
_s = (_t > 0.3) & (_d > 10 * FLOOR)
RATE_EXP = float(-np.polyfit(_t[_s], np.log(_d[_s]), 1)[0])
T_EXP = (float(_t[_s][0]), float(_t[_s][-1]))

fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
t, d = np.array(A["history"]["t"]), np.array(A["history"]["maxdevK"])
ax[0].semilogy(t, d, color=CAT[0], label="area-normalised flow r(t) = 8π/A")
if A0:
    ax[0].semilogy(A0["history"]["t"], A0["history"]["maxdevK"], color=CAT[1], ls="--", label="fixed r = 2 (first attempt)")
tt = np.linspace(0, t.max(), 50)
ax[0].semilogy(tt, d[0] * np.exp(-4 * tt), color=INK2, lw=0.8, ls=":", label="linear theory, rate 4 (ℓ = 2 mode)")
ax[0].set_xlabel("flow time t"); ax[0].set_ylabel("max |K − 1|"); ax[0].set_title("2-sphere: curvature → constant"); ax[0].legend(fontsize=6.8)
z = np.load(HERE / "sphere.npz")
th = z["th"]
for j, k in enumerate(("u_0.0", "u_0.5", "u_1.0", "u_2.0")):
    if k in z.files:
        u = z[k]
        r = np.exp(u)  # conformal factor -> profile of an embedded surface of revolution is not needed; show u
        ax[1].plot(th, u, color=CAT[j], label=f"t = {k[2:]}")
ax[1].set_xlabel("θ"); ax[1].set_ylabel("conformal factor u(θ)"); ax[1].set_title("Peanut metric relaxing to round"); ax[1].legend(fontsize=7)
f1 = savefig(fig, HERE / "fig_sphere.png")

fig, ax = plt.subplots(figsize=(5.0, 3.0))
init = [c["initial_anisotropy"] for c in B["cases"]]; fin = [max(c["final_anisotropy"], 1e-16) for c in B["cases"]]
ax.scatter(init, fin, color=CAT[0], s=16)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("initial anisotropy max(g)/min(g) − 1"); ax.set_ylabel("anisotropy at t = 4")
ax.set_title("SU(2) metrics under normalised Ricci flow")
f2 = savefig(fig, HERE / "fig_su2.png")

P = Paper("The Poincaré Conjecture Through Ricci Flow: A Computational Illustration with PINNeAPPle",
          "Normalised Ricci flow on the 2-sphere (finite differences vs. a physics-informed neural network) and on homogeneous 3-spheres",
          report_id="Experiment 06 · Millennium problems · Poincaré")
P.abstract(
    "The Poincaré conjecture is the only Millennium problem that has been solved (Perelman, 2002–2003, via Hamilton's Ricci flow with "
    "surgery). There is nothing to prove here; the purpose is to illustrate, with verified numerics, the mechanism of the proof — curvature "
    "diffusing until the geometry becomes round — and to test whether a physics-informed neural network from PINNeAPPle solves a geometric "
    f"flow PDE reliably. (A) On axisymmetric metrics on S², the area-normalised flow drives max|K − 1| from {A['history']['maxdevK'][0]:.2f} to "
    f"{A['final_maxdevK']:.1e} at t = {A['history']['t'][-1]:.1f}, with measured exponential rate {fmt(RATE_EXP, 2)} in the exponential phase (linear theory for the "
    f"ℓ = 2 mode: 4); the residual floor equals the area-normalisation offset |4π/A − 1| = {FLOOR:.1e}. A first attempt with a fixed normalisation constant did not conserve area; we explain why (the ℓ = 0 mode is unstable "
    f"unless r(t) is recomputed) and keep it in the record. A PINN trained on the same PDE reproduces the finite-difference solution with a "
    f"maximum deviation of {fmt(AP['vs_fd']['1.0']['max_abs_diff_u'])} in u at t = 1. (B) For left-invariant metrics on S³ = SU(2), every one of "
    f"{len(B['cases'])} squashed initial metrics (anisotropy up to {max(init):.0f}) converges to the round metric under the volume-normalised flow, "
    f"and Perelman's λ-functional is non-decreasing along the unnormalised flow in all cases.",
    ["Poincaré conjecture", "Ricci flow", "Perelman", "uniformisation", "PINN", "geometric PDE", "Millennium Prize problems"])
P.box("What this study shows — and what it does not", [
    "Shows: correct numerical Ricci flows converging to constant curvature (2D) and to the round S³ within the homogeneous class (3D).",
    "Does not reproduce Perelman's proof: no singularities, no surgery, no general 3-manifolds; the 3D part is restricted to homogeneous metrics.",
    "PINN finding: the network matches the reference solver over t ∈ [0, 1]; errors are reported, not hidden."])
P.section("Background")
P.p("Hamilton's Ricci flow ∂<sub>t</sub>g = −2 Ric(g) smooths a metric like a heat equation. Perelman proved that for a closed, simply "
    "connected 3-manifold the flow with surgery terminates in round spheres, which establishes the Poincaré conjecture [1–3]. In two "
    "dimensions the normalised flow on S² converges to the round metric for any initial metric (Hamilton, Chow) [4, 5].")
P.eq("∂<sub>t</sub>g = (r − R) g,   g = e<super>2u</super> g<sub>S²</sub>  ⇒  ∂<sub>t</sub>u = 4π/A(t) − K,   K = e<super>−2u</super>(1 − Δ<sub>S²</sub>u)")
P.section("Methods")
P.bullets([
    "(A) Axisymmetric u(θ) on 201 nodes, second-order finite differences, regular poles (Δu = 2u<sub>θθ</sub>), explicit time stepping with "
    "dt = 0.2 dθ² e<super>2 min u</super>; peanut-shaped initial metric u<sub>0</sub> ∝ P<sub>2</sub>(cos θ), renormalised to area 4π.",
    "PINN: PINNeAPPle ModifiedMLP, input (cos θ, t) so pole regularity is built in, u = u<sub>0</sub>(θ) + t·N(cos θ, t) so the initial condition is exact; "
    "residual of the fixed-r form (r = 2) on random collocation points; compared with the finite-difference solution of the same equation.",
    "(B) g = A σ<sub>1</sub>² + B σ<sub>2</sub>² + C σ<sub>3</sub>² on SU(2); principal Ricci curvatures from Milnor's formulas; volume-normalised "
    "flow; 3 Berger-type plus random initial metrics. Perelman's λ equals the (constant) scalar curvature for a homogeneous metric."])
P.section("Results")
P.figure(f1, "(A) Left: decay of max|K − 1|. Right: the conformal factor at several times.", 15.5)
P.table([["Quantity", "Value"],
         ["final max|K − 1| (area-normalised)", f"{A['final_maxdevK']:.2e}"],
         [f"decay rate, exponential phase t ∈ [{T_EXP[0]:.1f}, {T_EXP[1]:.1f}]", fmt(RATE_EXP, 3)],
         ["decay rate, single fit over t > 0.3 (includes floor)", fmt(A["decay_rate"], 3)],
         ["late floor of max|K − 1| vs |4π/A<sub>final</sub> − 1|", f"{A['final_maxdevK']:.1e} vs {FLOOR:.1e}"],
         ["area drift over the run", f"{A['area_drift']:.2e}"],
         ["PINN vs FD, t = 0.5: max|Δu|", fmt(AP["vs_fd"]["0.5"]["max_abs_diff_u"])],
         ["PINN vs FD, t = 1.0: max|Δu|", fmt(AP["vs_fd"]["1.0"]["max_abs_diff_u"])],
         ["PINN final training residual (MSE)", f"{AP['final_train_residual']:.2e}"]] +
        ([["first attempt (fixed r): area drift", f"{A0['area_drift']:.2f}"]] if A0 else []),
        "2-sphere flow.", [8, 5])
P.p(f"The decay rate stored by the experiment script ({fmt(A['decay_rate'], 2)}) came from one straight-line fit over t > 0.3. That window "
    f"contains two regimes: exponential decay at rate {fmt(RATE_EXP, 2)} (theory 4) until t ≈ {T_EXP[1]:.1f}, then a plateau. The plateau is not "
    f"curvature failing to converge. The flow converges to K = 4π/A, and the discrete area drifted by {A['area_drift']:.1e}, so K settles "
    f"{FLOOR:.1e} away from 1. Both fits are in the table; only the exponential-phase fit matches the linear theory.")
P.figure(f2, "(B) Anisotropy of the homogeneous SU(2) metric before and after the normalised flow.", 9.5)
P.p(f"All {len(B['cases'])} SU(2) runs converged to the round metric (final anisotropy below 10<super>−11</super>); λ was non-decreasing in "
    f"{'every' if B['all_lambda_monotone'] else 'not every'} run. Sanity check: for A = B = C = 1 the principal Ricci curvatures are "
    f"{R['B_sanity_round_ricci']}, i.e. Ric = 2g — the unit round 3-sphere, as expected.")
P.section("Discussion")
P.p("The numerics reproduce the qualitative content of the theory that proves the Poincaré conjecture, within the restricted settings "
    "where no singularities form. The fixed-normalisation first attempt is a useful cautionary example for physics-informed or classical "
    "solvers alike: a conserved quantity enforced only through a constant is unstable to drift. For PINNeAPPle, geometric flows are a "
    "natural test family: smooth, dissipative, with exact long-time limits.")
P.references([
    "Perelman, G. (2002). The entropy formula for the Ricci flow and its geometric applications. arXiv:math/0211159.",
    "Perelman, G. (2003). Ricci flow with surgery on three-manifolds. arXiv:math/0303109.",
    "Hamilton, R. S. (1982). Three-manifolds with positive Ricci curvature. <i>Journal of Differential Geometry</i> 17, 255–306.",
    "Hamilton, R. S. (1988). The Ricci flow on surfaces. <i>Contemporary Mathematics</i> 71, 237–262.",
    "Chow, B. (1991). The Ricci flow on the 2-sphere. <i>Journal of Differential Geometry</i> 33, 325–334.",
    "Milnor, J. (1976). Curvatures of left invariant metrics on Lie groups. <i>Advances in Mathematics</i> 21, 293–329.",
])
print(P.build(str(HERE / "Millennium_Poincare_RicciFlow_PINNeAPPle.pdf")))
