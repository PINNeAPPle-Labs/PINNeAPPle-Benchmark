"""Navier-Stokes existence & smoothness paper (computational probes; no claim of proof)."""
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
scan, A1, A2 = R["A_lambda_scan"], R["A_trainable_lambda"], R["A_trainable_lambda_second_branch"]
B = R["B_taylor_green"]
grids = sorted(B, key=int)
EPS_REF, T_REF = 0.0127, 9.0          # published high-resolution DNS of this case, approximate peak (refs [5, 6])
res = [s["residual_rms_fresh_points"] for s in scan]
spread = max(res) / min(res)
MONO = all(b > a for a, b in zip(res, res[1:]))
peak = {n: (B[n]["peak_eps"], B[n]["t_peak_eps"]) for n in grids}
t_end = {n: B[n]["t"][-1] for n in grids}
peak_at_end = {n: abs(peak[n][1] - t_end[n]) < 0.1 for n in grids}
finest = grids[-1]
WMAX = ", ".join(f"{max(B[n]['wmax']):.0f}" for n in grids)
interior_peak = [n for n in grids if not peak_at_end[n]]

# ---------------- figures
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
ax[0].semilogy([s["lambda"] for s in scan], res, "o-", color=CAT[0], ms=4, label="PINN residual, λ fixed")
for lam in (0.25, 0.5):
    ax[0].axvline(lam, color=INK2, lw=0.8, ls=":")
ax[0].set_xlabel("self-similar rate λ"); ax[0].set_ylabel("RMS residual (fresh points)")
ax[0].set_title("Admissible λ = 1/4, 1/2 are not singled out"); ax[0].legend(fontsize=7)
xi = np.linspace(-8, 8, 801)
ax[1].axis("off")
ax[1].text(0.0, 0.95, "Trainable λ", fontsize=9, weight="bold", va="top")
ax[1].text(0.0, 0.78, f"start 0.45 → found {A1['found']:.3f} (nearest admissible 0.5)\n"
                      f"start 0.22 → found {A2['found']:.3f} (nearest admissible 0.25)\n\n"
                      f"profile vs exact λ = 1/2 profile:\n  relative L2 = {A1['profile_rel_l2']:.2f}, "
                      f"max |ΔU| = {A1['profile_max_abs_err']:.1f}", fontsize=8, va="top")
f1 = savefig(fig, HERE / "fig_selfsimilar.png")

fig, ax = plt.subplots(1, 3, figsize=(8.6, 2.8))
for j, n in enumerate(grids):
    h = B[n]
    ax[0].plot(h["t"], h["E"], color=CAT[j], label=f"{n}³")
    ax[1].plot(h["t"], h["eps"], color=CAT[j], label=f"{n}³")
    ax[2].plot(h["t"], h["wmax"], color=CAT[j], label=f"{n}³")
ax[1].plot([T_REF], [EPS_REF], marker="x", color=INK2, ms=7, ls="none", label="DNS reference (approx.)")
ax[0].set_title("kinetic energy E"); ax[1].set_title("dissipation ε = 2νZ"); ax[2].set_title("max |ω|")
for a in ax:
    a.set_xlabel("t"); a.legend(fontsize=6.5)
f2 = savefig(fig, HERE / "fig_tgv.png")

# ---------------- paper
P = Paper("Navier–Stokes Existence and Smoothness: Two Computational Probes with PINNeAPPle",
          "Physics-informed search for self-similar blow-up on a model with a known answer, and a resolved Taylor–Green vortex",
          report_id="Experiment 06 · Millennium problems · Navier–Stokes")
P.abstract(
    "The Clay problem asks whether smooth, finite-energy solutions of the 3-D incompressible Navier–Stokes equations stay smooth for all time. "
    "No computation can settle it; we test two tools that are used in the search for candidate singularities. "
    f"(A) On inviscid Burgers, where smooth self-similar blow-up profiles exist only for λ = 1/(2i + 2), a PINNeAPPle PINN with a smoothness "
    f"penalty did not identify the admissible rates: across λ ∈ [{scan[0]['lambda']}, {scan[-1]['lambda']}] the residual "
    f"{'grew monotonically' if MONO else 'changed'} with λ (factor {spread:.1f}) and had no minimum at 0.25 or 0.5, and with λ trainable the network converged to λ = {A1['found']:.3f} and {A2['found']:.3f} instead of 0.5 and 0.25. "
    f"This is a negative result: the penalty was too weak to exclude non-smooth solutions, which exist for every λ. "
    f"(B) A pseudo-spectral DNS of the Taylor–Green vortex at Re = 1600 on {', '.join(n + '³' for n in grids)} grids keeps the energy balance "
    f"−dE/dt = ε to a relative error of at most {max(B[n]['energy_balance_rel_err'] for n in grids):.1e}; the peak dissipation on the finest grid is "
    f"{peak[finest][0]:.4f}" + (" (reached at the end of the run, so the true peak is later)" if peak_at_end[finest] else f" at t = {peak[finest][1]:.2f}")
    + f", versus about {EPS_REF} near t ≈ {T_REF:.0f} in published high-resolution DNS. Vorticity stays bounded over the run, as expected at finite Re.",
    ["Navier–Stokes", "blow-up", "self-similar solutions", "PINN", "Taylor–Green vortex", "DNS", "Millennium Prize problems"])
P.box("What this study shows — and what it does not", [
    "Does not prove or disprove global regularity; nothing computed at finite resolution and finite time can.",
    "Part A is a validation of a method on a model with a known answer, and the method failed that validation as configured. "
    "It should not be used on Navier–Stokes (or Euler) without a stronger admissibility criterion.",
    "Part B verifies the solver (energy balance, grid dependence) on a flow that is known to stay smooth; it is a baseline, not a blow-up search."])
P.section("Background")
P.p("For u<sub>t</sub> + u u<sub>x</sub> = 0 the self-similar ansatz u = (T − t)<super>λ</super> U(ξ), ξ = x/(T − t)<super>1+λ</super> gives "
    "−λU + (1 + λ)ξU′ + UU′ = 0 with U(0) = 0, U′(0) = −1. Solutions exist for every λ > 0, but they are smooth (analytic) only for "
    "λ = 1/(2i + 2); for λ = 1/2 the profile is the real root of ξ = −U − U³ [1, 2]. This is the setting in which physics-informed "
    "networks were recently used to find unstable self-similar solutions of fluid equations [3], so it is the natural first test.")
P.eq("−λU + (1 + λ) ξ U′ + U U′ = 0,   U(0) = 0,  U′(0) = −1")
P.section("Methods")
P.bullets([
    "(A) PINNeAPPle ModifiedMLP (4 × 64, 16 Fourier features); U = −ξ + ξ s² g(s²) with s = ξ/8, so oddness and U′(0) = −1 hold exactly. "
    "Loss: mean squared residual + 10<super>−6</super>·mean (U‴)² as the smoothness penalty; 6000 Adam steps then L-BFGS; residual measured on "
    "4001 fresh points. λ fixed on a grid, then λ trainable from 0.45 and from 0.22.",
    "(B) PINNeAPPle SpectralNS3D: periodic box [0, 2π)³, rotational form, 2/3-rule dealiasing, RK4 with an exact viscous integrating factor; "
    "u = (sin x cos y cos z, −cos x sin y cos z, 0), ν = 1/1600, t ∈ [0, 10]. Diagnostics: E, enstrophy Z, ε = 2νZ, max |ω| and the "
    "Beale–Kato–Majda integral ∫ max |ω| dt [4]."])
P.section("Results")
P.subsection("A. Self-similar Burgers profiles")
P.figure(f1, "(A) Left: residual with λ fixed; dotted lines mark the admissible rates 1/4 and 1/2. Right: trainable-λ outcomes.", 15.5)
P.table([["λ", "RMS residual (fresh points)"]] + [[f"{s['lambda']}", f"{s['residual_rms_fresh_points']:.2e}"] for s in scan],
        "Residual with λ fixed. A useful method would show clear minima at 0.25 and 0.5.", [4, 6])
P.p(f"The residual is small for every λ (at most {max(res):.1e}): the network fits a solution of the ODE for any rate, as it can, since "
    f"non-smooth solutions exist for all λ. The residual {'rises steadily with λ' if MONO else 'varies with λ'} and has no dip at the admissible "
    f"rates, so the smoothness penalty did not separate them. With λ trainable the rate drifted to {A1['found']:.3f} (from 0.45) and "
    f"{A2['found']:.3f} (from 0.22). The profile obtained from the first run differs from the exact λ = 1/2 profile by a relative L2 error of "
    f"{A1['profile_rel_l2']:.2f}, so it is a different solution, not an inaccurate version of the right one.")
P.subsection("B. Taylor–Green vortex at Re = 1600")
P.figure(f2, "(B) Energy, dissipation and maximum vorticity on each grid; × marks the approximate peak of published high-resolution DNS.", 16)
P.table([["grid", "wall time (s)", "peak ε", "t at peak ε", "energy-balance rel. error", "max |ω|"]] +
        [[f"{n}³", f"{B[n]['seconds']:.0f}", f"{peak[n][0]:.4f}", f"{peak[n][1]:.2f}" + (" (end)" if peak_at_end[n] else ""),
          f"{B[n]['energy_balance_rel_err']:.1e}", f"{max(B[n]['wmax']):.1f}"] for n in grids],
        "Grid dependence. \"(end)\": the maximum occurred at the last time step, so the dissipation was still rising.", [2, 2.5, 2, 2.6, 3.4, 2])
if len(grids) > 1:
    dev = {n: (peak[n][0] - EPS_REF) / EPS_REF for n in grids}
    mono = all(abs(dev[b]) < abs(dev[a]) and dev[a] * dev[b] > 0 for a, b in zip(grids, grids[1:]))
    P.p(f"The energy balance holds to within {max(B[n]['energy_balance_rel_err'] for n in grids):.1e} on every grid, so the time integration and "
        f"dissipation diagnostic are consistent. Relative to the reference peak, the computed peak dissipation deviates by "
        f"{', '.join(f'{dev[n] * 100:+.0f} % ({n}³)' for n in grids)}; "
        + ("the deviation shrinks monotonically with refinement. " if mono else
           "the deviation does not shrink monotonically: the coarsest grid under-predicts the peak and a finer one over-predicts it, "
           "the usual non-monotone behaviour of an under-resolved spectral simulation near the dissipation peak. ")
        + "Grids of 48³–96³ under-resolve this flow near its dissipation peak (published DNS uses 256³–512³), so these numbers are a "
        "resolution study, not a converged answer. The maximum vorticity also grows with resolution "
        f"({WMAX}), so it is not converged either; finer grids resolve thinner vortex sheets. "
        "It stays finite on every grid, which at Re = 1600 is expected and says nothing about the inviscid or high-Re limit.")
else:
    P.p(f"Only the {grids[0]}³ grid finished; the energy balance holds to {B[grids[0]]['energy_balance_rel_err']:.1e}, but no grid-convergence "
        "statement can be made from one grid.")
P.section("Discussion")
P.p("The self-similar probe is the more relevant tool for the Millennium question, and it is the one that failed here. What it lacks is an "
    "admissibility criterion strong enough to exclude the non-smooth solution family, for example imposing the analytic structure of U "
    "(expansion in ξ with the exact λ-dependent exponents) or demanding a smooth continuation beyond the finite ξ-window, as in [3]. We report "
    "the failure because a PINN that appears to converge for every λ would give false confidence if applied directly to Euler or "
    "Navier–Stokes. The DNS part is a verified baseline for later blow-up candidates, not evidence about regularity.")
P.references([
    "Fefferman, C. L. (2006). Existence and smoothness of the Navier–Stokes equation. In <i>The Millennium Prize Problems</i>, Clay Mathematics Institute, 57–67.",
    "Eggers, J., & Fontelos, M. A. (2009). The role of self-similarity in singularities of partial differential equations. <i>Nonlinearity</i> 22, R1–R44.",
    "Wang, Y., Lai, C.-Y., Gómez-Serrano, J., & Buckmaster, T. (2023). Asymptotic self-similar blow-up profile for three-dimensional axisymmetric Euler equations using neural networks. <i>Physical Review Letters</i> 130, 244002.",
    "Beale, J. T., Kato, T., & Majda, A. (1984). Remarks on the breakdown of smooth solutions for the 3-D Euler equations. <i>Communications in Mathematical Physics</i> 94, 61–66.",
    "Brachet, M. E., Meiron, D. I., Orszag, S. A., Nickel, B. G., Morf, R. H., & Frisch, U. (1983). Small-scale structure of the Taylor–Green vortex. <i>Journal of Fluid Mechanics</i> 130, 411–452.",
    "Wang, Z. J., et al. (2013). High-order CFD methods: current status and perspective. <i>International Journal for Numerical Methods in Fluids</i> 72, 811–845.",
])
print(P.build(str(HERE / "Millennium_NavierStokes_PINNeAPPle.pdf")))
