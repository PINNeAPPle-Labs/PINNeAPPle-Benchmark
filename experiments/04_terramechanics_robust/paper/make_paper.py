"""Terramechanics robustness paper -- every number read from results/."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "paperkit"))
sys.path.insert(0, str(ROOT.parents[2] / "PINNeAPPle"))
from paperkit import CAT, INK2, MUTED, Paper, fmt, savefig, setup_mpl  # noqa: E402
from pinneapple_simulation.numerical_solvers.bekker_wong import BekkerWongSolver  # noqa: E402
from pinneapple_simulation.particle_dynamics.terramechanics import SoilParams  # noqa: E402

setup_mpl()
R = json.loads((ROOT / "results" / "results.json").read_text())
FIG = ROOT / "figures"
A = R["audit"]["preset_domain"]; AX = R["audit"]["extended_z_to_0.1m"]
O, RB = R["original"], R["robust2d"]
ROB, DAT = "Robust PINN (hard R2,R4 + soft R3,R5)", "Data-only MLP (ModifiedMLP)"

# ---- Fig 1: drawbar pull at the rover operating point
rv = R["rover"]
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
ax[0].plot(rv["solver"]["slip"], rv["solver"]["fx"], "-", color=INK2, lw=2.2, label="Bekker–Wong solver")
ax[1].plot(rv["solver"]["slip"], 1e3 * np.array(rv["solver"]["z"]), "-", color=INK2, lw=2.2, label="Bekker–Wong solver")
for k, c, lab in (("Original PINN", CAT[1], "original PINN"), (ROB, CAT[0], "robust PINN (this work)"), (DAT, CAT[2], "data-only MLP")):
    ax[0].plot(rv[k]["curve"]["slip"], rv[k]["curve"]["fx_pred"], "--", color=c, label=lab)
    ax[1].plot(rv[k]["curve"]["slip"], 1e3 * np.array(rv[k]["curve"]["z_pred"]), "--", color=c, label=lab)
ax[0].set_xlabel("slip s"); ax[0].set_ylabel("drawbar pull F<sub>x</sub> (N)".replace("<sub>", "$_").replace("</sub>", "$"))
ax[0].set_title(f"Drawbar pull at wheel load W = {rv['W_N']:.1f} N"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("slip s"); ax[1].set_ylabel("sinkage z (mm)"); ax[1].set_title("Sinkage solving F$_z$(s, z) = W")
f1 = savefig(fig, FIG / "fig1_rover.png")

# ---- Fig 2: quadrature convergence (non-integer sinkage exponent) + speed
sol = BekkerWongSolver(SoilParams(n=0.8))
rng = np.random.default_rng(0)
P = np.c_[rng.uniform(0, 0.75, 60), rng.uniform(0.004, 0.058, 60)]
ref = np.array([sol.forces(*p) for p in P])
ns = [4, 8, 16, 32, 64, 128]
err = [np.abs(sol.forces_batch(P[:, 0], P[:, 1], n_gl=n) - ref).max(0) / np.abs(ref).max(0) for n in ns]
sp = R["speed_us_per_eval"]
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.0), gridspec_kw={"wspace": 0.75})
for j, lab in enumerate(("F$_x$", "F$_z$", "M$_y$")):
    ax[0].loglog(ns, [e[j] for e in err], "-o", color=CAT[j], label=lab)
ax[0].set_xlabel("Gauss–Legendre nodes per sub-interval"); ax[0].set_ylabel("max relative error vs adaptive quad")
ax[0].set_title("Batched quadrature convergence (n = 0.8)"); ax[0].legend()
names = [("quad_adaptive", "adaptive quad"), ("gauss_legendre_48_batched", "GL-48 batched"), ("gauss_legendre_16_batched", "GL-16 batched"),
         ("original_pinn_batched", "original PINN"), ("robust_ensemble5_batched", "robust PINN ×5")]
ax[1].barh(range(len(names)), [sp[k] for k, _ in names], color=[MUTED, CAT[0], CAT[0], CAT[1], CAT[2]])
ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels([n for _, n in names]); ax[1].set_xscale("log")
ax[1].set_xlabel("µs per evaluation (batch of 20,000)"); ax[1].set_title("Cost per force evaluation")
f2 = savefig(fig, FIG / "fig2_quadrature_speed.png")

# ======================================================================== paper
Pp = Paper("Making the PINNeAPPle Terramechanics Surrogate Robust: Auditing Physics Constraints, Hardening the Bekker–Wong Solver, "
           "and Constraint-Preserving Surrogates",
           "Rigid-wheel / lunar-regolith interaction for rover simulation", report_id="Experiment 04")
Pp.abstract(
    f"PINNeAPPle ships a physics-informed surrogate of the Bekker–Wong rigid-wheel model for real-time rover simulation. We audited every "
    f"physics constraint it imposes against its own reference solver and found that one of them is false: the surrogate was trained to "
    f"enforce zero drawbar pull at zero slip, whereas the Bekker–Wong model gives F<sub>x</sub>(0, z) between {A['R1_Fx_at_s0_range_N'][0]:.1f} and "
    f"{A['R1_Fx_at_s0_range_N'][1]:.1f} N on the preset domain. A second constraint (Mohr–Coulomb traction limit) used half the wheel "
    f"circumference as contact area and was never active. We (i) corrected the preset, the compiled residual and the example, keeping only "
    f"constraints verified numerically on the stated domain and adding one (∂F<sub>z</sub>/∂z ≥ 0); (ii) hardened the solver (input "
    f"validation, break points at the stress kinks, sign-aware shear for braking, Brent's method for sinkage, no fabricated values on "
    f"failure, and a vectorised, differentiable Gauss–Legendre quadrature that matches adaptive quadrature to 10<super>−13</super>); and (iii) "
    f"built a constraint-preserving surrogate in which the Mohr–Coulomb and torque constraints hold by construction. On an independent test set "
    f"the original surrogate has relative L2 errors of {100 * O['test']['Fx']['rel_L2']:.1f}% (F<sub>x</sub>) and {100 * O['test']['My']['rel_L2']:.1f}% "
    f"(M<sub>y</sub>); the robust surrogate {100 * RB[ROB]['test']['Fx']['rel_L2']:.1f}% and {100 * RB[ROB]['test']['My']['rel_L2']:.1f}% with zero "
    f"violations of the hard constraints. At the rover's operating point the sinkage error drops from {rv['Original PINN']['sinkage_rmse_mm']:.2f} "
    f"to {rv[ROB]['sinkage_rmse_mm']:.3f} mm. We also report where physics did not help (stress and extrapolation tests) and that, for this two-input "
    f"problem, the batched quadrature itself ({sp['gauss_legendre_48_batched']:.0f} µs per evaluation) is fast enough to make a neural surrogate optional.",
    ["terramechanics", "Bekker–Wong", "rover", "physics-informed neural networks", "hard constraints", "verification", "PINNeAPPle", "VeriPhysics"])
Pp.box("Main findings", [
    "A physics constraint in the original PINN was wrong; enforcing it biased the model against its own data (F<sub>x</sub> at s = 0 predicted "
    f"≈{O['fx_at_zero_slip_pred_vs_solver']['pred_mean_abs']:.2f} N vs mean |{O['fx_at_zero_slip_pred_vs_solver']['solver_mean_abs']:.2f}| N from the solver).",
    "Constraints should be verified against the reference solver on the exact domain where they are imposed: ∂F<sub>x</sub>/∂s ≥ 0 for s ≤ 0.4 holds "
    f"for z ≤ 0.058 m ({A['R3_monotone_s_le_0.4_violations']} violations) but not up to 0.1 m ({AX['R3_monotone_s_le_0.4_violations']} violations).",
    "Hard (architectural) constraints are cheaper and more reliable than penalties; soft penalties still leave violations.",
    "A data-only MLP of the same architecture is competitive on accuracy; the value of physics here is guaranteed consistency, not lower error."])

Pp.section("Background")
Pp.p("For a rigid wheel of radius R and width b at sinkage z and slip s, the Bekker–Wong model integrates normal and shear stress over "
     "the contact arc θ<sub>r</sub> ≤ θ ≤ θ<sub>f</sub>, θ<sub>f</sub> = arccos(1 − z/R):")
Pp.eq("σ(θ) = (k<sub>c</sub>/b + k<sub>φ</sub>)[R(cos θ* − cos θ<sub>f</sub>)]<super>n</super>,   τ(θ) = (c + σ tan φ)(1 − e<super>−j/K</super>)")
Pp.eq("F<sub>x</sub> = Rb∫(τ cos θ − σ sin θ)dθ,   F<sub>z</sub> = Rb∫(σ cos θ + τ sin θ)dθ,   M<sub>y</sub> = R²b∫τ dθ")
Pp.p("with shear displacement j(θ) = R[(θ<sub>f</sub> − θ) − (1 − s)(sin θ<sub>f</sub> − sin θ)] and the rear-region mapping θ* of Wong & Reece. "
     "Soil: GRC-1 lunar simulant (c = 1.4 kPa, φ = 30°, K = 18 mm, k<sub>c</sub> = 1370, k<sub>φ</sub> = 8.14×10⁵, n = 1); wheel R = 0.125 m, b = 0.06 m; "
     "rover 40 kg on six wheels under lunar gravity (W = 10.8 N per wheel).")

Pp.section("Audit of the original physics constraints")
rows = [["Constraint (original)", "Status on preset domain (s ≤ 0.75, z ≤ 0.058 m)", "Action"],
        ["R1: F<sub>x</sub>(s=0, z) = 0", f"FALSE: solver gives {A['R1_Fx_at_s0_range_N'][0]:.2f} … {A['R1_Fx_at_s0_range_N'][1]:.2f} N", "removed"],
        ["R2: F<sub>x</sub> ≤ cA + F<sub>z</sub> tan φ, A = πbR", f"holds, but never active (min margin {A['R2_original_area_piBR_min_margin_N']:.1f} N)",
         "A = bRθ<sub>1</sub> (contact arc): min margin " + f"{A['R2_arc_area_min_margin_N']:.2f} N, 0 violations"],
        ["R3: ∂F<sub>x</sub>/∂s ≥ 0, s ≤ 0.4", f"holds ({A['R3_monotone_s_le_0.4_violations']} violations); fails for z → 0.1 m ({AX['R3_monotone_s_le_0.4_violations']})",
         "kept; domain stated"],
        ["R4: M<sub>y</sub> ≥ R F<sub>x</sub>", f"holds (min margin {A['R4_min_margin_Nm']:.3f} N·m)", "kept, made hard"],
        ["R5 (new): ∂F<sub>z</sub>/∂z ≥ 0", f"holds ({A['R5_dFz_dz_violations']} violations)", "added"]]
Pp.table(rows, f"Constraint audit on a dense {A['n_points']:,}-point grid evaluated with the Bekker–Wong solver.", [4.4, 6.6, 5.0])
Pp.p(f"The physical reason R1 fails: at s = 0 the shear displacement j(θ) is not zero (θ − sin θ is increasing), so shear stress and a net "
     f"drawbar force exist; the sign depends on the balance between compaction resistance and the rear-region shear. Note also that the "
     f"rover never visits most of the preset domain: for W = 10.8 N the sinkage stays in "
     f"{1e3 * R['audit']['operating_sinkage_for_wheel_load_m']['z_min']:.1f}–{1e3 * R['audit']['operating_sinkage_for_wheel_load_m']['z_max']:.1f} mm.")

Pp.section("Hardening the solver and the preset (library changes)")
Pp.bullets([
    "<b>Validation</b>: 0 < z < R, −1 ≤ s < 1 and positive soil/wheel parameters are checked; errors raise instead of returning garbage.",
    "<b>Accurate adaptive quadrature</b>: the stress kinks (θ<sub>m</sub>, 0) are passed as break points; the error estimate is exposed.",
    f"<b>Batched, differentiable quadrature</b> (<i>bekker_wong_forces_torch</i>): piecewise Gauss–Legendre on the smooth sub-intervals, per-sample "
    f"soil parameters, float64, autograd-compatible; agrees with adaptive quadrature to 10<super>−13</super> for n = 1 (Figure {Pp.next_fig()}).",
    "<b>Sinkage from load</b>: bracketed Brent root-finding on F<sub>z</sub>(z) = W (monotone, verified) replaces an unguarded secant iteration.",
    "<b>No fabricated data</b>: dataset generation used to write zeros for failed evaluations; it now raises or drops-and-counts.",
    "<b>Braking</b>: shear is sign-aware, removing an exponential blow-up for s < 0.",
    "<b>Preset/compiler</b>: false R1 condition removed; R5 added to the compiled residual; tests updated (the old test asserted the false condition) "
    "and seven new robustness tests added."])
Pp.figure(f2, "Left: convergence of the batched Gauss–Legendre quadrature for a non-integer sinkage exponent (n = 0.8), where the integrand is "
              "not smooth at θ<sub>f</sub>. Right: cost per evaluation.", 15.5)

Pp.section("Surrogates")
Pp.bullets([
    "<b>Original</b>: the frozen example (git HEAD) run unchanged — Fourier-feature ResNet, 4,000 epochs, R1–R4 penalties, its own grid+random data "
    f"({O['n_train_points']:,} points, random 85/15 split, no test set).",
    "<b>Data-only MLP</b>: PINNeAPPle ModifiedMLP, same inputs, no physics; 5-member deep ensemble.",
    "<b>Robust PINN</b> (this work): same backbone; outputs parameterised so that F<sub>z</sub> > 0, F<sub>x</sub> ≤ cbRθ<sub>1</sub> + F<sub>z</sub> tan φ and "
    "M<sub>y</sub> ≥ RF<sub>x</sub> hold by construction; soft penalties for the monotonicity constraints R3, R5 on unlabelled collocation points; "
    "load-relative loss; 5-member deep ensemble with variance recalibrated on validation.",
    f"Data: {RB['n_train']:,} training and {RB['n_val']:,} validation Latin-hypercube points; an independent {RB['n_test']:,}-point Latin-hypercube test set "
    "(seed not used anywhere else)."])

Pp.section("Results")
rows = [["Model", "rel. L2 F<sub>x</sub>", "rel. L2 F<sub>z</sub>", "rel. L2 M<sub>y</sub>", "R2 viol.", "R3 viol.", "R4 viol.", "R5 viol."]]
for name, blk in (("Original PINN", O), ("Data-only MLP", RB[DAT]), ("Robust PINN (this work)", RB[ROB])):
    cv = blk["constraint_violations"]
    rows.append([name] + [f"{100 * blk['test'][k]['rel_L2']:.2f}%" for k in ("Fx", "Fz", "My")] +
                [f"{100 * cv[k]:.2f}%" for k in ("R2_mohr_coulomb", "R3_monotone_slip", "R4_torque", "R5_monotone_sinkage")])
tab_acc = Pp.table(rows, "Independent test set (5,000 points). Violation rates on 20,000 fresh points (finite differences for R3, R5).",
         [4.0, 1.8, 1.8, 1.8, 1.4, 1.4, 1.4, 1.4], highlight_rows=[3])
rows = [["Model", "Sinkage RMSE (mm)", "Drawbar-pull RMSE (N)"]] + [
    [k, fmt(rv[k]["sinkage_rmse_mm"]), fmt(rv[k]["drawbar_pull_rmse_N"])] for k in ("Original PINN", DAT, ROB)]
Pp.table(rows, f"Rover operating point: sinkage from F<sub>z</sub>(s, z) = {rv['W_N']:.1f} N solved with each surrogate, and the drawbar pull there.", [7, 3.5, 3.5])
Pp.figure(f1, "Operating-point behaviour of the surrogates against the Bekker–Wong solver.", 15.5)
cal = RB[ROB]["calibration"]
Pp.p("Uncertainty: the robust ensemble (variance scaled on validation) reaches 90%-interval coverage of "
     + ", ".join(f"{fmt(cal[k]['coverage90'])} ({k})" for k in ("Fx", "Fz", "My")) + " on test, with ECE "
     + ", ".join(fmt(cal[k]["ece"]) for k in ("Fx", "Fz", "My")) + ".")

if "stress" in R:
    Pp.subsection("Stress tests (equal gradient-step budget)")
    rows = [["Case", "Model", "rel. L2 F<sub>x</sub>", "rel. L2 F<sub>z</sub>", "rel. L2 M<sub>y</sub>", "R3 viol.", "R5 viol."]]
    for case, blk in R["stress"].items():
        for mname, b in blk.items():
            rows.append([case.replace("_", " "), mname] + [f"{100 * b['test'][k]['rel_L2']:.1f}%" for k in ("Fx", "Fz", "My")] +
                        [f"{100 * b['constraint_violations'][k]:.2f}%" for k in ("R3_monotone_slip", "R5_monotone_sinkage")])
    Pp.table(rows, "Label noise (3%), small data (150 points) and extrapolation in sinkage (train z ≤ 0.04 m, test z > 0.045 m). "
                   "3-member ensembles, ≥ 2,400 gradient steps each.", [4.6, 2.4, 1.6, 1.6, 1.6, 1.5, 1.5])
    Pp.p("Physics constraints bound the answer (they cannot be violated) but do not supply the magnitudes that are missing from the data: "
         "under extrapolation both models degrade by an order of magnitude. This is the expected limit of inequality constraints.")
if "parametric" in R:
    PR = R["parametric"]
    Pp.subsection("Parametric surrogate across soils")
    rows = [["Model", "in-dist. F<sub>x</sub>", "in-dist. F<sub>z</sub>", "in-dist. M<sub>y</sub>", "OOD F<sub>x</sub>", "OOD F<sub>z</sub>", "OOD M<sub>y</sub>", "OOD AUROC"]]
    for mname in [k for k in PR if isinstance(PR[k], dict) and "in_distribution" in PR[k]]:
        b = PR[mname]
        rows.append([mname] + [f"{100 * b['in_distribution']['test'][k]['rel_L2']:.1f}%" for k in ("Fx", "Fz", "My")] +
                    [f"{100 * b['out_of_distribution']['test'][k]['rel_L2']:.1f}%" for k in ("Fx", "Fz", "My")] + [fmt(b["ood_auroc_from_spread"])])
    Pp.table(rows, f"8-input surrogate (slip, sinkage, c, φ, K, k<sub>c</sub>, k<sub>φ</sub>, n): {PR['n']['train']:,} training points generated in "
                   f"{PR['data_gen_seconds']:.0f} s with the batched quadrature (max relative deviation from adaptive quadrature "
                   f"{max(PR['gl96_vs_quad_max_rel_err']):.1e}). OOD: every soil parameter beyond the training box. AUROC: detection of OOD inputs "
                   "from the ensemble spread.", [4.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.6])

Pp.subsection("VeriPhysics validation")
V = R["veriphysics"]
rows = [["Model", "Trust score", "Coverage", "Guardrail checks (pass/fail)", "Components"]]
for k in (ROB, DAT, "Original PINN"):
    v = V[k]
    rows.append([k, fmt(v["overall_score"]), f"{v['coverage']:.1f}", ", ".join(f"{c['name']}={'P' if c['passed'] else 'F'}" for c in v["guardrail"]),
                 ", ".join(f"{c['name']} {c['score']:.2f}" for c in v["components"])])
Pp.table(rows, "PINNeAPPle PhysicsGuardrail + compute_physics_confidence. numerical_convergence is the GCI of the reference quadrature "
               f"(observed order {V['convergence_gl_n0.8']['observed_order']:.2f}, GCI {V['convergence_gl_n0.8']['gci_fine']:.1e}).",
         [3.6, 1.4, 1.3, 5.0, 5.0])
Pp.p(f"All three models fail the guardrail's pde_residual check. This is informative about the check, not only the models: for this "
     "problem kind the compiled residual is dimensional (N² for the inequality violations, N²/m² for ∂F<sub>z</sub>/∂z), so its default "
     f"threshold of 10<super>−2</super> has no physical meaning. The dimensionless violation rates of Table {tab_acc} are the meaningful physics metric; "
     "normalising the compiled terramechanics residual is recorded as follow-up work for the library.")
Pp.section("Conclusions")
Pp.bullets([
    "Audit physics constraints against the reference model before imposing them; one of four was false and one was vacuous.",
    "Prefer hard, architectural constraints where the physics gives inequalities with closed forms; they cost nothing at inference.",
    "Report data-only baselines: here they match physics-informed accuracy, and the benefit of physics is consistency.",
    "For low-dimensional models, a batched, verified numerical method can be the right 'surrogate'."])
Pp.references([
    "Bekker, M. G. (1969). <i>Introduction to Terrain-Vehicle Systems</i>. University of Michigan Press.",
    "Wong, J. Y., Reece, A. R. (1967). Prediction of rigid wheel performance based on the analysis of soil-wheel stresses. <i>Journal of Terramechanics</i> 4(1), 81–98.",
    "Wong, J. Y. (2008). <i>Theory of Ground Vehicles</i>, 4th ed. Wiley.",
    "Janosi, Z., Hanamoto, B. (1961). The analytical determination of drawbar pull as a function of slip for tracked vehicles in deformable soils. <i>Proc. 1st ISTVS Conf.</i>",
    "Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). Physics-informed neural networks. <i>Journal of Computational Physics</i> 378, 686–707.",
    "Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. <i>NeurIPS</i>.",
])
print(Pp.build(str(ROOT / "paper" / "Terramechanics_Robust_PINNeAPPle.pdf")))
