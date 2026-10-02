"""Yang-Mills mass gap paper (lattice evidence at finite spacing)."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "paperkit"))
from paperkit import CAT, INK2, MUTED, Paper, fmt, savefig, setup_mpl  # noqa: E402

setup_mpl()
R = json.loads((HERE / "results.json").read_text())["runs"]
# Lucini & Teper, JHEP 0106:050 (2001), Table 1 (SU(2)): beta -> (a sqrt(sigma), a m_0++)
LT = {2.30: (0.3108, 0.0017, 1.090, 0.033), 2.40: (0.2634, 0.0014, 0.953, 0.019)}
LT_CONT = (3.844, 0.061)

fig, ax = plt.subplots(1, 3, figsize=(8.6, 2.9), sharey=True)
for j, r in enumerate(R):
    m, e = np.array(r["m_eff"][0], float), np.array(r["m_eff"][1], float)
    t = np.arange(len(m))
    ok = np.isfinite(m) & np.isfinite(e) & (e < 1.0)
    ax[j].errorbar(t[ok], m[ok], yerr=e[ok], fmt="o", color=CAT[0], ms=4, capsize=2, label="this work, $a\\,m_{\\rm eff}(t)$")
    if r["beta"] in LT:
        ax[j].axhspan(LT[r["beta"]][2] - LT[r["beta"]][3], LT[r["beta"]][2] + LT[r["beta"]][3], color=CAT[1], alpha=0.35,
                      label="Lucini–Teper $a\\,m_{0^{++}}$")
    ax[j].set_title(f"β = {r['beta']}  ({r['L']}³×{r['T']})"); ax[j].set_xlabel("t / a"); ax[j].set_ylim(0, 2.2)
ax[0].set_ylabel("effective mass (lattice units)"); ax[0].legend(fontsize=7, loc="lower left")
f1 = savefig(fig, HERE / "fig_meff.png")

fig, ax = plt.subplots(figsize=(4.6, 3.0))
for j, r in enumerate(R):
    W, We = np.array(r["wilson_square_loops_R123"][0]), np.array(r["wilson_square_loops_R123"][1])
    ax.errorbar([1, 2, 3], -np.log(W), yerr=We / W, fmt="-o", color=CAT[j], ms=4, capsize=2, label=f"β = {r['beta']}")
ax.set_xlabel("R (square R×R loop)"); ax.set_ylabel("−ln W(R,R)"); ax.set_title("Wilson loops (confinement: area law)"); ax.legend()
f2 = savefig(fig, HERE / "fig_wilson.png")

P = Paper("Probing the Yang–Mills Mass Gap on the Lattice: SU(2) Monte Carlo Evidence at Finite Spacing",
          "A reproducible PINNeAPPle-Labs baseline — what lattice numerics can and cannot say about a Millennium Prize problem",
          report_id="Experiment 06 · Millennium problems · Yang–Mills")
r24 = [r for r in R if r["beta"] == 2.4][0]
P.abstract(
    "The Yang–Mills existence and mass-gap problem asks for a rigorous construction of four-dimensional quantum Yang–Mills theory "
    "with a strictly positive mass gap. No numerical computation can settle it. What numerics can do is measure, at finite lattice "
    "spacing, whether the lightest excitation of the lattice-regularised theory is massive and whether its mass scales like a physical "
    "quantity as the spacing shrinks. We implement SU(2) pure-gauge Monte Carlo (Kennedy–Pendleton heat bath with over-relaxation, "
    "vectorised in PyTorch) on 8³×16 lattices at β = 2.3, 2.4 and 2.5, and measure the plaquette, square Wilson loops and the 0⁺⁺ "
    f"glueball correlator with APE-smeared operators. The plaquette agrees with the literature to four digits (⟨P⟩ = {r24['plaquette'][0]:.4f} at β = 2.4). "
    "Wilson loops follow an area law (confinement). The glueball effective mass is non-zero at all three couplings; at β = 2.4 it is "
    f"{r24['m_eff'][0][1]:.2f} ± {r24['m_eff'][1][1]:.2f} at t = a (biased upward by excited states relative to the published 0.953(19)) and "
    f"{r24['m_eff'][0][2]:.2f} ± {r24['m_eff'][1][2]:.2f} at t = 2a (consistent, but with large errors). Our small-loop string tension is ≈2× "
    "the published one because Coulomb terms "
    "are not suppressed at R ≤ 3. We report these as limitations. The results are lattice evidence consistent with a gap; they are not, "
    "and cannot be, a construction of the continuum theory.",
    ["Yang–Mills", "mass gap", "lattice gauge theory", "SU(2)", "glueball", "Monte Carlo", "Millennium Prize problems"])
P.box("What this study shows — and what it does not", [
    "Shows: a correct SU(2) lattice Monte Carlo (plaquette matches published values to ~10⁻⁴) with confinement (area law) and a non-zero "
    "0⁺⁺ effective mass at three couplings.",
    "Does not show: the existence of continuum Yang–Mills theory, a rigorous gap, or a continuum-extrapolated glueball mass. "
    "Statistics (800 measurements per coupling) and volume (8³×16) are far below state-of-the-art.",
    "Does not use a neural model: the physics here is Monte Carlo; the PINNeAPPle role is the reproducible experiment harness."])
P.section("Problem statement and strategy")
P.p("The Clay problem [1] requires proving that, for any compact simple gauge group G, quantum Yang–Mills theory on ℝ⁴ exists (satisfying "
    "axioms at least as strong as Wightman's or Osterwalder–Schrader's) and has a mass gap Δ > 0. The only first-principles numerical "
    "approach to the non-perturbative theory is Wilson's lattice regularisation [2]: the Euclidean path integral on a hypercubic lattice "
    "with spacing a, sampled by Markov-chain Monte Carlo. A mass gap manifests as exponential decay of zero-momentum correlators, "
    "C(t) ∼ e<super>−m t</super>, with m > 0; the continuum question is whether m/√σ (σ the string tension) tends to a finite non-zero limit as a → 0.")
P.eq("S[U] = β Σ<sub>p</sub> (1 − ½ Re Tr U<sub>p</sub>),   β = 4/g²")
P.section("Methods")
P.bullets([
    "Links U<sub>μ</sub>(x) ∈ SU(2) stored as unit quaternions; checkerboard parity and direction are updated together (all links of the same "
    "parity and direction are conditionally independent).",
    "One update = Kennedy–Pendleton heat bath [3] + one over-relaxation step (U → V†U†V†, which preserves the local action), followed by re-unitarisation.",
    "Cold start, 200 thermalisation updates, then 800 measurements separated by 2 updates, per coupling.",
    "Observables: plaquette; square Wilson loops W(R,R), R = 1..3, averaged over planes; zero-momentum 0⁺⁺ operator from spatial plaquettes "
    "after 4 APE smearing steps (α = 0.5); connected correlator C(t) and effective mass m<sub>eff</sub>(t) = ln[C(t)/C(t+1)].",
    "Errors: jackknife with 20 blocks. Reference values: Lucini & Teper [4], Table 1 (SU(2), larger lattices, ~10⁵ sweeps)."])
P.section("Results")
rows = [["β", "Lattice", "⟨P⟩ (this work)", "a²σ (small-loop estimate)", "a²σ reference [4]", "a m<sub>eff</sub>(t=a) / (t=2a)", "a m<sub>0⁺⁺</sub> reference [4]", "CPU s"]]
for r in R:
    ref = LT.get(r["beta"])
    m1, e1, m, e = r["m_eff"][0][1], r["m_eff"][1][1], r["m_eff"][0][2], r["m_eff"][1][2]
    rows.append([str(r["beta"]), f"{r['L']}³×{r['T']}", f"{r['plaquette'][0]:.5f}({r['plaquette'][1] * 1e5:.0f})",
                 f"{fmt(r['string_tension_a2sigma_from_squares'][0])} ± {fmt(r['string_tension_a2sigma_from_squares'][1])}",
                 fmt(ref[0] ** 2, 4) if ref else "—", f"{m1:.2f}±{e1:.2f} / " + (f"{m:.2f}±{e:.2f}" if np.isfinite(e) else "n/a"),
                 f"{ref[2]} ± {ref[3]}" if ref else "—", f"{r['seconds']:.0f}"])
P.table(rows, "Summary per coupling. The reference string tensions come from Polyakov-loop flux-tube masses on larger lattices; "
              "ours from −ln[W(3)W(1)/W(2)²]/2, which cancels the perimeter term but not the Coulomb term.",
        [0.9, 1.5, 2.1, 2.6, 1.9, 2.0, 2.2, 1.2])
P.figure(f1, "Effective mass of the 0⁺⁺ glueball correlator versus Euclidean time. Band: published a·m<sub>0⁺⁺</sub> [4] where available. "
             "Points with jackknife error > 1 are omitted (noise dominates beyond t ≈ 3a with our statistics).", 15.8)
P.figure(f2, "−ln W(R,R) versus R. Growth close to R² (area law) signals confinement; the curvature increases as β decreases (larger a).", 9.5)
P.section("Discussion")
P.p("Three things are established at the level of this computation: the sampler is correct (plaquettes), the theory confines on these "
    "lattices (area law), and the lightest 0⁺⁺ state has a non-zero mass in lattice units. What is not established — and what the "
    "Millennium problem is about — is the continuum limit and its mathematical existence. Even the numerical continuum ratio m/√σ, "
    f"known to be {LT_CONT[0]}({int(LT_CONT[1] * 1000)}) for SU(2) [4], would require larger volumes, variational operator bases and ~100× our statistics to reproduce.",
    "A physics-AI angle that is actually useful here is not a neural surrogate of the answer but faster sampling: normalising-flow and "
    "machine-learned samplers for lattice gauge theory are an active research direction, and would be the next PINNeAPPle experiment.")
P.references([
    "Jaffe, A., Witten, E. Quantum Yang–Mills theory. Clay Mathematics Institute Millennium Problem description (2000).",
    "Wilson, K. G. (1974). Confinement of quarks. <i>Physical Review D</i> 10, 2445.",
    "Kennedy, A. D., Pendleton, B. J. (1985). Improved heatbath method for Monte Carlo calculations in lattice gauge theories. <i>Physics Letters B</i> 156, 393.",
    "Lucini, B., Teper, M. (2001). SU(N) gauge theories in four dimensions: exploring the approach to N = ∞. <i>JHEP</i> 0106:050 (arXiv:hep-lat/0103027).",
    "Creutz, M. (1980). Monte Carlo study of quantized SU(2) gauge theory. <i>Physical Review D</i> 21, 2308.",
])
print(P.build(str(HERE / "Millennium_YangMills_MassGap_PINNeAPPle.pdf")))
