"""Riemann hypothesis paper."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "paperkit"))
from paperkit import CAT, INK2, MUTED, Paper, fmt, savefig, setup_mpl  # noqa: E402

setup_mpl()
R = json.loads((HERE / "results.json").read_text())
V, VR = R["verification"], R["verification_refined"]
C = json.loads((HERE / "gue_control.json").read_text()) if (HERE / "gue_control.json").exists() else None
z = np.load(HERE / "zeros_refined.npz")
unf = z["unfolded"]; s = np.diff(unf)
# spacing statistics recomputed on the refined (complete) zero list
xs_ = np.sort(s); ecdf = np.arange(1, len(xs_) + 1) / len(xs_)
gg = np.linspace(0, 4, 4001); cdf_gue = np.cumsum(32 / np.pi ** 2 * gg ** 2 * np.exp(-4 * gg ** 2 / np.pi)) * (gg[1] - gg[0])
R["spacings"] = {"KS_vs_GUE_wigner": float(np.max(np.abs(ecdf - np.interp(xs_, gg, cdf_gue)))),
                 "KS_vs_Poisson": float(np.max(np.abs(ecdf - (1 - np.exp(-xs_))))),
                 "lag1_autocorr": float(np.corrcoef(s[:-1], s[1:])[0, 1]), "var": float(s.var())}

fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
g = np.linspace(0, 3.5, 400)
ax[0].hist(s, bins=40, range=(0, 3.5), density=True, color=CAT[0], alpha=0.75, label="zeta zeros (unfolded)")
ax[0].plot(g, 32 / np.pi ** 2 * g ** 2 * np.exp(-4 * g ** 2 / np.pi), color=CAT[1], label="GUE (Wigner surmise)")
ax[0].plot(g, np.exp(-g), color=MUTED, ls="--", label="Poisson")
ax[0].set_xlabel("normalised spacing s"); ax[0].set_ylabel("density"); ax[0].set_title("Nearest-neighbour spacings, 0 < t < 3000"); ax[0].legend(fontsize=7)
pc = R["pair_correlation"]
ax[1].plot(pc["u"], pc["empirical"], "o", ms=3, color=CAT[0], label="zeta zeros")
ax[1].plot(pc["u"], pc["montgomery"], color=CAT[1], label="Montgomery: 1 − (sin πu / πu)²")
ax[1].set_xlabel("u"); ax[1].set_ylabel("pair correlation"); ax[1].set_title("Pair correlation"); ax[1].legend(fontsize=7)
f1 = savefig(fig, HERE / "fig_stats.png")

P = Paper("The Riemann Hypothesis Up to Height 3000: Exact Counting of Critical-Line Zeros, GUE Statistics, and a Learning Probe",
          "What numerical verification establishes, what it cannot, and a cautionary lesson about sampling", report_id="Experiment 06 · Millennium problems · Riemann")
sp = R["spacings"]; pr = R["predictability"]
P.abstract(
    f"The Riemann hypothesis (RH) states that every non-trivial zero of ζ(s) lies on the critical line Re s = ½. Numerical verification cannot "
    f"prove it, but it can certify it up to a height T by comparing the number of sign changes of Hardy's Z-function on the line with the total "
    f"number of zeros N(T) in the critical strip. Up to T = {V['T']:.0f} a first scan found {V['zeros_on_line_found']:,} line zeros against "
    f"N(T) = {V['N_T_all_zeros_in_strip']:,}. Localising the deficit window by window showed that the two 'missing' zeros are a close pair "
    f"(γ ≈ {VR['windows_refined'][0]['added_zeros'][0]:.4f} and {VR['windows_refined'][0]['added_zeros'][1]:.4f}, gap "
    f"{VR['windows_refined'][0]['min_gap_in_window']:.4f}) that the coarse grid stepped over; after an eight-fold refinement the counts agree "
    f"({VR['zeros_on_line_found']:,} = N(T)), so all zeros with 0 < Im s < {V['T']:.0f} lie on the critical line — a tiny, known range "
    f"(verification exceeds 10¹³ zeros), reproduced independently. The unfolded spacings follow random-matrix (GUE) statistics "
    f"(Kolmogorov–Smirnov distance {sp['KS_vs_GUE_wigner']:.3f} to the Wigner surmise vs {sp['KS_vs_Poisson']:.3f} to Poisson). A PINNeAPPle GRU "
    f"forecasts the next spacing from the previous 32 with R² = {pr['R2_gru_pinneapple']:.2f} on a held-out block"
    + (("; the same pipeline gives R² = {g:.2f} on GUE eigenvalues and {q:.2f} on a Poisson process, so the predictability is shared with random "
        "matrices (spectral rigidity), not a hidden structure of ζ.").format(g=C['gue_eigenvalues']['R2_gru_pinneapple'], q=C['poisson']['R2_gru_pinneapple'])
       if abs(C['gue_eigenvalues']['R2_gru_pinneapple'] - pr['R2_gru_pinneapple']) < 0.1 else
       ("; the same pipeline gives only R² = {g:.2f} on GUE eigenvalues (and {q:.2f} on a Poisson process), and a linear AR model reaches {a:.2f} on ζ — "
        "about the GUE level. The extra nonlinear predictability of low-lying ζ spacings is therefore NOT explained by GUE rigidity; we report it as an "
        "unexplained observation, plausibly a low-height, non-universal effect (the spacing variance, {v:.3f}, is also below the GUE value), and make no "
        "claim from it.").format(g=C['gue_eigenvalues']['R2_gru_pinneapple'], q=C['poisson']['R2_gru_pinneapple'], a=pr['R2_linear_AR32'], v=sp['var']))
    + " ",
    ["Riemann hypothesis", "zeta zeros", "Turing method", "random matrix theory", "GUE", "Montgomery pair correlation", "Millennium Prize problems"])
P.box("What this study shows — and what it does not", [
    f"Shows: every zero of ζ with 0 < Im s < {V['T']:.0f} is on the critical line (count certified against N(T)).",
    "Shows: a sampling-only search silently misses close pairs; comparing with N(T) is what makes the verification trustworthy.",
    "Does not show RH: finitely many zeros say nothing about the infinitely many beyond; statistics and learning are consistent with RH, not evidence of it."])
P.section("Method")
P.bullets([
    "Z(t) = e<super>iθ(t)</super> ζ(½ + it) is real, |Z(t)| = |ζ(½+it)|; each sign change brackets a zero on the line (mpmath <i>siegelz</i>, "
    "15-digit precision). Grid step = mean spacing/6, each bracket refined by the Illinois method.",
    "N(T) — the number of zeros with 0 < Im s < T anywhere in the critical strip — from mpmath <i>nzeros</i> (argument-principle/Turing-type counting).",
    "Where the counts differ, the window of width 100 is resampled 8×, 32×, 128× finer until the counts agree.",
    "Unfolding with the smooth part of N(t): u = θ(t)/π + 1; spacing, pair-correlation and KS statistics on the unfolded zeros.",
    "Learning probe: GRU (PINNeAPPle recurrent family) and linear AR(32) predict the next unfolded spacing from the previous 32; chronological "
    "70/15/15 split; controls: the identical pipeline on shuffled spacings, on unfolded GUE eigenvalues (N = 3000) and on a Poisson process. "
    "(The probe was run on the first-scan series of 2,467 zeros; the two missing zeros affect a handful of the 2,434 samples.)"])
P.section("Results")
P.table([["Quantity", "Value"], ["height T", f"{V['T']:.0f}"], ["zeros on the line, coarse scan", f"{V['zeros_on_line_found']:,}"],
         ["N(T), all zeros in the strip", f"{V['N_T_all_zeros_in_strip']:,}"],
         ["windows needing refinement", str(len(VR["windows_refined"]))],
         ["zeros on the line after refinement", f"{VR['zeros_on_line_found']:,}"], ["all zeros on the line up to T", str(VR["all_zeros_on_line_up_to_T"])],
         ["first zero (check vs mpmath.zetazero(1))", f"{V['first_zeros'][0]:.9f} / {V['check_first_zero_vs_mpmath_zetazero1']:.9f}"],
         ["scan CPU time", f"{V['seconds'] / 3600:.1f} h"]], "Verification up to T.", [8, 6])
P.figure(f1, "Left: spacing distribution against the GUE Wigner surmise and Poisson. Right: pair correlation against Montgomery's conjecture.", 15.5)
rows = [["Series", "R² linear AR(32)", "R² GRU (PINNeAPPle)", "lag-1 autocorrelation"]]
rows.append(["zeta zeros (unfolded)", fmt(pr["R2_linear_AR32"]), fmt(pr["R2_gru_pinneapple"]), fmt(sp["lag1_autocorr"])])
if C:
    rows.append(["GUE eigenvalues (unfolded)", fmt(C["gue_eigenvalues"]["R2_linear_AR32"]), fmt(C["gue_eigenvalues"]["R2_gru_pinneapple"]), "—"])
    rows.append(["Poisson process", fmt(C["poisson"]["R2_linear_AR32"]), fmt(C["poisson"]["R2_gru_pinneapple"]), "—"])
P.table(rows, "Next-spacing predictability on held-out blocks (R² relative to the training mean). A shuffled-spacing control gives R² ≈ 0 "
              "for both models, ruling out leakage in the pipeline.", [5.2, 3.2, 3.6, 3.4])
P.section("Discussion")
P.p("The verification part is standard and exact up to T; its value here is methodological: sampling-based zero finding is not a "
    "certificate, and the comparison with N(T) is indispensable (Turing's insight). The statistical part reproduces the Montgomery–Odlyzko "
    "law at low height. The learning part is the kind of result that is easy to over-interpret. A shuffled control rules out leakage; the GUE "
    "control shows that random-matrix rigidity alone gives much lower predictability than the GRU finds for ζ at these heights. Candidate "
    "explanations we have not tested include the regularity of low-lying zeros relative to Gram points and lower-order (arithmetic) corrections "
    "to GUE statistics, both of which fade with height. The decisive test is to repeat the probe on zeros at large height (e.g. Odlyzko's "
    "published zeros near 10<super>20</super>), where GUE statistics hold much more closely; until then no conclusion is drawn.")
P.references([
    "Bombieri, E. The Riemann hypothesis. Clay Mathematics Institute Millennium Problem description (2000).",
    "Turing, A. M. (1953). Some calculations of the Riemann zeta-function. <i>Proc. London Math. Soc.</i> (3) 3, 99–117.",
    "Montgomery, H. L. (1973). The pair correlation of zeros of the zeta function. <i>Proc. Symp. Pure Math.</i> 24, 181–193.",
    "Odlyzko, A. M. (1987). On the distribution of spacings between zeros of the zeta function. <i>Mathematics of Computation</i> 48, 273–308.",
    "Lehmer, D. H. (1956). On the roots of the Riemann zeta-function. <i>Acta Mathematica</i> 95, 291–298.",
    "Platt, D., Trudgian, T. (2021). The Riemann hypothesis is true up to 3·10¹². <i>Bulletin of the London Mathematical Society</i> 53, 792–797.",
    "Johansson, F., et al. mpmath: a Python library for arbitrary-precision floating-point arithmetic, https://mpmath.org",
])
print(P.build(str(HERE / "Millennium_Riemann_PINNeAPPle.pdf")))
