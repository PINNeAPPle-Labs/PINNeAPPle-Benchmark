"""Birch and Swinnerton-Dyer paper."""
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
S, Lr = R["summary"], R["learning"]
V = Lr["validation_selected"]
primes = np.array(Lr["primes"])
fig, ax = plt.subplots(figsize=(7.4, 3.0))
for k in ("0", "1", "2"):
    if k in Lr["murmuration_mean_ap_norm"]:
        y = np.array(Lr["murmuration_mean_ap_norm"][k])
        ax.plot(primes, y, "-", color=CAT[int(k)], lw=1.1, label=f"rank {k}")
ax.axhline(0, color=INK2, lw=0.7)
ax.set_xlabel("prime p"); ax.set_ylabel(r"mean $a_p/\sqrt{p}$ over curves"); ax.set_title("Average Frobenius traces by rank (\"murmurations\" signal)")
ax.legend()
f1 = savefig(fig, HERE / "fig_ap.png")

fig, ax = plt.subplots(figsize=(5.2, 2.6))
names = ["majority class", "Mestre–Nagao sum", "first attempt\n(ModifiedMLP)", "validation-selected\nPINNeAPPle MLP"]
vals = [V["majority_class_accuracy"], V["mestre_nagao_accuracy_same_split"], R["learning_first_attempt"]["mlp_accuracy"], V["test_accuracy"]]
ax.bar(range(4), vals, color=[CAT[3], CAT[2], CAT[7], CAT[0]], width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=7.2); ax.set_ylim(0, 1); ax.set_ylabel("rank accuracy (test)")
f2 = savefig(fig, HERE / "fig_learning.png")

P = Paper("Birch and Swinnerton-Dyer Under the Microscope: Exact Rank and Sha Checks for 1,328 Elliptic Curves, and What Learning Sees",
          "PARI/GP computation with a PINNeAPPle learning probe — evidence, not proof",
          report_id="Experiment 06 · Millennium problems · BSD")
d = S["rank_distribution_proven"]
P.abstract(
    f"The Birch and Swinnerton-Dyer (BSD) conjecture relates the rank of the Mordell–Weil group E(ℚ) of an elliptic curve to the order of "
    f"vanishing of its L-function at s = 1, and predicts the leading Taylor coefficient in terms of arithmetic invariants including the "
    f"order of the Tate–Shafarevich group Ш. We take all non-singular curves y² = x³ + ax + b with |a|, |b| ≤ 25, remove isomorphic "
    f"duplicates ({S['n_curves']:,} curves; conductors {S['conductor_range'][0]}–{S['conductor_range'][1]:,}), and compute with PARI/GP: "
    f"the analytic rank, the algebraic rank by 2-descent, and — for ranks 0 and 1 — the analytic order of Ш predicted by the BSD formula. "
    f"The algebraic rank is determined in all {S['n_rank_proven']:,} cases (distribution {d[0]}/{d[1]}/{d[2]}/{d[3]} for ranks 0/1/2/3) and equals "
    f"the analytic rank in {S['r_an_equals_rank_when_proven']:,}/{S['n_rank_proven']:,}. In all {S['sha_checked']:,} rank-0/1 cases the "
    f"predicted |Ш| is a perfect square to within {S['sha_max_dist_to_square']:.1e} (values 1, 4 and 9). A validation-selected PINNeAPPle "
    f"MLP predicts the rank from 95 normalised Frobenius traces with {100 * V['test_accuracy']:.1f}% accuracy versus "
    f"{100 * V['mestre_nagao_accuracy_same_split']:.1f}% for a Mestre–Nagao sum. For analytic rank ≤ 1 the rank equality is already a theorem "
    f"(Gross–Zagier, Kolyvagin, modularity); these computations are consistency evidence, not new mathematics.",
    ["Birch and Swinnerton-Dyer", "elliptic curves", "L-functions", "Tate–Shafarevich group", "PARI/GP", "machine learning", "Millennium Prize problems"])
P.box("What this study shows — and what it does not", [
    "Shows: for 1,328 curves, analytic and algebraic ranks agree; the BSD-predicted |Ш| is always a perfect square (as it must be when Ш is finite).",
    "Does not show BSD: ranks ≤ 1 are covered by theorems already, and for rank ≥ 2 the analytic rank is a numerical determination of the order "
    "of vanishing (a certified proof of L(E,1) = L′(E,1) = 0 is a different matter).",
    "The learning result shows that Frobenius traces carry rank information (known: Mestre–Nagao, murmurations), not that rank is computable this way."])
P.section("Background and strategy")
P.p("For E/ℚ with L-function L(E,s), BSD states ord<sub>s=1</sub> L(E,s) = rank E(ℚ) = r and "
    "L<super>(r)</super>(E,1)/r! = Ω<sub>E</sub>·Reg<sub>E</sub>·|Ш|·Π<sub>p</sub>c<sub>p</sub> / |E(ℚ)<sub>tors</sub>|². "
    "Numerically we can (i) compare the numerically determined order of vanishing with a rank certified by descent, and (ii) solve the "
    "formula for |Ш| and test whether it is a positive integer square, a strong non-trivial consistency condition.")
P.section("Methods")
P.bullets([
    "Curves: all (a, b) with |a|, |b| ≤ 25 and 4a³ + 27b² ≠ 0, reduced to minimal models; duplicates removed by (j-invariant, discriminant).",
    "Analytic rank and L<super>(r)</super>(E,1): <i>ellanalyticrank</i>. Algebraic rank: <i>ellrank</i> (2-descent bounds; 'proven' when lower = upper bound).",
    "Regulator: canonical-height matrix of generators after <i>ellsaturation</i> (so the generators span E(ℚ)/tors, not a finite-index subgroup). "
    "Ω·Πc<sub>p</sub>/|tors|²: <i>ellbsd</i>.",
    "Learning: features a<sub>p</sub>/√p for the 95 primes p < 500; target rank (0, 1, 2); 70/15/15 split; PINNeAPPle registry candidates selected on "
    "validation; baseline: Mestre–Nagao sum Σ a<sub>p</sub> log p / p with class means fitted on training curves."])
P.section("Results")
P.table([["Quantity", "Value"],
         ["curves (distinct)", f"{S['n_curves']:,}"], ["rank certified by descent", f"{S['n_rank_proven']:,}"],
         ["rank distribution 0/1/2/3", " / ".join(str(x) for x in d)],
         ["analytic rank = algebraic rank", f"{S['r_an_equals_rank_when_proven']:,} of {S['n_rank_proven']:,}"],
         ["rank-0/1 curves with |Ш| checked", f"{S['sha_checked']:,}"],
         ["max distance of |Ш|<sub>an</sub> to a square", f"{S['sha_max_dist_to_square']:.1e}"],
         ["|Ш|<sub>an</sub> values (count)", ", ".join(f"{k}: {v}" for k, v in S["sha_values_counts"].items())],
         ["CPU time", f"{S['seconds'] / 3600:.1f} h"]], "BSD checks (PARI/GP).", [7, 7])
P.figure(f1, "Average of a<sub>p</sub>/√p over curves of each rank. Higher rank goes with systematically negative traces at small p — the "
             "classical Mestre–Nagao heuristic, recently rediscovered as 'murmurations' by machine-learning studies.", 14)
P.figure(f2, "Rank prediction accuracy on held-out curves.", 10)
P.p(f"The selected model ({V['selected_on_validation']}) confusion matrix (rows = true rank 0/1/2): {V['confusion']}. The first attempt, a single "
    f"Fourier-feature ModifiedMLP with a fixed bandwidth, scored {100 * R['learning_first_attempt']['mlp_accuracy']:.1f}% — below the majority class — "
    "and is reported for transparency; it was replaced by validation-based model selection, not by looking at test scores.")
P.section("Discussion")
P.p("Every computed number agrees with BSD, as expected from the theorems that cover ranks 0 and 1 and from decades of computations "
    "(e.g. the Cremona and LMFDB databases). Where computation adds something is scale and automation; where it cannot help is the proof, "
    "which needs to control Ш and the L-function in general. The learning probe reproduces a known heuristic signal with slightly better "
    "accuracy than the hand-crafted sum; it offers no mechanism.")
P.references([
    "Wiles, A. The Birch and Swinnerton-Dyer conjecture. Clay Mathematics Institute Millennium Problem description (2000).",
    "Birch, B. J., Swinnerton-Dyer, H. P. F. (1965). Notes on elliptic curves II. <i>J. reine angew. Math.</i> 218, 79–108.",
    "Gross, B. H., Zagier, D. B. (1986). Heegner points and derivatives of L-series. <i>Inventiones Mathematicae</i> 84, 225–320.",
    "Kolyvagin, V. A. (1988). Finiteness of E(Q) and Ш(E,Q) for a subclass of Weil curves. <i>Math. USSR Izvestiya</i> 32, 523–541.",
    "Nagao, K. (1997). Q(T)-rank of elliptic curves and certain limit coming from the local points. <i>Manuscripta Mathematica</i> 92, 13–32.",
    "He, Y.-H., Lee, K.-H., Oliver, T., Pozdnyakov, A. (2022). Murmurations of elliptic curves. arXiv:2204.10140.",
    "The PARI Group. PARI/GP (cypari2 bindings), https://pari.math.u-bordeaux.fr",
])
print(P.build(str(HERE / "Millennium_BSD_PINNeAPPle.pdf")))
