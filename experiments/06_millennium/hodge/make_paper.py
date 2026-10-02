"""Hodge conjecture paper."""
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
rows = R["fermat_fourfolds"]["rows"]
m = np.array([r["m"] for r in rows]); tot = np.array([r["hodge_characters"] for r in rows])
dec = np.array([r["pair_decomposable"] for r in rows])
prime = np.array([all(k % d for d in range(2, int(k ** 0.5) + 1)) for k in m])

fig, ax = plt.subplots(figsize=(7.6, 3.1))
ax.bar(m, dec, color=CAT[0], width=0.72, label="pair-decomposable (linear-subspace cycles)")
ax.bar(m, tot - dec, bottom=dec, color=CAT[1], width=0.72, label="not pair-decomposable")
for k, t, p in zip(m, tot, prime):
    if p:
        ax.text(k, t * 1.08, "prime", ha="center", fontsize=6.5, color=INK2, rotation=90)
ax.set_yscale("log"); ax.set_xlabel("degree m of the Fermat fourfold"); ax.set_ylabel("number of Hodge characters")
ax.set_title("Hodge classes of Fermat fourfolds (Shioda characters)"); ax.set_xticks(m); ax.legend(fontsize=7, loc="upper left")
f1 = savefig(fig, HERE / "fig_fermat.png")

L = R["learning_h11"]
fig, ax = plt.subplots(figsize=(5.0, 2.6))
names = ["h¹¹ = #P factors\n(favourable guess)", "first attempt\n(ModifiedMLP, σ fixed)", "validation-selected\nPINNeAPPle MLP"]
vals = [L["baseline_h11_equals_num_P_factors_accuracy"], R["learning_h11_first_attempt"]["mlp_accuracy"], L["test_accuracy"]]
ax.bar(range(3), vals, color=[CAT[3], CAT[7], CAT[0]], width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=7.5); ax.set_ylim(0, 1); ax.set_ylabel("exact h¹¹ accuracy (test)")
ax.set_title("Learning h¹¹ of CICY threefolds")
f2 = savefig(fig, HERE / "fig_learning.png")

P = Paper("The Hodge Conjecture, Computationally: What Can Be Checked, What Cannot, and What Machine Learning Adds",
          "Topological consistency of 7,890 CICY threefolds, Shioda's Hodge characters of Fermat fourfolds, and learning h¹¹ with PINNeAPPle",
          report_id="Experiment 06 · Millennium problems · Hodge")
E = R["cicy_euler_check"]
P.abstract(
    "The Hodge conjecture asserts that on a smooth complex projective variety every rational (p,p) cohomology class is a rational "
    "combination of classes of algebraic cycles. It is a statement about the existence of cycles, not about numbers, so it is not "
    "falsifiable by the kind of numerical evidence available for the other Millennium problems. We therefore separate three tasks. "
    f"(i) Consistency: for all {E['n']:,} complete-intersection Calabi–Yau threefolds of the Candelas et al. list, the Euler characteristic "
    f"computed independently from the configuration matrix (Chern classes and intersection numbers on products of projective spaces) "
    f"equals 2(h¹¹ − h²¹) in {E['agree']:,}/{E['n']:,} cases. For threefolds the Hodge conjecture is already a theorem, so this validates data "
    "and code, not the conjecture. (ii) The conjecture proper, in a family where it is combinatorially accessible: using Shioda's character "
    "description we count the Hodge classes of the Fermat fourfolds of degree 3 ≤ m ≤ 20 and how many of them are spanned by products of "
    "linear-subspace cycles (pair-decomposable characters). For prime m and m = 4 all Hodge characters are pair-decomposable; for composite m a "
    "large fraction is not, which is exactly where other constructions are needed (the conjecture has nonetheless been proved for many such "
    "degrees, most recently for all odd m ≤ 199). (iii) Learning: a validation-selected PINNeAPPle MLP predicts h¹¹ exactly for "
    f"{100 * R['learning_h11']['test_accuracy']:.1f}% of held-out CICYs versus {100 * R['learning_h11']['baseline_h11_equals_num_P_factors_accuracy']:.1f}% "
    "for the favourable-embedding guess.",
    ["Hodge conjecture", "Calabi–Yau", "CICY", "Fermat varieties", "algebraic cycles", "machine learning", "Millennium Prize problems"])
P.box("What this study shows — and what it does not", [
    "Shows: exact topological consistency of the CICY Hodge-number list (7,890/7,890) with an independent Chern-class computation.",
    "Shows: exact counts of Hodge classes of Fermat fourfolds (3 ≤ m ≤ 20) and of the subset reached by the classical linear-cycle argument.",
    "Does not show anything for or against the Hodge conjecture: no computation here can produce or rule out an algebraic cycle. "
    "Non-pair-decomposable characters are NOT counterexamples."])
P.section("Strategy")
P.p("For each Millennium problem our protocol is the same: identify the strongest statement a computation can actually certify, compute it "
    "exactly or with quantified error, and say what remains. For the Hodge conjecture that statement is narrow. We chose (a) a large, "
    "published family of varieties to validate our algebraic-geometry pipeline against known Hodge numbers, and (b) the Fermat family, where "
    "Hodge classes have an exact combinatorial description (Shioda [3]) and the conjecture has a concrete combinatorial shadow.")
P.section("Methods")
P.subsection("Euler characteristics of CICY threefolds")
P.p("A CICY X ⊂ ℙ<super>n1</super>×…×ℙ<super>nk</super> is the common zero locus of K polynomials with multidegrees given by a k×K matrix q. "
    "Its total Chern class is c(X) = Π<sub>i</sub>(1+J<sub>i</sub>)<super>n<sub>i</sub>+1</super> / Π<sub>j</sub>(1+Σ<sub>i</sub>q<sub>ij</sub>J<sub>i</sub>), "
    "and χ(X) = ∫<sub>X</sub> c<sub>3</sub> is the coefficient of Π J<sub>i</sub><super>n<sub>i</sub></super> in c<sub>3</sub>·Π<sub>j</sub>(Σ<sub>i</sub>q<sub>ij</sub>J<sub>i</sub>). "
    "We expand c(X) to degree 3 exactly (integer arithmetic) and extract the top coefficient by memoised recursion over the linear factors. "
    "For a Calabi–Yau threefold χ = 2(h¹¹ − h²¹).")
P.subsection("Hodge characters of Fermat fourfolds")
P.p("For X<sub>m</sub>: x<sub>0</sub><super>m</super>+…+x<sub>5</sub><super>m</super> = 0, the primitive middle cohomology decomposes into "
    "one-dimensional eigenspaces indexed by α = (a<sub>0</sub>,…,a<sub>5</sub>), a<sub>i</sub> ∈ ℤ/m∖{0}, Σa<sub>i</sub> = 0; α lies in H<super>p,q</super> "
    "with p = |α| − 1, |α| = Σ⟨a<sub>i</sub>⟩/m. The rational Hodge classes are spanned by the Galois orbits of characters with |tα| = 3 for every "
    "unit t ∈ (ℤ/m)<super>×</super> [3]. A character is pair-decomposable if its entries split into pairs with a<sub>i</sub>+a<sub>j</sub> ≡ 0; these are the "
    "classes obtained from linear subspaces. Built-in checks: the method reproduces 19 primitive Hodge classes for the Fermat quartic surface "
    "(Picard number 20) and 6 for the cubic surface (Picard number 7).")
P.subsection("Learning h¹¹")
P.p("Input: the configuration matrix zero-padded to 12×15 plus the number of projective factors; output: h¹¹ as a class. 70/15/15 random split. "
    "Candidates from the PINNeAPPle model registry (GenericMLP, GenericResMLP, ModifiedMLP) are trained with early stopping; the one with "
    "the best validation accuracy is the only one scored on test. A first attempt with a single Fourier-feature ModifiedMLP (bandwidth fixed "
    "a priori) is reported as well: it failed.")
P.section("Results")
P.p(f"<b>CICY consistency.</b> {E['agree']:,} of {E['n']:,} configurations satisfy χ = 2(h¹¹ − h²¹) exactly; there are {E['n_disagree']} disagreements.")
tab = [["m", "Hodge characters", "pair-decomposable", "not pair-decomposable", "fraction reached"]] + [
    [str(r["m"]) + (" (prime)" if p else ""), f"{r['hodge_characters']:,}", f"{r['pair_decomposable']:,}", f"{r['not_pair_decomposable']:,}",
     f"{r['pair_decomposable'] / r['hodge_characters']:.3f}"] for r, p in zip(rows, prime)]
P.table(tab, "Hodge characters (dimension over ℂ of the primitive rational Hodge classes, counted character by character) of Fermat fourfolds.",
        [2.2, 3.0, 3.0, 3.4, 2.6])
P.figure(f1, "Hodge characters of Fermat fourfolds, split into those reached by products of linear cycles and the rest (log scale).", 15)
P.p("For every prime degree the linear-cycle classes exhaust the Hodge classes, recovering Shioda's theorem for prime m; for m = 4 as well. "
    "For composite m the fraction drops quickly (25% at m = 20). Those classes require other cycles (e.g. the Aoki–Shioda constructions); "
    "a 2026 computer-assisted proof covers every odd m ≤ 199 [6]. Our counts are an exact, reproducible census, not a test.")
P.figure(f2, "Exact-h¹¹ accuracy on held-out CICYs.", 10)
Lc = L["candidates"]
P.table([["Candidate (PINNeAPPle registry)", "Validation accuracy"]] + [[k, fmt(v["val_accuracy"])] for k, v in Lc.items()],
        f"Model selection on validation. Selected: {L['selected_on_validation']}; test accuracy {fmt(L['test_accuracy'])}; "
        f"on the non-favourable test CICYs (h¹¹ ≠ number of factors) {fmt(L['accuracy_on_non_favourable_test'])}.", [7, 4])
P.section("Discussion")
P.p("The honest summary is that computation can certify consistency and count Hodge classes exactly, and learning can reproduce known "
    "invariants; none of these touches the existence of algebraic cycles, which is the content of the conjecture. The one lesson for the "
    "PINNeAPPle toolchain is methodological: Fourier-feature networks designed for smooth fields generalise poorly on discrete configuration "
    "data; plain MLPs selected on validation are the right default there.")
P.references([
    "Deligne, P. The Hodge conjecture. Clay Mathematics Institute Millennium Problem description (2000).",
    "Candelas, P., Dale, A. M., Lütken, C. A., Schimmrigk, R. (1988). Complete intersection Calabi–Yau manifolds. <i>Nuclear Physics B</i> 298, 493.",
    "Shioda, T. (1979). The Hodge conjecture for Fermat varieties. <i>Mathematische Annalen</i> 245, 175–198.",
    "He, Y.-H. (2017). Deep-learning the landscape. arXiv:1706.02714.",
    "Bull, K., He, Y.-H., Jejjala, V., Mishra, C. (2018). Machine learning CICY threefolds. <i>Physics Letters B</i> 785, 65.",
    "Jumagulov, R. (2026). The Hodge conjecture for Fermat fourfolds of odd degree at most 199. arXiv:2608.18134.",
    "CICY list: http://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/",
])
print(P.build(str(HERE / "Millennium_Hodge_PINNeAPPle.pdf")))
