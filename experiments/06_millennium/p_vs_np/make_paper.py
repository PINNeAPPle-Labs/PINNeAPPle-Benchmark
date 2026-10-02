"""P vs NP paper: empirical hardness of random 3-SAT (what experiments can and cannot say)."""
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
PT = R["phase_transition"]
SC = R.get("scaling_alpha_4.26")
AN = R.get("analog_solver")
LS = R.get("learning_satisfiability_N100")
LS0 = R.get("learning_satisfiability_N100_first_attempt")
ALPHA_C = 4.267                      # 1-RSB threshold for random 3-SAT (Mézard, Parisi & Zecchina 2002)
Ns = sorted(PT["curves"], key=int)
cross = {n: PT["alpha_half"][n] for n in Ns}
_pk = [max(PT["curves"][n], key=lambda r: r["median_conflicts"])["alpha"] for n in Ns]
PEAK_LO, PEAK_HI = min(_pk), max(_pk)

# ---------------- figures
figs = []
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
for j, n in enumerate(Ns):
    rows = PT["curves"][n]
    a = [r["alpha"] for r in rows]
    ax[0].plot(a, [r["p_sat"] for r in rows], "o-", ms=3, color=CAT[j], label=f"N = {n}")
    ax[1].semilogy(a, [max(r["median_conflicts"], 1) for r in rows], "o-", ms=3, color=CAT[j], label=f"N = {n}")
for a_ in ax:
    a_.axvline(ALPHA_C, color=INK2, lw=0.8, ls=":")
    a_.set_xlabel("clause density α = M/N"); a_.legend(fontsize=7)
ax[0].set_ylabel("P(satisfiable)"); ax[0].set_title("Satisfiability transition")
ax[1].set_ylabel("median CDCL conflicts"); ax[1].set_title("Solver cost peaks near α_c")
figs.append(savefig(fig, HERE / "fig_phase.png"))
if SC or AN:
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
    if SC:
        N = np.array([r["N"] for r in SC["rows"]], float)
        y = np.array([r["median_conflicts"] for r in SC["rows"]])
        ax[0].semilogy(N, y, "o", color=CAT[0], label="median")
        ax[0].semilogy(N, [r["q90"] for r in SC["rows"]], "s", ms=3, color=CAT[1], label="90th percentile")
        ce, cp = SC["fits"]["exponential"]["coef"], SC["fits"]["power_law"]["coef"]
        NN = np.linspace(N.min(), N.max(), 100)
        ax[0].semilogy(NN, np.exp(ce[0] + ce[1] * NN), color=CAT[0], lw=1, label="exp fit")
        ax[0].semilogy(NN, np.exp(cp[0]) * NN ** cp[1], color=INK2, lw=1, ls="--", label="power-law fit")
        ax[0].set_xlabel("N"); ax[0].set_ylabel("CDCL conflicts at α = 4.26"); ax[0].set_title("Cost vs size"); ax[0].legend(fontsize=6.8)
    else:
        ax[0].axis("off")
    if AN:
        ok = [r for r in AN if r["median_analog_time"] is not None]
        ax[1].semilogy([r["N"] for r in ok], [r["median_analog_time"] for r in ok], "o-", color=CAT[2], label="median analog time")
        ax[1].set_xlabel("N"); ax[1].set_ylabel("analog time to solution"); ax[1].set_title("Continuous-time solver (SAT instances)")
        for r in AN:
            ax[1].annotate(f"{r['solved_frac']:.0%}", (r["N"], r["median_analog_time"] or 1), fontsize=6, textcoords="offset points", xytext=(0, 5), ha="center")
        ax[1].legend(fontsize=6.8)
    else:
        ax[1].axis("off")
    figs.append(savefig(fig, HERE / "fig_scaling.png"))

# ---------------- paper
P = Paper("P versus NP: What Random 3-SAT Experiments Can and Cannot Show",
          "Phase transition, solver scaling, a physics-inspired analog solver and learned satisfiability, with PINNeAPPle",
          report_id="Experiment 06 · Millennium problems · P vs NP")
abs_parts = [
    "P ≠ NP is a statement about worst-case complexity over all instances; no finite experiment on random instances can prove or refute it. "
    "We measure what can be measured and state its limits.",
    f"(1) With a complete CDCL solver (Glucose 4), the 50 % satisfiability crossing of random 3-SAT "
    f"moves from {cross[Ns[0]]:.2f} (N = {Ns[0]}) to {cross[Ns[-1]]:.2f} (N = {Ns[-1]}), consistent with the known threshold α_c ≈ {ALPHA_C}; "
    f"the median solver cost peaks at α = {PEAK_LO}–{PEAK_HI} on the 0.25-spaced grid, i.e. at the transition."]
if SC:
    better = "exponential" if SC["fits"]["exponential"]["aic"] < SC["fits"]["power_law"]["aic"] else "power-law"
    abs_parts.append(f"(2) At α = 4.26 the median conflict count grows from {SC['rows'][0]['median_conflicts']:.0f} (N = {SC['rows'][0]['N']}) to "
                     f"{SC['rows'][-1]['median_conflicts']:.0f} (N = {SC['rows'][-1]['N']}); by AIC the {better} fit is preferred over this range.")
if AN:
    _lo, _hi = min(r["solved_frac"] for r in AN), max(r["solved_frac"] for r in AN)
    _frac_txt = f"{_lo:.0%}" if _lo == _hi else f"{_lo:.0%}–{_hi:.0%}"
    abs_parts.append(f"(3) A continuous-time analog solver solved {_frac_txt} "
                     f"of satisfiable instances within its time budget for N = {AN[0]['N']}–{AN[-1]['N']}.")
if LS:
    abs_parts.append(f"(4) A validation-selected PINNeAPPle classifier on instance statistics reached {LS['all_features']['test_accuracy']:.1%} test "
                     f"accuracy against {LS['alpha_only']['test_accuracy']:.1%} for α alone.")
P.abstract(" ".join(abs_parts), ["P vs NP", "random 3-SAT", "phase transition", "CDCL", "analog computing", "PINNeAPPle", "Millennium Prize problems"])
P.box("What this study shows — and what it does not", [
    "Shows: the average-case behaviour of one solver family on one random ensemble, measured with fixed seeds and reported with its spread.",
    "Does not bear on P vs NP: random 3-SAT near threshold may be hard on average for reasons unrelated to worst-case complexity, and a polynomial "
    "algorithm could exist that no current solver implements. A fit over N ≤ " + (str(SC["rows"][-1]["N"]) if SC else "a few hundred") +
    " cannot distinguish exponential growth from a high-degree polynomial with confidence.",
    "Learning satisfiability from cheap statistics exploits finite-N fluctuations of the ensemble; it is not a decision procedure."])
P.section("Background")
P.p("Cook and Levin showed that SAT is NP-complete [1, 2]; a polynomial algorithm for 3-SAT would give P = NP. For random 3-SAT with M = αN "
    "clauses, satisfiability drops sharply near α_c ≈ 4.267 as N → ∞ [3, 4], and complete solvers are slowest there [5]. Continuous-time "
    "dynamical systems whose attractors are the solutions have been proposed as physics-inspired solvers [6]; their time-to-solution was "
    "reported to grow polynomially in analog time at the cost of exponentially growing auxiliary variables.")
P.section("Methods")
P.bullets([
    f"Instances: uniform random 3-SAT (3 distinct variables per clause, random signs), fixed seed. Phase transition: N ∈ {{{', '.join(Ns)}}}, "
    "α from 3.0 to 5.5 in steps of 0.25, 100 instances per point; 50 % crossing by linear interpolation.",
    "Scaling: α = 4.26, 60 instances per N; median and 90th percentile of CDCL conflicts; exponential (log y = a + bN) and power-law "
    "(log y = a + k log N) fits compared by AIC, both with two parameters.",
    "Analog solver: the Ercsey-Ravasz–Toroczkai dynamics (soft spins s ∈ [−1, 1]<super>N</super>, exponentially growing clause weights), "
    "explicit Euler with a step cap, budget t ≤ 500, 30 satisfiable instances per N at α = 4.25 (unsatisfiable ones are discarded, since the "
    "solver can only succeed on satisfiable instances).",
    "Learned satisfiability: N = 100, α ∼ U(3.8, 4.8), 3000 instances labelled by the complete solver; features α, occurrence std and max, "
    "literal-bias mean and std, fraction of unused variables; 70/15/15 split; the first attempt used one fixed PINNeAPPle ModifiedMLP, the second "
    "selects among PINNeAPPle classifiers on validation only; test set used once."])
P.section("Results")
P.figure(figs[0], "Left: probability of satisfiability; right: median CDCL conflicts. Dotted line: α_c ≈ 4.267.", 15.5)
P.table([["N", "α at P(sat) = 0.5", "distance to α_c"]] +
        [[n, f"{cross[n]:.3f}" if cross[n] else "—", f"{cross[n] - ALPHA_C:+.3f}" if cross[n] else "—"] for n in Ns],
        "Finite-size crossing points.", [2, 4, 4])
if SC:
    P.figure(figs[1], "Left: cost at α = 4.26 with both fits. Right: analog time to solution; labels give the fraction solved within budget.", 15.5)
    f = SC["fits"]
    P.table([["model", "coefficients", "RSS (log scale)", "AIC"],
             ["exponential", f"b = {f['exponential']['coef'][1]:.4f} per variable", f"{f['exponential']['rss']:.3f}", f"{f['exponential']['aic']:.2f}"],
             ["power law", f"k = {f['power_law']['coef'][1]:.2f}", f"{f['power_law']['rss']:.3f}", f"{f['power_law']['aic']:.2f}"]],
            "Fits of median conflicts vs N at α = 4.26.", [3, 5, 3.5, 2.5])
    P.p("Both models have two parameters, so the AIC difference is the log-likelihood difference. Over a factor of "
        f"{SC['rows'][-1]['N'] / SC['rows'][0]['N']:.0f} in N the two curves remain close; the preference is a statement about this range only.")
if AN:
    P.table([["N", "solved within budget", "median analog time"]] +
            [[str(r["N"]), f"{r['solved_frac']:.0%}", fmt(r["median_analog_time"]) if r["median_analog_time"] is not None else "—"] for r in AN],
            "Analog solver on satisfiable instances at α = 4.25.", [2, 4, 4])
if LS:
    rows = [["model", "test accuracy", "selected on validation"]]
    if LS0:
        rows += [[f"first attempt: {k.replace('_', ' ')}", f"{LS0[k]['test_accuracy']:.1%}", "fixed ModifiedMLP"] for k in ("alpha_only", "all_features")]
    rows += [[k.replace("_", " "), f"{LS[k]['test_accuracy']:.1%}", LS[k].get("selected_on_validation", "fixed ModifiedMLP")] for k in ("alpha_only", "all_features")]
    conf = LS["all_features"].get("confusion")
    if conf:
        tot = np.array(conf).sum(1)
        rows.append(["majority class (test)", f"{tot.max() / tot.sum():.1%}", "—"])
    P.table(rows, "Predicting satisfiability at N = 100.", [5, 3, 5])
    gain = LS["all_features"]["test_accuracy"] - LS["alpha_only"]["test_accuracy"]
    P.p(f"Instance statistics beyond α change test accuracy by {gain * 100:+.1f} percentage points. Near the threshold, α explains most of what "
        "cheap statistics can see; the remainder reflects finite-size fluctuations of the ensemble, not structure a polynomial algorithm could exploit.")
P.section("Discussion")
P.p("The measurements reproduce the known average-case picture: a sharp transition near α_c and a cost peak there. None of this is evidence "
    "for or against P ≠ NP. What PINNeAPPle adds is the reproducible harness (fixed seeds, validation-only model selection, a single test pass) "
    "for probes of this kind; the honest conclusion is that the question is not experimental.")
P.references([
    "Cook, S. A. (1971). The complexity of theorem-proving procedures. <i>Proceedings of the 3rd ACM STOC</i>, 151–158.",
    "Levin, L. A. (1973). Universal sequential search problems. <i>Problems of Information Transmission</i> 9, 265–266.",
    "Kirkpatrick, S., & Selman, B. (1994). Critical behavior in the satisfiability of random Boolean expressions. <i>Science</i> 264, 1297–1301.",
    "Mézard, M., Parisi, G., & Zecchina, R. (2002). Analytic and algorithmic solution of random satisfiability problems. <i>Science</i> 297, 812–815.",
    "Mitchell, D., Selman, B., & Levesque, H. (1992). Hard and easy distributions of SAT problems. <i>Proceedings of AAAI-92</i>, 459–465.",
    "Ercsey-Ravasz, M., & Toroczkai, Z. (2011). Optimization hardness as transient chaos in an analog approach to constraint satisfaction. <i>Nature Physics</i> 7, 966–970.",
])
print(P.build(str(HERE / "Millennium_PvsNP_PINNeAPPle.pdf")))
