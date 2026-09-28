"""PDR thesis reproduction + improvement papers (two PDFs from one results.json)."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parent / "paperkit"))
from paperkit import CAT, INK2, Paper, fmt, savefig, setup_mpl  # noqa: E402

setup_mpl()
import os
R = json.loads((ROOT / os.environ.get("PDR_RESULTS", "results") / "results.json").read_text())   # PDR_RESULTS=results_smoke for a dry run
D = R["data"]
KINDS = [k for k in ("tlio", "dive") if f"ch3_{k}" in R]
C3 = {k: R[f"ch3_{k}"] for k in KINDS}
IMU = R.get("imu_only_dead_reckoning_90s")
C4 = R.get("ch4_grid_pinn", {})
IMP = R.get("improvements")
VP = R.get("veriphysics")
ALPHA = 0.05

# numbers reported in the thesis (Leica dataset, full-length experiments): Tables 3-1 and 4-2 of Ligthart (2025)
THESIS = {"tlio": {"median": 0.17, "p95": 0.84, "drift_med": 8.1, "drift_p95": 28.8},
          "dive": {"median": 0.26, "p95": 1.26, "drift_med": 15.1, "drift_p95": 47.1}}
THESIS_PINN = {"0.0": (0.14, 8.9), "0.01": (0.13, 8.3), "0.1": (0.14, 8.2), "1.0": (0.16, 10.9)}   # Extended ResNet


def drifts(block):
    return np.array([s["drift_rate_pct"] for s in block["dead_reckoning_90s"]["segments"]])


def paired(a, b):
    """Wilcoxon signed-rank on the same 90 s test segments; returns (median difference b - a, p)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or len(a) < 6:
        return float("nan"), float("nan")
    return float(np.median(b - a)), float(wilcoxon(a, b).pvalue)


def ptxt(t):
    """'+0.12, p = 0.03' or a note when the paired test could not run."""
    return "not tested (n < 6)" if not np.isfinite(t[1]) else f"{t[0]:+.2f}, p = {t[1]:.2g}"


def bias_z(block):
    m, se = np.array(block["standalone_gt_attitude"]["bias_heading_frame_mean"]), np.array(block["standalone_gt_attitude"]["bias_heading_frame_se"])
    return m, se, m / np.maximum(se, 1e-12)


def vp_rows(V):
    rows = [["model", "physics confidence", "coverage", "components"]]
    for n, v in V.items():
        if n == "not_run":
            continue
        rows.append([n, fmt(v["overall_score"]), f"{v['coverage']:.1f}", "; ".join(f"{c['name']} {c['score']:.2f}" for c in v["components"])])
    return rows


DATA_BULLET = (f"Data: RIDI [4], 94 smartphone sequences with Google Tango visual-inertial ground truth, 200 Hz. Person-disjoint split: train "
               f"{', '.join(D['train']['people'])} ({D['train']['n_seq']} sequences, {D['train']['hours']:.2f} h); validation "
               f"{', '.join(D['val']['people'])} ({D['val']['n_seq']}, {D['val']['hours']:.2f} h); test {', '.join(D['test']['people'])} "
               f"({D['test']['n_seq']}, {D['test']['hours']:.2f} h).")

# =====================================================================================  paper 1: reproduction
figs = {}
if KINDS:
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
    ax[0].boxplot([drifts(C3[k]) for k in KINDS], tick_labels=[k.upper() for k in KINDS], whis=(5, 95), showfliers=False)
    ax[0].set_ylabel("drift rate over 90 s [%]"); ax[0].set_title("EKF + network, test people")
    w = 0.35
    for j, k in enumerate(KINDS):
        m, se, _ = bias_z(C3[k])
        ax[1].bar(np.arange(3) + (j - 0.5) * w, m, w, yerr=1.96 * se, color=CAT[j], label=k.upper(), capsize=2)
    ax[1].axhline(0, color=INK2, lw=0.7)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["along heading", "cross heading", "vertical"])
    ax[1].set_ylabel("mean velocity error [m/s]"); ax[1].set_title("Prediction bias (95 % CI)"); ax[1].legend(fontsize=7)
    figs["ch3"] = savefig(fig, HERE / "fig_ch3.png")
lams = sorted(C4, key=float)
if lams:
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.boxplot([drifts(C4[l]) for l in lams], tick_labels=[f"λ = {l}" for l in lams], whis=(5, 95), showfliers=False)
    ax.set_ylabel("drift rate over 90 s [%]"); ax.set_title("Grid-output network, physics-loss weight")
    figs["ch4"] = savefig(fig, HERE / "fig_ch4.png")

claims = {}
if len(KINDS) == 2:
    d, p = paired(drifts(C3["tlio"]), drifts(C3["dive"]))
    claims["tlio_better"] = (d, p, d > 0 and p < ALPHA)
if KINDS:
    best = min(KINDS, key=lambda k: C3[k]["dead_reckoning_90s"]["drift_median_pct"])
    m, se, z = bias_z(C3[best])
    claims["bias"] = (m, z, bool(np.abs(z[:2]).max() > 3))
if len(lams) > 1:
    base = drifts(C4[lams[0]])
    tests = {l: paired(base, drifts(C4[l])) for l in lams[1:]}
    claims["pinn"] = tests
    pinn_helps = any(t[0] < 0 and t[1] < ALPHA / len(tests) for t in tests.values())   # Bonferroni over λ

P = Paper("Reproducing a Physics-Informed Pedestrian Dead-Reckoning Study on Public Data with PINNeAPPle",
          "EKF with velocity-predicting networks (adapted TLIO, DIVE) and grid-output PINN variants, after Ligthart (2025)",
          report_id="Experiment 07 · Paper 1 of 2 · Reproduction")
abst = ["Ligthart's MSc thesis [1] studied a pedestrian dead-reckoning (PDR) algorithm in which an extended Kalman filter (EKF) is corrected "
        "by a neural network that predicts walking velocity [2], and asked whether physics-informed training improves it. It concluded that "
        "an adapted TLIO network beats DIVE, that a bias in the predicted velocity is the bottleneck, and that a physics loss does not help. "
        "The thesis used a proprietary 25-hour dataset; we repeat the study with PINNeAPPle on the public RIDI dataset (2.7 h, person-disjoint split)."]
if "tlio_better" in claims:
    d, p, ok = claims["tlio_better"]
    abst.append(f"Adapted TLIO {'again outperforms' if ok else 'does not significantly outperform'} DIVE: median 90-second drift "
                f"{C3['tlio']['dead_reckoning_90s']['drift_median_pct']:.1f} % vs {C3['dive']['dead_reckoning_90s']['drift_median_pct']:.1f} % "
                f"(paired Wilcoxon {'p = ' + format(p, '.2g') if np.isfinite(p) else 'test not run, n < 6'}).")
if "bias" in claims:
    m, z, ok = claims["bias"]
    abst.append(f"The best network's error has a {'significant' if ok else 'non-significant'} mean in the heading frame "
                f"({m[0]:+.3f} m/s along and {m[1]:+.3f} m/s across the walking direction, |z| up to {np.abs(z[:2]).max():.0f}), "
                f"{'confirming' if ok else 'not confirming'} the bias the thesis identified.")
if "pinn" in claims:
    abst.append(f"A physics loss from the accelerometer model {'improved' if pinn_helps else 'did not significantly improve'} drift for "
                f"λ ∈ {{{', '.join(lams[1:])}}} relative to λ = 0 (Bonferroni-corrected), "
                f"{'contradicting' if pinn_helps else 'consistent with'} the thesis.")
P.abstract(" ".join(abst), ["pedestrian dead reckoning", "inertial navigation", "extended Kalman filter", "PINN", "TLIO", "reproducibility", "PINNeAPPle"])
P.box("What differs from the thesis", [
    "Data: public RIDI smartphone data (2.7 h, 11 people) instead of Leica's proprietary 25 h; absolute numbers are not comparable, only the "
    "direction of each conclusion.",
    "Filter: state (position, velocity) with the phone's own orientation; the thesis' EKF also estimated attitude and IMU biases.",
    "Drift is measured over fixed 90 s segments of the test sequences, not over whole experiments.",
    "Networks: PINNeAPPle Conv1DModel (dilated residual 1-D CNN) backbones instead of the thesis' ResNet/U-Net/Transformer variants."])
P.section("Background")
P.p("In PDR the IMU is integrated forward and the drift that accumulates from sensor errors is corrected by a pseudo-measurement. In the "
    "algorithm of Liu et al. [2] (TLIO) the pseudo-measurement is a velocity (or displacement) with its uncertainty, predicted by a network "
    "from a window of IMU data. The EKF assumes that this measurement error has zero mean; the thesis showed that the networks violate this "
    "and that the violation limits accuracy. DIVE [3] instead rotates the IMU data into a gravity-aligned, yaw-free frame.")
P.section("Methods")
P.bullets([
    DATA_BULLET,
    "Networks: 1 s window (thesis baseline), velocity + log-standard-deviation head; MSE for 10 epochs then Gaussian maximum likelihood, "
    "early stopping on validation. Training augmentation as in the thesis: random accelerometer (±0.02 m/s²) and gyroscope (±0.002 rad/s) "
    "biases, gravity-direction perturbation and random yaw, redrawn every epoch.",
    "EKF: PINNeAPPle ExtendedKalmanFilter with state (position, velocity), IMU mechanisation at 50 Hz using the phone's game rotation vector "
    "(heading aligned to the ground truth once, at the segment start), network velocity update at 10 Hz; drift rate = final position error / "
    "distance walked, per 90 s segment.",
    "PINN (thesis chapter 4): network predicting the velocity over a 4 s window at 50 Hz; physics loss = mismatch between the accelerometer "
    "measurement model and the central-difference derivative of the predicted velocity; λ ∈ {0, 0.01, 0.1, 1}.",
    f"Statistics: all comparisons are paired on the same test segments (Wilcoxon signed-rank, α = {ALPHA}); the bias test uses the "
    "standard error of the mean heading-frame error; test data are used once."])
P.section("Results")
if KINDS:
    P.figure(figs["ch3"], "Left: drift over 90 s segments (whiskers 5–95 %). Right: mean velocity error in the true heading frame.", 15.5)
    rows = [["model", "standalone error median / p95 [m/s]", "drift median / p95 [%]", "thesis: error median / p95", "thesis: drift median / p95"]]
    for k in KINDS:
        s, dd, t = C3[k]["standalone_gt_attitude"], C3[k]["dead_reckoning_90s"], THESIS[k]
        rows.append([k.upper(), f"{s['median']:.2f} / {s['p95']:.2f}", f"{dd['drift_median_pct']:.1f} / {dd['drift_p95_pct']:.1f}",
                     f"{t['median']:.2f} / {t['p95']:.2f}", f"{t['drift_med']:.1f} / {t['drift_p95']:.1f}"])
    if IMU:
        rows.append(["IMU only (no network)", "—", f"{IMU['drift_median_pct']:.0f} / {IMU['drift_p95_pct']:.0f}", "—", "—"])
    P.table(rows, "Chapter 3 reproduction. Thesis values from its Table 3-1 (different data and drift definition).", [3, 3.6, 3, 3.3, 3.3])
    if "tlio_better" in claims:
        d, p, ok = claims["tlio_better"]
        P.p(f"On the same {len(drifts(C3['tlio']))} test segments, DIVE's drift minus TLIO's: {ptxt((d, p))}. "
            f"{'The thesis ranking holds.' if ok else 'The thesis ranking is not confirmed at this sample size.'}")
    if "bias" in claims:
        m, z, ok = claims["bias"]
        P.p(f"Bias of the {best.upper()} network in the true heading frame: {m[0]:+.3f} m/s along, {m[1]:+.3f} m/s across and {m[2]:+.4f} m/s "
            f"vertical (z = {z[0]:.0f}, {z[1]:.0f}, {z[2]:.1f}). "
            + ("A consistent along-heading under-prediction is exactly the kind of non-zero-mean error that the EKF cannot absorb." if ok and m[0] < 0 else
               "The horizontal bias is statistically clear." if ok else "No horizontal bias is detectable."))
if lams:
    P.figure(figs["ch4"], "Chapter 4 reproduction: drift for each physics-loss weight.", 9.5)
    rows = [["λ", "standalone error median [m/s]", "drift median / p95 [%]", "Δ drift vs λ = 0 (median, p)", "thesis Ext. ResNet: error / drift"]]
    for l in lams:
        s, dd = C4[l]["standalone_gt_attitude"], C4[l]["dead_reckoning_90s"]
        tst = claims.get("pinn", {}).get(l)
        th = THESIS_PINN.get(str(float(l)))
        rows.append([l, f"{s['median']:.3f}", f"{dd['drift_median_pct']:.1f} / {dd['drift_p95_pct']:.1f}",
                     ptxt(tst) if tst else "reference", f"{th[0]:.2f} / {th[1]:.1f}" if th else "—"])
    P.table(rows, "Grid-output network with physics loss.", [1.5, 3.5, 3, 4, 4])
P.section("Discussion")
P.p("The reproduction transfers the thesis' questions to public data and checks each conclusion with a paired test rather than by comparing "
    "medians. Absolute drift rates differ from the thesis because the data, the filter and the drift definition differ; the direction of each "
    "conclusion is what can be compared. The companion paper uses the bias finding directly: if the error has a non-zero mean in the heading "
    "frame, it can be estimated on validation data and removed.")
REFS = [
    "Ligthart, L. (2025). <i>Pedestrian dead reckoning using data-driven and physics-informed machine learning</i>. MSc thesis, Delft University of Technology.",
    "Liu, W., Caruso, D., Ilg, E., Dong, J., Mourikis, A. I., Daniilidis, K., Kumar, V., & Engel, J. (2020). TLIO: Tight learned inertial odometry. <i>IEEE Robotics and Automation Letters</i> 5, 5653–5660.",
    "Cao, X., Zhou, C., Zeng, D., & Wang, Y. (2022). RIO: Rotation-equivariance supervised learning of robust inertial odometry. <i>CVPR</i>, 6614–6623.",
    "Yan, H., Shan, Q., & Furukawa, Y. (2018). RIDI: Robust IMU double integration. <i>ECCV</i>, 621–636.",
    "Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. <i>Journal of Computational Physics</i> 378, 686–707.",
]
P.references(REFS)
print(P.build(str(HERE / "PDR_Reproduction_PINNeAPPle.pdf")))

# =====================================================================================  paper 2: improvement
if IMP:
    V = IMP["variants"]
    names = list(V)
    base = drifts(V[names[0]])
    tests = {n: paired(base, drifts(V[n])) for n in names[1:]}
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.1))
    ax[0].boxplot([drifts(V[n]) for n in names], tick_labels=[f"V{i}" for i in range(len(names))], whis=(5, 95), showfliers=False)
    ax[0].set_ylabel("drift rate over 90 s [%]"); ax[0].set_title("Improvements, test people")
    for i, n in enumerate(names):
        m, se, _ = bias_z(V[n])
        ax[1].errorbar(i, m[0], yerr=1.96 * se[0], fmt="o", color=CAT[0], label="along heading" if i == 0 else None)
        ax[1].errorbar(i + 0.15, m[1], yerr=1.96 * se[1], fmt="s", color=CAT[1], label="cross heading" if i == 0 else None)
    ax[1].axhline(0, color=INK2, lw=0.7); ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels([f"V{i}" for i in range(len(names))])
    ax[1].set_ylabel("mean velocity error [m/s]"); ax[1].set_title("Heading-frame bias (95 % CI)"); ax[1].legend(fontsize=7)
    f_imp = savefig(fig, HERE / "fig_improve.png")
    last = names[-1]
    d_last, p_last = tests[last]
    sig = {n: (t[0] < 0 and t[1] < ALPHA / len(tests)) for n, t in tests.items()}
    Q = Paper("Removing the Velocity Bias in Learned Pedestrian Dead Reckoning with PINNeAPPle",
              "Yaw test-time augmentation, validation-fitted bias correction and variance recalibration on public RIDI data",
              report_id="Experiment 07 · Paper 2 of 2 · Improvement")
    Q.abstract(
        "The thesis we reproduced [1] found that the main limit of a network-corrected pedestrian dead-reckoning EKF is a bias in the predicted "
        "velocity, and left its mitigation to future work. We target it with three post-hoc changes that need no retraining, each fitted on "
        f"validation people only: yaw-equivariant test-time augmentation, a bias correction in the walking-direction frame, and a rescaling of "
        f"the predicted variance so that the EKF weights the network correctly. Starting from the {IMP['base_kind'].upper()} network (chosen on "
        f"validation), the full combination changes the median 90-second drift on unseen test people from "
        f"{V[names[0]]['dead_reckoning_90s']['drift_median_pct']:.1f} % to {V[last]['dead_reckoning_90s']['drift_median_pct']:.1f} % "
        f"(paired median difference and Wilcoxon test: {ptxt((d_last, p_last))}), "
        + ("a significant improvement." if sig[last] else "which is not significant after correction for multiple comparisons.")
        + " VeriPhysics scores are reported for every variant.",
        ["pedestrian dead reckoning", "bias correction", "test-time augmentation", "uncertainty calibration", "EKF", "PINNeAPPle"])
    Q.box("Protocol", [
        "Every correction is fitted on the validation people; the test people are used once, for the numbers below.",
        f"Comparisons are paired on the same {len(base)} test segments with Bonferroni correction over {len(tests)} variants.",
        "No retraining: the improvements act on a trained network's outputs, so they can be applied to the thesis' own networks."])
    Q.section("Methods")
    Q.bullets([
        DATA_BULLET,
        "Yaw test-time augmentation: the window is rotated by 8 yaw angles, each prediction rotated back and averaged. A network trained with random "
        "yaw should be equivariant; averaging removes the part of the error that is not.",
        "Bias correction: in the frame of the predicted walking direction (along, across, vertical), a per-axis affine map (scale and offset) "
        "from predicted to true velocity, fitted by least squares on validation windows with speed above 0.3 m/s.",
        "Variance recalibration: one scale factor for the predicted standard deviation, set so that the normalised validation errors have unit "
        "RMS, because the EKF gain depends on the predicted variance.",
        "VeriPhysics: benchmark agreement against the RIDI ground truth and calibration (ECE, 90 % coverage) through "
        "pinneapple_analysis.verification.compute_physics_confidence."])
    Q.section("Results")
    Q.figure(f_imp, "Left: drift over 90 s test segments. Right: heading-frame bias per variant. V0 … V" + str(len(names) - 1) + " as in the table.", 15.5)
    rows = [["", "variant", "error median [m/s]", "bias along / across [m/s]", "z RMS (x, y, z)", "drift median / p95 [%]", "Δ vs V0 (p)"]]
    for i, n in enumerate(names):
        s, dd = V[n]["standalone_gt_attitude"], V[n]["dead_reckoning_90s"]
        t = tests.get(n)
        rows.append([f"V{i}", n, f"{s['median']:.3f}", f"{s['bias_heading_frame_mean'][0]:+.3f} / {s['bias_heading_frame_mean'][1]:+.3f}",
                     ", ".join(f"{v:.2f}" for v in s.get("z_rms", [])) or "—", f"{dd['drift_median_pct']:.1f} / {dd['drift_p95_pct']:.1f}",
                     f"{ptxt(t)}{' *' if sig[n] else ''}" if t else "—"])
    Q.table(rows, "Improvement variants on test people. z RMS = 1 means calibrated variance. * significant after Bonferroni correction.", [0.8, 5, 2, 3, 2.6, 2.6, 2.4])
    if VP:
        Q.table(vp_rows(VP), "VeriPhysics physics-confidence scores (test people). Not run: " +
                "; ".join(f"{k.replace('_', ' ')} ({v})" for k, v in VP.get("not_run", {}).items()) + ".", [4.5, 2.2, 1.6, 7.5])
    Q.section("Discussion")
    Q.p("The bias the thesis identified is measurable in the heading frame and can be estimated from held-out people. Whether removing it helps "
        "the filter is the question the paired test answers: "
        + ("it does, significantly." if sig[last] else "here the change is not significant, so the bias is not the whole story on this data.")
        + " Variance recalibration matters for a different reason: the EKF uses the predicted variance as the measurement noise, so an over- or "
        "under-confident network shifts the balance between inertial propagation and network correction, independently of the mean error.")
    Q.references(REFS[:4])
    print(Q.build(str(HERE / "PDR_Improvement_PINNeAPPle.pdf")))
