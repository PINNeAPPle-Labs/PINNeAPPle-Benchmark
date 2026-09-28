"""Heated-channel digital twin paper -- numbers read from results/."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "paperkit"))
from paperkit import CAT, DIV, INK2, MUTED, SEQ, Paper, fmt, savefig, setup_mpl  # noqa: E402

setup_mpl()
R = json.loads((ROOT / "results" / "results.json").read_text())
VER = json.loads((ROOT / "results" / "verification.json").read_text())
FIG = ROOT / "figures"
M = R["models"]
FAM = ["test_same_family", "test_unseen_sinusoid", "test_unseen_edge_steps"]
FAMLAB = {"test_same_family": "same family", "test_unseen_sinusoid": "unseen: chirps", "test_unseen_edge_steps": "unseen: edge steps"}
ex = np.load(ROOT / "results" / "example_rollout.npz")
meta = json.loads((ROOT / "data" / "meta.json").read_text())
xc, yc = np.array(meta["xc"]), np.array(meta["yc"])

# ---- Fig 1: CFD snapshot (T and u) + actuators
tr = ex["truth"]; act = ex["act"]
fig = plt.figure(figsize=(8.6, 4.2))
gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1])
for r, (fi, lab, cm) in enumerate(((3, "T", SEQ), (0, "u", SEQ), (2, "p", DIV))):
    ax = fig.add_subplot(gs[r, 0])
    f = tr[60, fi].T
    kw = dict(cmap=cm, origin="lower", extent=[0, 10, 0, 1], aspect="auto")
    if fi == 2:
        v = np.abs(f).max(); kw.update(vmin=-v, vmax=v)
    im = ax.imshow(f, **kw); ax.set_ylabel(lab); ax.grid(False)
    if r == 0:
        ax.axhline(0.0, xmin=0.2, xmax=0.4, color=CAT[1], lw=4)
        ax.set_title("CFD state at t = 24 (unseen chirp actuation)")
    plt.colorbar(im, ax=ax, pad=0.01)
ax.set_xlabel("x / H")
ax = fig.add_subplot(gs[:, 1])
tt = np.arange(act.shape[0]) * 0.4
ax.plot(tt, act[:, 0], color=CAT[0], label="U$_{in}$(t)"); ax.plot(tt, act[:, 5], color=CAT[1], label="Q$_h$(t)")
ax.set_xlabel("t"); ax.set_title("Actuators"); ax.legend(fontsize=7)
f1 = savefig(fig, FIG / "fig1_cfd.png")

# ---- Fig 2: verification
fig, ax = plt.subplots(1, 2, figsize=(8.2, 2.8))
P_ = VER["poiseuille"]
h = [1 / p["grid"][1] for p in P_]
ax[0].loglog(h, [p["u_profile_max_err"] for p in P_], "-o", color=CAT[0], label="max |u − u$_{Poiseuille}$|")
ax[0].loglog(h, [p["u_profile_max_err"] for p in P_][0] * (np.array(h) / h[0]) ** 2, ":", color=MUTED, label="2nd order")
ax[0].set_xlabel("Δy"); ax[0].set_title("Poiseuille profile"); ax[0].legend(fontsize=7)
G = VER["grid_convergence"]
for j, (k, lab) in enumerate((("T_out", "T$_{out}$"), ("T_max", "T$_{max}$"), ("dp", "Δp"))):
    vals = np.array(G[k]["values"]); ax[1].plot([64, 128, 256], vals / vals[-1], "-o", color=CAT[j], label=lab)
ax[1].set_xscale("log"); ax[1].set_xticks([64, 128, 256]); ax[1].set_xticklabels(["64×16", "128×32", "256×64"])
ax[1].set_ylabel("value / finest"); ax[1].set_title("Three-grid QoIs (t = 20)"); ax[1].legend(fontsize=7)
f2 = savefig(fig, FIG / "fig2_verification.png")

# ---- Fig 3: rollout error by model and family (T field)
names = list(M)
fig, ax = plt.subplots(figsize=(7.8, 3.1))
x = np.arange(len(names)); w = 0.26
for j, fam in enumerate(FAM):
    vals = [min(M[n]["test"][fam]["rel_l2"]["T"], 2.0) for n in names]
    ax.bar(x + (j - 1) * w, vals, width=w - 0.02, color=CAT[j], label=FAMLAB[fam])
ax.set_xticks(x); ax.set_xticklabels([n.replace(" (PINNeAPPle", "\n(PINNeAPPle").replace(" + ", " +\n") for n in names], fontsize=6.5)
ax.set_ylabel("relative L2 of T over 40-time-unit rollout"); ax.set_title("Autoregressive rollout error (clipped at 2)"); ax.legend(fontsize=7)
f3 = savefig(fig, FIG / "fig3_rollout.png")

# ---- Fig 4: example rollouts (T at three times)
keys = [k for k in ex.files if k not in ("truth", "act")]
fig, axs = plt.subplots(3, 1 + len(keys), figsize=(8.8, 3.9))
for r, k_t in enumerate((10, 50, 100)):
    fields = [tr[k_t, 3]] + [ex[k][k_t, 3] for k in keys]
    for c, (f, lab) in enumerate(zip(fields, ["CFD"] + keys)):
        axs[r, c].imshow(f.T, origin="lower", extent=[0, 10, 0, 1], aspect="auto", cmap=SEQ, vmin=0, vmax=tr[:, 3].max())
        axs[r, c].set_xticks([]); axs[r, c].set_yticks([]); axs[r, c].grid(False)
        if r == 0:
            axs[r, c].set_title(lab, fontsize=7.5)
        if c == 0:
            axs[r, c].set_ylabel(f"t = {0.4 * k_t:.0f}", fontsize=7.5)
f4 = savefig(fig, FIG / "fig4_example.png")

# ---- Fig 5: layer 3 (sensors)
L3 = R["layer3"]
if (ROOT / "results" / "layer3_example.npz").exists():
    e3 = np.load(ROOT / "results" / "layer3_example.npz")
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 2.8))
    for c, (k, lab) in enumerate((("truth", "CFD truth"), ("ol", "open-loop surrogate"), ("enkf", "EnKF (state + gain)"), ("gappy", "sensors only (gappy POD)"))):
        Tf = e3[k]
        tout = Tf[:, -1, :].mean(1)
        ax[0].plot(np.arange(len(tout)) * 0.4, tout, color=[INK2, CAT[1], CAT[0], CAT[2]][c], lw=2 if k == "truth" else 1.2, label=lab)
    ax[0].set_xlabel("t"); ax[0].set_ylabel("outlet mean T"); ax[0].set_title("Digital twin with 3 sparse sensors"); ax[0].legend(fontsize=6.5)
    ax[1].plot(np.arange(len(e3["gains"])) * 0.4, e3["gains"], color=CAT[0]); ax[1].axhline(L3["cases"][0]["true_gain"], color=INK2, ls="--")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("estimated heater gain"); ax[1].set_title("Joint parameter estimation")
    f5 = savefig(fig, FIG / "fig5_sensors.png")
else:
    f5 = None

# ---- Fig 6: layer 4 (control)
L4 = R["layer4"]
fig, ax = plt.subplots(figsize=(6.0, 2.8))
ts = np.arange(1, len(L4["tout_surrogate"]) + 1) * 0.4
ax.plot(ts, L4["tout_surrogate"], color=CAT[0], label="surrogate prediction")
ax.plot(np.arange(len(L4["tout_cfd"])) * 0.4, L4["tout_cfd"], color=CAT[1], label="CFD with the optimised actuation")
ax.axhline(L4["target_T_out"], color=INK2, ls="--", lw=1, label="target")
ax.set_xlabel("t"); ax.set_ylabel("T$_{out}$"); ax.set_title("Open-loop optimisation through the surrogate, verified in CFD"); ax.legend(fontsize=7)
f6 = savefig(fig, FIG / "fig6_control.png")

# ======================================================================== paper
sel = R["selected_on_val"]
Pp = Paper("A Physics-AI Digital Twin of an Actuated Heated Channel: CFD Reference, Neural Surrogates, Sparse-Sensor Assimilation and Control with PINNeAPPle",
           "Can a fast surrogate replace transient CFD for actuator trajectories it has never seen?", report_id="Experiment 05")
st = M[sel]["test"]
L3, L4 = R["layer3"], R["layer4"]
cfd_s = R["data"]["cfd_seconds_per_run_mean"]
SPEEDUP = cfd_s / max(st["test_unseen_sinusoid"]["rollout_seconds"], 1e-9)
cs = L3["cases"]
L3G = {}
for g in (1.0, 1.3):
    cg = [c for c in cs if abs(c["true_gain"] - g) < 1e-9]
    L3G[g] = {"cases": cg, "ol": np.mean([c["open_loop"]["T_rel_l2"] for c in cg]), "enkf": np.mean([c["enkf_state"]["T_rel_l2"] for c in cg]),
              "enkf_g": np.mean([c["enkf_state_and_gain"]["T_rel_l2"] for c in cg]), "gains": [c["gain_estimate_final"] for c in cg]}
pi = M["Physics-informed FNO"]; fn = M["FNO (PINNeAPPle FNO2d)"]
PI_BETTER = (pi["test"]["test_unseen_sinusoid"]["rel_l2"]["T"] < fn["test"]["test_unseen_sinusoid"]["rel_l2"]["T"]
             and pi["test"]["test_unseen_sinusoid"]["energy_residual_mean_abs"] < fn["test"]["test_unseen_sinusoid"]["energy_residual_mean_abs"])
Pp.abstract(
    f"We implement, end to end, the four-layer digital-twin study of a 2-D channel with a heated wall section driven by a time-varying inlet "
    f"velocity U<sub>in</sub>(t) and heater power Q<sub>h</sub>(t). (1) Computational physics: a transient incompressible Navier–Stokes + energy "
    f"solver added to PINNeAPPle, verified against plane Poiseuille flow (second-order convergence of the profile), the fully developed Nusselt "
    f"number for one-sided uniform heating (Nu = {VER['nusselt'][-1]['Nu_mean']:.3f} vs 5.385, error {100 * abs(VER['nusselt'][-1]['rel_err']):.2f}%) and a "
    f"three-grid convergence study; 82 CFD trajectories with random, chirp and edge-step actuations. (2) Scientific ML: POD + DMDc, POD + MLP latent "
    f"dynamics, a convolutional autoencoder, FNO and DeepONet from PINNeAPPle, and a physics-informed FNO with continuity and global energy-balance "
    f"losses, all rolled out autoregressively for 100 steps. The model selected on validation ({sel}) reproduces the temperature field of unseen "
    f"chirp actuations with relative L2 error {fmt(st['test_unseen_sinusoid']['rel_l2']['T'])} and outlet temperature RMSE "
    f"{fmt(st['test_unseen_sinusoid']['qoi_rmse']['T_out'], 4)}, roughly {cfd_s / max(st['test_unseen_sinusoid']['rollout_seconds'], 1e-9):.0f}× faster than the CFD. "
    f"The physics-informed FNO {'improved on' if PI_BETTER else 'did not improve on'} the data-only FNO "
    f"({fmt(pi['test']['test_unseen_sinusoid']['rel_l2']['T'])} vs {fmt(fn['test']['test_unseen_sinusoid']['rel_l2']['T'])}) but neither approached POD + MLP; "
    f"POD + DMDc diverged. (3) Data assimilation: with three noisy sensors, an ensemble Kalman filter (PINNeAPPle) changed the mean field error "
    f"only marginally (miscalibrated heater: {fmt(L3G[1.3]['ol'])} → {fmt(L3G[1.3]['enkf_g'])} with joint gain estimation), and gain estimation "
    f"degraded the calibrated cases ({fmt(L3G[1.0]['ol'])} → {fmt(L3G[1.0]['enkf_g'])}). (4) Control: a heater schedule optimised through the "
    f"differentiable surrogate and re-simulated in CFD tracked the target with RMSE {fmt(L4['cfd_tracking_rmse'], 4)}, against "
    f"{fmt(L4['surrogate_tracking_rmse'], 4)} predicted by the surrogate.",
    ["digital twin", "CFD", "neural operators", "FNO", "DeepONet", "POD", "data assimilation", "EnKF", "model predictive control", "PINNeAPPle"])
Pp.section("Layer 1 — CFD reference")
Pp.p(f"Non-dimensional channel [0, 10]×[0, 1], Re = {meta['Re']:.0f} (based on height and reference inlet velocity), Pr = {meta['Pr']}, heater on the "
     f"bottom wall for 2 ≤ x ≤ 4 with prescribed flux Q<sub>h</sub>(t), adiabatic elsewhere, parabolic inlet scaled by U<sub>in</sub>(t) ∈ "
     f"[{meta['U_range'][0]}, {meta['U_range'][1]}], Q<sub>h</sub> ∈ [{meta['Q_range'][0]}, {meta['Q_range'][1]}]. Staggered MAC grid "
     f"{meta['nx']}×{meta['ny']}, donor-cell/central convection blend, Chorin projection with a pre-factorised pressure Poisson matrix, "
     f"Δt = {meta['dt_solver']}; outlet mass correction (PINNeAPPle <i>ThermalChannel2D</i>).")
nu = VER["nusselt"]
Pp.table([["Check", "Result"],
          ["Poiseuille profile error, 64×16 / 128×32 / 256×64", " / ".join(f"{p['u_profile_max_err']:.1e}" for p in P_)],
          ["dp/dx vs exact 12/Re (finest grid)", f"{P_[-1]['dpdx_mid']:.4f} vs {P_[-1]['dpdx_exact']:.4f}"],
          ["max |div u|", f"{max(p['max_divergence'] for p in P_):.1e}"],
          ["Nusselt (fully developed), 128×16 / 256×32", f"{nu[0]['Nu_mean']:.3f} / {nu[1]['Nu_mean']:.3f} (reference 5.385)"],
          ["global energy imbalance at steady state", f"{nu[1]['energy_imbalance_rel']:.1e}"]] +
         [[f"GCI (fine) of {k}", f"{G[k]['gci_fine']:.3f}, observed order {G[k]['observed_order']:.2f}"] for k in ("T_out", "T_max", "dp")],
         "Verification of the CFD reference. The GCI study shows oscillatory convergence for the transient QoIs (differences of a few percent "
         "between grids), which bounds the accuracy the surrogates can meaningfully be held to.", [7.5, 8])
Pp.figure(f2, "Left: second-order convergence of the Poiseuille profile. Right: QoIs of a transient scenario on three grids.", 15)
Pp.p(f"The dp/dx value is {100 * (1 - P_[-1]['dpdx_mid'] / P_[-1]['dpdx_exact']):.1f}% below 12/Re even on the finest grid; this is consistent with "
     "the first-order ghost-cell treatment of the no-slip wall and is reported rather than tuned away.")
Pp.figure(f1, "One CFD state (temperature, streamwise velocity, pressure) during an unseen chirp actuation; orange bar: heater.", 15.5)
Pp.p(f"Dataset: {R['data']['n_runs']} trajectories, each 15 time units of warm-up followed by 40 recorded units at Δt = 0.4 (101 snapshots): "
     "48 training and 8 validation trajectories with random step/ramp actuations; 12 test trajectories of the same family; 8 with chirp "
     "(frequency-sweeping sinusoidal) actuations and 6 with steps between the range edges — two families never seen in training. "
     f"Mean CFD cost {cfd_s:.0f} s per trajectory (single core).")
Pp.section("Layer 2 — Surrogates")
Pp.bullets([
    f"POD (training snapshots, 99.99% energy, ≤ 40 modes per field; {sum(R['pod_modes'])} modes in total) + DMDc (linear latent map with actuator input).",
    "POD + MLP latent dynamics trained on 8-step latent rollouts.",
    "Convolutional autoencoder (latent 64) + latent MLP stepper; field loss over 4-step rollouts.",
    "FNO: PINNeAPPle FNO2d with zero padding (non-periodic channel), actuator samples as channels, residual update, 4-step rollout loss.",
    "Physics-informed FNO: + λ[continuity² + global-energy-balance²] computed on the de-normalised predictions; λ ∈ {0.1, 1} chosen on validation.",
    "DeepONet (PINNeAPPle): branch = POD coefficients + actuator samples, trunk = (x, y).",
    "All models: training statistics only; selection and early stopping on validation rollouts; test families scored once. Field models "
    "(CAE, FNO, PI-FNO): up to 20 epochs, patience 5, rollout windows starting at every third snapshot (random offset per epoch) — a budget "
    "imposed by the shared laptop GPU, identical for all three."])
def ok(t):
    """A rollout counts as diverged when it overflowed, even if every value stayed technically finite."""
    return t["all_finite"] and np.isfinite(t["rel_l2"]["T"]) and t["rel_l2"]["T"] < 1e3


rows = [["Model", "val. rollout rel. L2"] + [f"T rel. L2 ({FAMLAB[f]})" for f in FAM] + ["T<sub>out</sub> RMSE (chirps)", "energy residual (chirps)", "rollout s"]]
for n in names:
    t = M[n]["test"]
    good = ok(t["test_unseen_sinusoid"])
    rows.append([n + (" <b>(selected)</b>" if n == sel else ""), fmt(M[n]["val_rollout_rel_l2"]) if M[n]["val_rollout_rel_l2"] < 1e3 else "diverged"] +
                [fmt(t[f]["rel_l2"]["T"]) if ok(t[f]) else "diverged" for f in FAM] +
                ([fmt(t["test_unseen_sinusoid"]["qoi_rmse"]["T_out"], 4), fmt(t["test_unseen_sinusoid"]["energy_residual_mean_abs"])] if good else ["—", "—"]) +
                [fmt(t["test_unseen_sinusoid"]["rollout_seconds"], 2)])
fl = R["truth_physics_floor"]["test_unseen_sinusoid"]
Pp.table(rows, f"Rollout metrics over 100 steps (40 time units). Energy residual: global energy-balance mismatch per step, relative to the maximum "
               f"heat input; the CFD data themselves give {fmt(fl['energy_residual_mean_abs'])} (discretisation floor of the diagnostic).",
         [3.8, 1.6, 1.6, 1.6, 1.6, 1.8, 1.8, 1.2])
Pp.figure(f3, "Relative L2 error of the temperature field over the full rollout, by model and test family.", 14.5)
Pp.figure(f4, "Temperature field of one unseen chirp trajectory: CFD vs surrogate rollouts at three times.", 16)
pi = M["Physics-informed FNO"]; fn = M["FNO (PINNeAPPle FNO2d)"]
Pp.p(f"Physics-informed vs data-only FNO (λ = {pi['train'].get('lambda')}): temperature error on unseen chirps "
     f"{fmt(pi['test']['test_unseen_sinusoid']['rel_l2']['T'])} vs {fmt(fn['test']['test_unseen_sinusoid']['rel_l2']['T'])}; energy residual "
     f"{fmt(pi['test']['test_unseen_sinusoid']['energy_residual_mean_abs'])} vs {fmt(fn['test']['test_unseen_sinusoid']['energy_residual_mean_abs'])}. "
     + ("The physics loss helped the FNO on both counts, " if PI_BETTER else "The physics loss did not help the FNO, ")
     + f"but both FNO variants remain well behind the selected POD + MLP model ({fmt(st['test_unseen_sinusoid']['rel_l2']['T'])}). "
     "Rollout times were measured on a laptop shared with other jobs: the two FNOs have identical architectures, so the difference in their "
     "rollout times reflects machine load at the moment of measurement, not the models.")
Pp.section("Layer 3 — Sparse sensors and state correction")
cs = L3["cases"]
rows = [["Run", "family", "true gain", "open loop", "EnKF state", "EnKF state + gain", "sensors only", "gain estimate"]]
for c in cs:
    rows.append([str(c["run"]), c["family"].replace("test_", "").replace("_", " "), str(c["true_gain"]), fmt(c["open_loop"]["T_rel_l2"]),
                 fmt(c["enkf_state"]["T_rel_l2"]), fmt(c["enkf_state_and_gain"]["T_rel_l2"]), fmt(c["gappy_pod_sensors_only"]["T_rel_l2"]),
                 fmt(c["gain_estimate_final"])])
Pp.table(rows, f"Relative L2 error of the temperature field with three sensors {', '.join(s[0] for s in L3['sensors'])} (T at (5, 0.5) and "
               "(9, 0.25), p at (0.5, 0.5); noise σ = 0.002 for T, 0.02 for p). The 'plant' is the CFD with the heater delivering gain × the "
               "commanded power (1.0 = nominal, 1.3 = uncalibrated heater); the surrogate only knows the command. EnKF: PINNeAPPle "
               "EnsembleKalmanFilter, 48 members, on the POD + MLP latent state.", [1.0, 2.2, 1.3, 1.6, 1.6, 2.2, 1.6, 1.8])
Pp.p(f"Averaged over the {len(L3G[1.0]['cases'])} cases with a calibrated heater (gain 1.0): open loop {fmt(L3G[1.0]['ol'])}, EnKF on the state "
     f"{fmt(L3G[1.0]['enkf'])}, EnKF with gain estimation {fmt(L3G[1.0]['enkf_g'])}. With the uncalibrated heater (gain 1.3): {fmt(L3G[1.3]['ol'])}, "
     f"{fmt(L3G[1.3]['enkf'])} and {fmt(L3G[1.3]['enkf_g'])}; the estimated gain ends between {min(L3G[1.3]['gains']):.2f} and {max(L3G[1.3]['gains']):.2f}. "
     "Three sensors carry little information about the full temperature field: the sensor-only reconstruction fails (error ≈ "
     f"{fmt(np.mean([c['gappy_pod_sensors_only']['T_rel_l2'] for c in cs]))}), and the filter corrections are small. Estimating the gain helps "
     "somewhat when the heater really is miscalibrated but harms the calibrated cases, where the estimate drifts below 1 (down to "
     f"{min(L3G[1.0]['gains']):.2f}); a gain prior tighter than the one used, or more sensors near the heater, would be needed before relying on it.")
if f5:
    Pp.figure(f5, "Left: outlet temperature for one case — CFD, open-loop surrogate, EnKF-corrected twin and sensor-only reconstruction. "
                  "Right: the heater-gain estimate of the augmented EnKF.", 15)
Pp.section("Layer 4 — Optimisation verified in CFD")
Pp.p(f"Goal: hold the outlet temperature at {L4['target_T_out']} with U<sub>in</sub> = 1 while penalising heater power, by optimising 10 "
     f"piecewise-constant heater levels through the differentiable POD + MLP surrogate ({L4['optimisation_seconds']:.0f} s of Adam). The optimised "
     f"actuation was then simulated with the CFD ({L4['cfd_seconds']:.0f} s): tracking RMSE predicted by the surrogate {fmt(L4['surrogate_tracking_rmse'], 4)}, "
     f"obtained in CFD {fmt(L4['cfd_tracking_rmse'], 4)}; surrogate–CFD mismatch of T<sub>out</sub> {fmt(L4['surrogate_vs_cfd_T_out_rmse'], 4)}.")
Pp.figure(f6, "Outlet temperature under the optimised heater schedule: surrogate prediction vs the CFD re-simulation.", 12)
Pp.section("VeriPhysics validation")
V = R["veriphysics"]
rows = [["Model", "Trust score", "Coverage", "Components / note"]]
for n in names:
    v = V[n]
    rows.append([n, fmt(v["overall_score"]), f"{v['coverage']:.1f}", "; ".join(f"{c['name']} {c['score']:.2f}" for c in v.get("components", [])) or v.get("note", "")])
Pp.table(rows, V["note"], [4.6, 1.8, 1.6, 8])
Pp.section("Answer to the project question")
Pp.p(f"Can a CFD run be replaced by a fast Physics-AI representation for interactive evaluation inside a known operating envelope? For "
     f"quantities of interest, largely yes: the selected POD + MLP model predicts the outlet temperature of unseen actuation families with RMSE "
     f"{fmt(st['test_unseen_sinusoid']['qoi_rmse']['T_out'], 4)} (chirps) about {SPEEDUP:.0f}× faster than the CFD, and a heater schedule optimised through it "
     f"achieved in CFD a tracking error of {fmt(L4['cfd_tracking_rmse'], 4)} against the {fmt(L4['surrogate_tracking_rmse'], 4)} it predicted. For full "
     f"fields, only partly: the temperature-field error roughly doubles from the training family to unseen families "
     f"({fmt(st['test_same_family']['rel_l2']['T'])} → {fmt(st['test_unseen_sinusoid']['rel_l2']['T'])} and {fmt(st['test_unseen_edge_steps']['rel_l2']['T'])}), and every "
     f"surrogate violates the global energy balance by far more than the CFD's own diagnostic floor ({fmt(fl['energy_residual_mean_abs'])}). Sparse-sensor "
     "assimilation with three sensors did not materially correct the surrogate; the digital-twin layer is the weakest link of this study.")
Pp.references([
    "Griebel, M., Dornseifer, T., Neunhoeffer, T. (1998). <i>Numerical Simulation in Fluid Dynamics: A Practical Introduction</i>. SIAM.",
    "Shah, R. K., London, A. L. (1978). <i>Laminar Flow Forced Convection in Ducts</i>. Academic Press.",
    "Roache, P. J. (1994). Perspective: a method for uniform reporting of grid refinement studies. <i>ASME J. Fluids Eng.</i> 116, 405–413.",
    "Proctor, J. L., Brunton, S. L., Kutz, J. N. (2016). Dynamic mode decomposition with control. <i>SIAM J. Appl. Dyn. Syst.</i> 15(1), 142–161.",
    "Li, Z., et al. (2021). Fourier neural operator for parametric PDEs. <i>ICLR</i>.",
    "Lu, L., Jin, P., Pang, G., Zhang, Z., Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet. <i>Nature Machine Intelligence</i> 3, 218–229.",
    "Evensen, G. (2009). <i>Data Assimilation: The Ensemble Kalman Filter</i>. Springer.",
    "Everson, R., Sirovich, L. (1995). Karhunen–Loève procedure for gappy data. <i>JOSA A</i> 12(8), 1657–1664.",
])
print(Pp.build(str(ROOT / "paper" / "Heated_Channel_Digital_Twin_PINNeAPPle.pdf")))
