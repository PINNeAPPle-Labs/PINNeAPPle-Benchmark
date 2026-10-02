"""Physics Digital Twin paper -- every number read from results/."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "paperkit"))
from paperkit import CAT, INK2, MUTED, Paper, fmt, savefig, setup_mpl  # noqa: E402

setup_mpl()
R = json.loads((ROOT / "results" / "results.json").read_text())
FIG = ROOT / "figures"
T = R["test"]
MODELS = ["Persistence", "Linear (Ridge)", "Random Forest", "Gradient Boosting", "MLP", "Physics ODE (identified)",
          "NN (no physics loss)", "PINN (PINNeAPPle)"]
d = pd.read_csv(ROOT / "data" / "weather_dataset.csv", parse_dates=["time"])
tp = np.load(ROOT / "results" / "test_predictions.npz")

# ---- Fig 1: data overview (one week) + diurnal cycle
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0), gridspec_kw={"width_ratios": [2, 1]})
wk = d[(d.time >= "2023-03-01") & (d.time < "2023-03-08")]
ax[0].plot(wk.time, wk.soil_temperature_0_to_7cm, color=CAT[0], label="facility thermal mass T (0–7 cm)")
ax[0].plot(wk.time, wk.temperature_2m, color=CAT[1], label="ambient air $T_a$")
ax2 = None
ax[0].set_ylabel("°C"); ax[0].set_title("Virtual-sensor telemetry (one test week)"); ax[0].legend(fontsize=7)
ax[0].tick_params(axis="x", labelrotation=30, labelsize=7)
h = d.time.dt.hour
ax[1].plot(range(24), d.groupby(h).shortwave_radiation.mean(), color=CAT[3])
ax[1].set_xlabel("hour of day"); ax[1].set_ylabel("mean S (W/m²)"); ax[1].set_title("Solar forcing (2015–2024)")
f1 = savefig(fig, FIG / "fig1_data.png")

# ---- Fig 2: RMSE vs horizon
fig, ax = plt.subplots(figsize=(6.4, 3.2))
hz = [1, 3, 6]
for j, m in enumerate(MODELS):
    ax.plot(hz, [T[m]["metrics"][f"h{h}"]["RMSE"] for h in hz], "-o", ms=3.5, color=CAT[j % 8] if m != "Persistence" else MUTED,
            label=m, lw=2.2 if m == "PINN (PINNeAPPle)" else 1.3)
ax.set_xlabel("horizon (h)"); ax.set_ylabel("RMSE (°C), test 2023–2024"); ax.set_title("Prediction error vs horizon"); ax.set_xticks(hz)
ax.legend(fontsize=6.6, ncol=2)
f2 = savefig(fig, FIG / "fig2_rmse.png")

# ---- Fig 3: example forecasts
st = tp["start"]
k = int(np.searchsorted(st, int(np.where(d.time >= "2023-10-10 09:00")[0][0])))
sl = slice(max(k - 6, 0), k + 1)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
idx0 = st[k]
tt = d.time.iloc[idx0 - 12: idx0 + 7]
ax.plot(tt, d.soil_temperature_0_to_7cm.iloc[idx0 - 12: idx0 + 7], color=INK2, lw=2, label="observed")
fut = d.time.iloc[idx0 + 1: idx0 + 7]
for j, (key, lab) in enumerate((("PINN_PINNeAPPle", "PINN"), ("Physics_ODE_identified", "physics ODE"), ("Gradient_Boosting", "gradient boosting"))):
    if key in tp.files:
        ax.plot(fut, tp[key][k], "--o", ms=3, color=CAT[j], label=lab)
ax.axvline(d.time.iloc[idx0], color=MUTED, lw=0.8)
ax.set_ylabel("T (°C)"); ax.set_title("A 6-h forecast from 09:00 (test period)"); ax.legend(fontsize=7); ax.tick_params(axis="x", labelrotation=30, labelsize=7)
f3 = savefig(fig, FIG / "fig3_example.png")

# ---- Fig 4: parameter identification
fig, ax = plt.subplots(1, 4, figsize=(8.6, 2.5))
hist = R["pinn_sweep"][str(R["pinn_lambda_selected"])]["history"]
names = [("a", "a (K h$^{-1}$ per W m$^{-2}$)"), ("b0", "$b_0$ (h$^{-1}$)"), ("b1", "$b_1$ (h$^{-1}$ per m s$^{-1}$)"), ("q", "q (K h$^{-1}$)")]
ls = R["physics_ode"]["params"]; ek = R["eki"]["mean"]; es = R["eki"]["std"]
for j, (kk, lab) in enumerate(names):
    ax[j].plot([x["epoch"] for x in hist], [x[kk] for x in hist], color=CAT[0], label="PINN (trained jointly)")
    ax[j].axhline(ls[kk], color=CAT[1], ls="--", label="least squares (ODE)")
    ax[j].axhspan(ek[kk] - 2 * es[kk], ek[kk] + 2 * es[kk], color=CAT[2], alpha=0.3, label="EKI ±2σ")
    yr = [v[kk] for v in R["physics_ode"]["per_year"].values()]
    ax[j].plot([hist[-1]["epoch"]] * len(yr), yr, "x", color=INK2, ms=4, label="LS per training year")
    ax[j].set_title(lab, fontsize=8); ax[j].set_xlabel("epoch", fontsize=7); ax[j].tick_params(labelsize=6.5)
ax[0].legend(fontsize=5.8)
f4 = savefig(fig, FIG / "fig4_params.png")

# ---- Fig 5: what-if
W = R["whatif"]["scenarios"]
wm = ["Physics ODE (identified)", "PINN (PINNeAPPle)", "NN (no physics loss)", "Gradient Boosting", "MLP"]
fig, ax = plt.subplots(figsize=(7.6, 3.1))
x = np.arange(len(W)); wdt = 0.16
for j, m in enumerate(wm):
    ax.bar(x + (j - 2) * wdt, [W[s][m]["mean_dT6"] for s in W], width=wdt - 0.01, color=CAT[j], label=m)
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(list(W), fontsize=7); ax.set_ylabel("mean ΔT at +6 h (K)")
ax.set_title("What-if responses (reference: identified physics hypothesis)"); ax.legend(fontsize=6.5, ncol=2)
f5 = savefig(fig, FIG / "fig5_whatif.png")

# ---- Fig 6: anomaly example
ex = R["anomaly"]["example"]
fig, ax = plt.subplots(figsize=(6.6, 2.8))
tt = d.time.iloc[ex["t"]]
ax.plot(tt, ex["T_clean"], color=INK2, lw=1.6, label="true signal")
ax.plot(tt, ex["T_corrupt"], color=CAT[1], lw=1.2, label="sensor with injected drift")
stt = np.array(ex["status"])
for lvl, c, lab in ((1, "#fab219", "WARNING"), (2, "#d03b3b", "ANOMALY")):
    sel = stt == lvl
    ax.scatter(tt[sel], np.array(ex["T_corrupt"])[sel], color=c, s=14, zorder=3, label=lab)
ax.set_ylabel("T (°C)"); ax.set_title("Twin-residual anomaly detection (injected fault)"); ax.legend(fontsize=6.8)
ax.tick_params(axis="x", labelrotation=30, labelsize=7)
f6 = savefig(fig, FIG / "fig6_anomaly.png")

# ======================================================================== paper
Pp = Paper("A Physics-AI Digital Twin with PINNeAPPle: Learning the Thermal Dynamics of a Facility from Public Weather Telemetry",
           "Virtual sensors, a physics-informed neural network with parameter identification, anomaly detection, what-if simulation and decision support",
           report_id="Experiment 02")
pin, ode, gb, per = T["PINN (PINNeAPPle)"]["metrics"], T["Physics ODE (identified)"]["metrics"], T["Gradient Boosting"]["metrics"], T["Persistence"]["metrics"]
best_ml = min(("Linear (Ridge)", "Random Forest", "Gradient Boosting", "MLP"), key=lambda m: T[m]["metrics"]["h6"]["RMSE"])
lam = R["pinn_lambda_selected"]
Pp.abstract(
    f"We build the digital twin proposed in the PINNeAPPle Physics-AI challenge: public weather data (Open-Meteo, Natal–RN, 2015–2024, hourly) "
    f"are replayed as virtual sensor telemetry; the facility is represented by a thermal-mass element obeying the energy balance "
    f"C dT/dt = αS − k(W)(T − T<sub>a</sub>) + Q; and a physics-informed neural network (PINNeAPPle ModifiedMLP) learns the dynamics while identifying "
    f"the physical parameters. Training uses 2015–2021, model and weight selection 2022, and a single test on 2023–2024. The PINN "
    f"(λ = {lam} selected on validation) predicts the facility temperature with RMSE {fmt(pin['h1']['RMSE'])}, {fmt(pin['h3']['RMSE'])} and "
    f"{fmt(pin['h6']['RMSE'])} °C at 1, 3 and 6 h, against {fmt(per['h6']['RMSE'])} °C for persistence, {fmt(ode['h6']['RMSE'])} °C for the identified "
    f"physics ODE alone and {fmt(T[best_ml]['metrics']['h6']['RMSE'])} °C for the best purely data-driven model ({best_ml}). The parameters identified "
    f"by the PINN agree with least squares and with ensemble Kalman inversion. Anomaly detection uses the twin residual — the incompatibility "
    f"between observation and the physically expected state — and is evaluated with injected faults. What-if scenarios (solar, wind, ambient "
    f"temperature, internal heat) are answered by the PINN in agreement with the physics hypothesis, including internal heat, which never "
    f"varies in the data; data-driven models cannot respond to it. We state plainly which parts of the challenge the data cannot support: "
    f"Open-Meteo provides reanalysis, not sensor readings, and no facility temperature exists, so the thermal-mass proxy is a modelling choice.",
    ["digital twin", "physics-informed neural networks", "parameter identification", "anomaly detection", "what-if simulation",
     "Open-Meteo", "PINNeAPPle", "VeriPhysics"])
Pp.box("Honest scope", [
    "No physical sensors exist: the 'sensors' replay ERA5/ERA5-Land reanalysis served by Open-Meteo.",
    "No indoor facility temperature exists in the data. The state T is the 0–7 cm soil temperature (ERA5-Land), used as the facility's "
    "thermal mass; it is heated by the sun and exchanges heat with the ambient air, which is the structure of the proposed energy balance.",
    "Only α/C, k/C and Q/C are identifiable from temperatures; absolute α, k, Q are quoted for an assumed C (moist soil, 0.07 m).",
    "What-if answers are model predictions under the stated physics hypothesis; counterfactuals are not observable in the data."])

Pp.section("System and data")
Pp.p("Open-Meteo's historical API (ERA5 / ERA5-Land reanalysis) was queried for Natal–RN (−5.79°, −35.21°) from 2015-01-01 to 2024-12-31: "
     f"{R['data']['n_rows']:,} hourly records, no missing values, with temperature, relative humidity, surface pressure, wind speed and direction, "
     "precipitation, shortwave radiation and soil temperature. Each variable is exposed as a PINNeAPPle <i>Sensor</i> in a <i>SensorRegistry</i>; "
     "a <i>VirtualSensorStream</i> replays calibrated <i>Observation</i>s (timestamp, value, metadata) in order and persists them to SQLite "
     "(Figure 1).")
Pp.figure(f1, "Left: the state variable (thermal mass) and the ambient air during one test week. Right: mean diurnal solar forcing.", 15.5)
Pp.section("Physics model")
Pp.eq("C dT/dt = α S − (k<sub>0</sub> + k<sub>1</sub> W)(T − T<sub>a</sub>) + Q<sub>int</sub>")
Pp.eq("dT/dt = a S − (b<sub>0</sub> + b<sub>1</sub>W)(T − T<sub>a</sub>) + q + Δq,     a = α/C,  b = k/C,  q = Q/C  (per hour)")
Pp.p("S is the hourly-mean shortwave radiation (piecewise constant), T<sub>a</sub> and W are linearly interpolated inside each hour, and Δq is "
     "a what-if internal-heat increment that is zero in all data. Integration uses RK4 with four sub-steps per hour; its discretisation error "
     "is verified by a three-level Richardson study (Section 6).")
Pp.section("Models")
Pp.bullets([
    "Task: from the state at hour n and the forcing over [n, n+6] (weather assumed known, e.g. from a forecast), predict T at n+1…n+6. "
    "All learned models receive the same 37-number context.",
    "Baselines (no explicit physics): persistence, ridge regression, random forest, histogram gradient boosting, MLP — each tuned on 2022.",
    "Physics ODE: parameters identified by 6-h multiple-shooting least squares (L-BFGS) on 2015–2021; cross-checked by PINNeAPPle "
    "EnsembleKalmanInversion and by re-identification on each training year.",
    "PINN: T(τ) = T<sub>n</sub> + (τ/6)·N([context, τ/6]) with N a PINNeAPPle ModifiedMLP, so the initial condition holds exactly; loss = data MSE "
    "at τ = 1…6 + λ·mean squared residual of the ODE at random collocation times, on observed contexts and on perturbed, unlabelled contexts "
    "(solar ×0.5–1.6, wind ×0.2–2, ambient ±3 K, internal heat 0–150 W/m²). a, b<sub>0</sub>, b<sub>1</sub>, q are trainable and start from "
    "neutral values, not from the least-squares answer. λ ∈ {0, 0.1, 1, 10} selected on 2022; λ = 0 is the same network without physics."])
Pp.section("Results")
rows = [["Model", "RMSE 1 h", "RMSE 3 h", "RMSE 6 h", "95% CI 6 h", "MAE 6 h", "rel. L2 of ΔT, 6 h", "physics residual (K/h)"]]
for m in MODELS:
    t = T[m]
    rows.append([m, fmt(t["metrics"]["h1"]["RMSE"]), fmt(t["metrics"]["h3"]["RMSE"]), fmt(t["metrics"]["h6"]["RMSE"]),
                 f"{fmt(t['rmse_h6_ci95'][0])}–{fmt(t['rmse_h6_ci95'][1])}", fmt(t["metrics"]["h6"]["MAE"]),
                 fmt(t["metrics"]["h6"]["rel_L2_dT"]), fmt(t["physics_residual_discrete_mean_abs"])])
Pp.table(rows, "Test period 2023–2024 (°C). CI: weekly block bootstrap. Physics residual: mean |T̂(h) − Φ<sub>1h</sub>(T̂(h−1))| under the identified ODE; "
               f"the observations themselves have {fmt(R['observed_physics_residual']['mean_abs_K_per_h'])} K/h (model-mismatch floor).",
         [3.6, 1.4, 1.4, 1.4, 2.1, 1.4, 1.9, 2.2], highlight_rows=[8])
Pp.figure(f2, "RMSE versus horizon for all models (test).", 12.5)
Pp.figure(f3, "One 6-h forecast issued at 09:00 in the test period.", 12)
Pp.subsection("Overfitting check")
G = R["generalization"]
Pp.table([["Model", "train RMSE", "validation RMSE", "test RMSE"]] + [[m, fmt(v["train"]), fmt(v["val"]), fmt(v["test"])] for m, v in G.items()],
         "RMSE over all horizons in each period (train: one window in seven). Large train–test gaps flag overfitting.", [6, 3, 3, 3])
sp = R["seed_spread"]
Pp.p(f"Seed robustness (3 seeds, test RMSE at 6 h): PINN {fmt(sp['pinn']['mean'])} ± {fmt(sp['pinn']['std'])}; "
     f"same network without physics {fmt(sp['nn']['mean'])} ± {fmt(sp['nn']['std'])} °C.")
Pp.subsection("Parameter identification")
ph = R["physics_ode"]["physical"]
pp = R["pinn_sweep"][str(lam)]["params"]
Pp.table([["Parameter", "PINN (joint)", "Least squares", "EKI mean ± sd", "Per-year LS range"]] + [
    [lab, f"{pp[j]:.4g}", f"{ls[kk]:.4g}", f"{ek[kk]:.4g} ± {es[kk]:.2g}",
     f"{min(v[kk] for v in R['physics_ode']['per_year'].values()):.3g} – {max(v[kk] for v in R['physics_ode']['per_year'].values()):.3g}"]
    for j, (kk, lab) in enumerate(names)],
    f"Identified parameters. With C = {ph['C_J_m2K']:.3g} J m⁻² K⁻¹ (assumed), least squares gives α = {ph['alpha']:.3f}, "
    f"k<sub>0</sub> = {ph['k0_W_m2K']:.1f} W m⁻² K⁻¹, k<sub>1</sub> = {ph['k1_W_m2K_per_ms']:.2f} W m⁻² K⁻¹ per m/s, Q = {ph['Q_int_W_m2']:.1f} W m⁻².",
    [3.2, 2.4, 2.4, 3.2, 3.6])
Pp.figure(f4, "Parameter trajectories of the PINN during training versus least squares, EKI (±2σ) and per-year least squares.", 15.8)
Pp.p("The absorbed fraction α is small because the 0–7 cm soil layer also conducts heat downward, which the single-node model folds into "
     "the other terms; the year-to-year spread of the least-squares parameters is larger than the EKI posterior width, i.e. the physics "
     "hypothesis is an approximation whose parameters drift, which we report rather than hide.")
Pp.subsection("Extrapolation to an unseen radiation regime")
ex = R["extrapolation"]
Pp.table([["Model", "RMSE 1 h", "RMSE 6 h"]] + [[m, fmt(v["h1"]["RMSE"]), fmt(v["h6"]["RMSE"])] for m, v in ex["results"].items()],
         f"Trained on windows with max S ≤ 750 W/m² ({ex['n_train']:,}), tested on windows with max S > 900 W/m² ({ex['n_test']:,}).", [6, 3, 3])
Pp.subsection("What-if simulation")
Pp.figure(f5, "Mean 6-h temperature change for each scenario (600 morning states of the test period).", 14.5)
rows = [["Scenario", "Model", "mean ΔT₆ (K)", "|ΔT − physics|", "sign agreement"]]
for s, blk in W.items():
    for m in ("PINN (PINNeAPPle)", "NN (no physics loss)", "Gradient Boosting"):
        rows.append([s, m, fmt(blk[m]["mean_dT6"]), fmt(blk[m]["mae_vs_physics"]), fmt(blk[m]["sign_agreement"], 2)])
Pp.table(rows, "What-if agreement with the physics hypothesis (the only available reference for counterfactuals).", [4.6, 4.2, 2.2, 2.4, 2.4])
Pp.p("Data-driven models never saw the internal-heat input vary, so they cannot respond to it (ΔT ≈ 0), and tree models cannot extrapolate "
     "radiation beyond the observed range. The PINN responds because the physics residual was imposed on unlabelled perturbed contexts: "
     "this is the concrete advantage of a physics model as a scenario engine — and it is only as good as the hypothesis it encodes.")
Pp.subsection("Anomaly detection")
an = R["anomaly"]
Pp.figure(f6, "An injected sensor drift: the twin flags the observation as incompatible with the expected physical state.", 12.5)
rows = [["Fault type", "events", "recall (WARNING or ANOMALY)", "recall (ANOMALY)", "recall (naive band)", "median delay (h)"]] + [
    [k, str(v["n"]), fmt(v["recall_twin_warn_or_anom"], 2), fmt(v["recall_twin_anomaly"], 2), fmt(v["recall_naive"], 2),
     str(v["median_delay_h"])] for k, v in an["by_type"].items()]
Pp.table(rows, f"Injected faults on the test period. False alarms on {an['n_clean_hours']:,} clean hours: twin "
               f"{100 * an['false_alarm_rate']['twin_WARNING_or_ANOMALY']:.2f}% (WARNING or ANOMALY), {100 * an['false_alarm_rate']['twin_ANOMALY']:.2f}% "
               f"(ANOMALY); naive 'temperature outside the 0.5–99.5% band' detector {100 * an['false_alarm_rate']['naive_band']:.2f}%. "
               f"Thresholds (validation quantiles 99% / 99.9%): 1 h {fmt(an['thresholds_K']['warn_1h'])} / {fmt(an['thresholds_K']['anom_1h'])} °C, "
               f"6 h {fmt(an['thresholds_K']['warn_6h'])} / {fmt(an['thresholds_K']['anom_6h'])} °C.", [2.4, 1.4, 3.2, 2.4, 2.4, 2.4])
Pp.section("VeriPhysics validation")
V = R["veriphysics"]
rows = [["Model", "Trust score", "Coverage", "Components"]]
for m in ("PINN (PINNeAPPle)", "Physics ODE (identified)", "Gradient Boosting", "NN (no physics loss)"):
    rows.append([m, fmt(V[m]["overall_score"]), f"{V[m]['coverage']:.1f}", "; ".join(f"{c['name']} {c['score']:.2f}" for c in V[m]["components"])])
Pp.table(rows, f"PINNeAPPle compute_physics_confidence. Benchmark: observed 6-h change on 2023–2024 (registered as a named benchmark). "
               f"Numerical convergence: Richardson/GCI of the RK4 simulator (observed order {fmt(V['convergence']['observed_order'], 2)}, "
               f"GCI {V['convergence']['gci_fine']:.1e}). Not run: {', '.join(V['not_run'])}.", [4.2, 2, 1.6, 8.4])
Pp.section("Challenge checklist")
Pp.table([["Challenge item", "Where"], ["Data (weather_dataset.csv)", "data/, src/fetch_openmeteo.py"],
          ["Virtual sensors, streaming, SQLite", "src/virtual_sensors.py (PINNeAPPle Sensor/Observation)"],
          ["Baseline ML", "Section 5, Table 1"], ["Physics model", "Section 3"], ["PINN (data + physics loss)", "Section 4"],
          ["Parameter identification", "Section 5.2"], ["Validation: MAE, RMSE, relative error, physics residual", "Table 1, Section 6"],
          ["Anomaly detection NORMAL/WARNING/ANOMALY", "Section 5.5"], ["What-if (solar, wind, ambient, internal heat)", "Section 5.4, app"],
          ["Insights and recommendations", "app/streamlit_app.py (energy-balance term decomposition)"],
          ["Arena-style comparison ML × PINN", "Tables 1–4"], ["Streamlit application", "app/streamlit_app.py"]],
         "Mapping of the challenge deliverables to this work.", [7, 9])
Pp.references([
    "Open-Meteo Historical Weather API, https://open-meteo.com (ERA5 / ERA5-Land reanalysis).",
    "Hersbach, H., et al. (2020). The ERA5 global reanalysis. <i>Quarterly Journal of the Royal Meteorological Society</i> 146, 1999–2049.",
    "Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). Physics-informed neural networks. <i>Journal of Computational Physics</i> 378, 686–707.",
    "Wang, S., Teng, Y., Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. <i>SIAM J. Sci. Comput.</i> 43(5).",
    "Iglesias, M. A., Law, K. J. H., Stuart, A. M. (2013). Ensemble Kalman methods for inverse problems. <i>Inverse Problems</i> 29, 045001.",
    "Roache, P. J. (1994). Perspective: a method for uniform reporting of grid refinement studies. <i>ASME J. Fluids Eng.</i> 116, 405–413.",
])
print(Pp.build(str(ROOT / "paper" / "Physics_Digital_Twin_PINNeAPPle.pdf")))
