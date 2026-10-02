"""PINNeAPPle Physics Digital Twin -- Streamlit demonstrator.

Run:  streamlit run app/streamlit_app.py   (after src/run_experiment.py has produced results/)

Panels: virtual sensors (replayed Open-Meteo telemetry) -> twin state -> prediction vs observation ->
physics validation -> anomaly status -> what-if scenario -> physics-based insights -> suggested actions.
Every number shown is computed live from the trained PINN / identified ODE; suggestions are decision
support, never executed automatically.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parents[2] / "PINNeAPPle"))
from twin_core import (C_ASSUMED, H, PhysicsModel, Standardizer, ThermalParams, ThermalPINN, Windows,  # noqa: E402
                       load_weather, make_windows, physics_residual, to_t)
from virtual_sensors import VirtualSensorStream  # noqa: E402

st.set_page_config(page_title="PINNeAPPle Physics Digital Twin", layout="wide")


@st.cache_resource
def load_all():
    d = load_weather()
    R = json.loads((ROOT / "results" / "results.json").read_text())
    ck = torch.load(ROOT / "results" / "pinn_twin.pt", weights_only=False)
    m = ThermalPINN(ctx_dim=ck["ctx_dim"])
    m.load_state_dict(ck["state"]); m.eval()
    stdz = Standardizer(np.zeros((2, ck["ctx_dim"]), np.float32)); stdz.mu, stdz.sd = ck["mu"], ck["sd"]
    pm = PhysicsModel(ThermalParams(**json.loads((ROOT / "results" / "physics_params.json").read_text())))
    return d, R, m, stdz, pm


d, R, model, stdz, pm = load_all()
thr = R["anomaly"]["thresholds_K"]


def pinn_predict(w: Windows):
    T0, *_ = to_t(w)
    with torch.no_grad():
        return model(torch.tensor(stdz(w.context())), T0, torch.arange(1, H + 1, dtype=torch.float32).expand(len(w), H)).numpy()


def residual(w: Windows):
    T0, Ta, W, S, dq, _ = to_t(w)
    r, _ = physics_residual(model, torch.tensor(stdz(w.context())), T0, Ta, W, S, dq, torch.rand(len(w), 16) * H)
    return float(r.detach().abs().mean())


# ------------------------------------------------------------------ sidebar: stream control
st.sidebar.title("🍍 PINNeAPPle Twin")
test_start = int(np.where(d.time.dt.year >= 2023)[0][0])
idx = st.sidebar.slider("Replay position (test period, hours)", 12, 24 * 365 * 2 - 12, 24 * 200 + 10)
i = test_start + idx
inject = st.sidebar.checkbox("Inject a +5 °C sensor fault at the current hour (demo)")
stream = VirtualSensorStream(d)
obs = {o.sensor_id: (next(iter(o.values.values())), o.metadata["unit"]) for o in stream.observations_at(i)}
T_obs = obs["T_facility"][0] + (5.0 if inject else 0.0)

# ------------------------------------------------------------------ state
w_now = make_windows(d, np.array([i - 1]))
w_6 = make_windows(d, np.array([i - 6]))
p1, p6 = pinn_predict(w_now)[0, 0], pinn_predict(w_6)[0, 5]
r1, r6 = T_obs - p1, T_obs - p6
status = "ANOMALY" if (abs(r1) > thr["anom_1h"] or abs(r6) > thr["anom_6h"]) else (
    "WARNING" if (abs(r1) > thr["warn_1h"] or abs(r6) > thr["warn_6h"]) else "NORMAL")
icon = {"NORMAL": "✅", "WARNING": "⚠️", "ANOMALY": "🚨"}[status]

st.title("PINNeAPPle Physics Digital Twin — Natal (RN)")
st.caption(f"Virtual sensors replay Open-Meteo (ERA5/ERA5-Land) data as telemetry · {pd.Timestamp(d.time[i]):%Y-%m-%d %H:%M} local time")
c = st.columns(6)
c[0].metric("Facility T (thermal mass)", f"{T_obs:.1f} °C")
c[1].metric("Ambient T", f"{obs['T_ambient'][0]:.1f} °C")
c[2].metric("Humidity", f"{obs['RH'][0]:.0f} %")
c[3].metric("Wind", f"{obs['WS'][0]:.1f} km/h")
c[4].metric("Solar", f"{obs['SW'][0]:.0f} W/m²")
c[5].metric("Twin status", f"{icon} {status}")

# ------------------------------------------------------------------ prediction vs observation
st.subheader("Prediction vs observation (next 6 h)")
wf = make_windows(d, np.array([i]))
pred = pinn_predict(wf)[0]
ode = pm.predict(wf)[0]
hist = d.iloc[i - 24:i + 7]
chart = pd.DataFrame({"observed": hist.soil_temperature_0_to_7cm.values}, index=hist.time)
chart["PINN forecast"] = np.nan; chart["physics ODE"] = np.nan
chart.iloc[-7:, 1] = np.r_[d.soil_temperature_0_to_7cm.iloc[i], pred]
chart.iloc[-7:, 2] = np.r_[d.soil_temperature_0_to_7cm.iloc[i], ode]
st.line_chart(chart)
met = R["test"]["PINN (PINNeAPPle)"]["metrics"]
st.caption(f"Held-out 2023–2024 accuracy of this PINN: RMSE {met['h1']['RMSE']:.2f} / {met['h3']['RMSE']:.2f} / {met['h6']['RMSE']:.2f} °C at 1 / 3 / 6 h.")

# ------------------------------------------------------------------ physics validation
st.subheader("Model validation")
v = st.columns(4)
v[0].metric("Residual now (1 h)", f"{r1:+.2f} °C")
v[1].metric("Residual now (6 h)", f"{r6:+.2f} °C")
v[2].metric("Physics residual |dT/dt − f|", f"{residual(wf):.3f} K/h")
p = model.params()
v[3].metric("Identified α / k₀ (C assumed)", f"{float(p[0]) * C_ASSUMED / 3600:.3f} / {float(p[1]) * C_ASSUMED / 3600:.1f}")
st.caption(f"Thresholds from validation residuals: WARNING |r₁|>{thr['warn_1h']:.2f} or |r₆|>{thr['warn_6h']:.2f} °C; "
           f"ANOMALY |r₁|>{thr['anom_1h']:.2f} or |r₆|>{thr['anom_6h']:.2f} °C.")

# ------------------------------------------------------------------ what-if
st.subheader("What-if scenario (6 h ahead)")
s = st.columns(4)
f_S = s[0].slider("Solar radiation ×", 0.2, 1.6, 1.0, 0.05)
f_W = s[1].slider("Wind speed ×", 0.1, 2.0, 1.0, 0.05)
dTa = s[2].slider("Ambient temperature Δ (K)", -3.0, 3.0, 0.0, 0.1)
dQ = s[3].slider("Internal heat Δ (W/m²)", 0.0, 150.0, 0.0, 5.0)
ws = Windows(T0=wf.T0, Ta=wf.Ta + dTa, W=wf.W * f_W, S=np.clip(wf.S * f_S, 0, 1300), RH=wf.RH, hour=wf.hour,
             Y=wf.Y, start=wf.start, dq=np.array([dQ * 3600 / C_ASSUMED], np.float32))
pw, ow = pinn_predict(ws)[0], pm.predict(ws)[0]
r = st.columns(3)
r[0].metric("PINN: T in 6 h", f"{pw[-1]:.2f} °C", f"{pw[-1] - pred[-1]:+.2f} °C vs current")
r[1].metric("Physics ODE: T in 6 h", f"{ow[-1]:.2f} °C", f"{ow[-1] - ode[-1]:+.2f} °C vs current")
r[2].metric("Scenario physics residual", f"{residual(ws):.3f} K/h")

# ------------------------------------------------------------------ insights (equation terms, not correlations)
a, b0, b1, q = (float(x) for x in pm.ptuple())
S_m, W_m, Ta_m = float(ws.S.mean()), float(ws.W.mean()), float(ws.Ta.mean())
S_0, W_0, Ta_0 = float(wf.S.mean()), float(wf.W.mean()), float(wf.Ta.mean())
Tm = float(pw.mean())
terms_new = {"solar gain a·S": a * S_m, "loss −(b₀+b₁W)(T−Tₐ)": -(b0 + b1 * W_m) * (Tm - Ta_m), "internal q+Δq": q + dQ * 3600 / C_ASSUMED}
terms_old = {"solar gain a·S": a * S_0, "loss −(b₀+b₁W)(T−Tₐ)": -(b0 + b1 * W_0) * (float(pred.mean()) - Ta_0), "internal q+Δq": q}
st.subheader("Insights (energy-balance terms, K/h, 6-h mean)")
st.dataframe(pd.DataFrame({"current": terms_old, "scenario": terms_new}).round(3))
ins = []
if f_S != 1.0:
    ins.append(f"Physics insight: solar input changes the energy-gain term by {a * (S_m - S_0):+.3f} K/h, "
               f"so the predicted temperature {'rises' if S_m > S_0 else 'falls'}.")
if f_W != 1.0:
    ins.append(f"Scenario insight: wind ×{f_W:.2f} changes the convective coefficient b₀+b₁W from {b0 + b1 * W_0:.3f} to {b0 + b1 * W_m:.3f} 1/h; "
               f"{'less' if f_W < 1 else 'more'} heat is carried away while T > Tₐ.")
if dQ > 0:
    ins.append(f"Internal heat +{dQ:.0f} W/m² adds {dQ * 3600 / C_ASSUMED:.3f} K/h to dT/dt (for the assumed heat capacity).")
if status != "NORMAL":
    ins.append(f"Twin insight: the observed temperature is {r1:+.2f} °C away from what the physics-informed twin expects "
               f"for the current weather — incompatible with the expected physical state, not merely 'high'.")
for x in ins or ["No scenario change applied."]:
    st.info(x)

st.subheader("Suggested actions (decision support — not executed)")
dT6 = pw[-1] - pred[-1]
if dT6 > 1.0:
    st.warning(f"Scenario raises the 6-h temperature by {dT6:+.1f} °C. Evaluate increased ventilation or shading during the "
               "high-radiation period; the model attributes the rise to the terms above.")
elif status == "ANOMALY":
    st.error("Inspect the facility temperature sensor and local heat sources: the reading is inconsistent with the physical model.")
else:
    st.success("No action suggested for this scenario.")
st.caption("Model: PINN (PINNeAPPle ModifiedMLP, energy-balance residual, parameters identified jointly). "
           "Physics hypothesis: C dT/dt = αS − (k₀+k₁W)(T−Tₐ) + Q. Data are ERA5 reanalysis replayed as sensors.")
