# SCIENTIFIC_CONTEXT — PINNeAPPle-Benchmark

The scientific content of every experiment currently in this repo (`05_thermal_channel_twin`,
`07_pdr_thesis_reproduction`, `08_supersonic_shock_train`) is documented in
`PINNeAPPle-Climate/.agent/SCIENTIFIC_CONTEXT.md` (sections "05", "07", "08") — it is not duplicated here
because the code moved unchanged; re-read that file for the governing physics, methods, and verification
approach of each. A short pointer per experiment, for orientation without switching files:

- **05 — heated 2-D channel digital twin**: incompressible Navier–Stokes + energy equation, actuated by
  inlet velocity `U_in(t)` and heater power `Q_h(t)`; a 4-layer study (CFD reference, ML surrogates
  including a physics-informed FNO, sparse-sensor EnKF assimilation, control through the surrogate
  re-verified in CFD).
- **07 — pedestrian dead reckoning**: reproduction (then an attempted improvement) of Ligthart's 2025 MSc
  thesis — an EKF corrected by a velocity-predicting neural network (adapted TLIO vs. DIVE architectures),
  plus grid-output PINN variants with an accelerometer-model physics loss. Public RIDI dataset, not the
  thesis' proprietary Leica data — absolute numbers are not comparable to the thesis, only the direction of
  each conclusion.
- **08 — supersonic shock train**: a new PINNeAPPle compressible finite-volume solver (HLLC/MUSCL,
  viscous, SSP-RK2) verified against Rankine–Hugoniot and Sod-shock-tube exact solutions, then used to
  reproduce a commercial-CFD (CONVERGE) visualisation of a pseudo-shock train in a duct with a
  fixed-pressure outlet, sweeping back pressure and doing a 3-grid Richardson/GCI study.

When more experiments move in from `PINNeAPPle-Climate` (02, 03, 04, the rest of `06_millennium`), extend
this file or keep pointing at the Climate repo's copy — whichever is decided when that move happens (not
decided as of this writing).
