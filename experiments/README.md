# Experiments

Research experiments built on PINNeAPPle, each with a technical paper (PDF) generated from its own
results. Moved here from `PINNeAPPle-Climate/experiments/` (2026-09-28 and 2026-10-02) — none of them are
climate work; `01_sst_satellite_forecasting`, the one that is, stays in `PINNeAPPle-Climate`.

| # | Experiment | Paper(s) |
|---|---|---|
| 02 | Physics digital twin (Natal thermal mass: PINN + parameter identification, anomaly detection, what-if) | [Physics_Digital_Twin_PINNeAPPle.pdf](02_physics_digital_twin/paper/Physics_Digital_Twin_PINNeAPPle.pdf) |
| 03 | Bumper-beam crash surrogate (OpenRadioss DoE, Transolver reproduction + improvement, classical baselines) | [Bumper_Transolver_Reproduction_PINNeAPPle.pdf](03_bumper_beam_transolver/paper/Bumper_Transolver_Reproduction_PINNeAPPle.pdf), [Bumper_Transolver_Improvement_PINNeAPPle.pdf](03_bumper_beam_transolver/paper/Bumper_Transolver_Improvement_PINNeAPPle.pdf) |
| 04 | Terramechanics surrogate robustness (Bekker–Wong constraint audit, hard constraints, parametric 8-D surrogate) | [Terramechanics_Robust_PINNeAPPle.pdf](04_terramechanics_robust/paper/Terramechanics_Robust_PINNeAPPle.pdf) |
| 05 | Heated 2-D channel digital twin (CFD reference, FNO/DeepONet/POD surrogates, EnKF, control) | [Heated_Channel_Digital_Twin_PINNeAPPle.pdf](05_thermal_channel_twin/paper/Heated_Channel_Digital_Twin_PINNeAPPle.pdf) |
| 06 | Millennium Prize problems (Riemann, Yang–Mills, BSD, Hodge, Poincaré, Navier–Stokes, P vs NP) | see `06_millennium/*/Millennium_*.pdf` |
| 07 | Pedestrian dead reckoning (reproduction + improvement of Ligthart, 2025) | [PDR_Reproduction_PINNeAPPle.pdf](07_pdr_thesis_reproduction/paper/PDR_Reproduction_PINNeAPPle.pdf), [PDR_Improvement_PINNeAPPle.pdf](07_pdr_thesis_reproduction/paper/PDR_Improvement_PINNeAPPle.pdf), plus a plain-language [PDR_Stakeholder_Report_PINNeAPPle.pdf](07_pdr_thesis_reproduction/paper/PDR_Stakeholder_Report_PINNeAPPle.pdf) |
| 08 | Supersonic shock train in a duct with a fixed-pressure outlet | [Supersonic_ShockTrain_PINNeAPPle.pdf](08_supersonic_shock_train/paper/Supersonic_ShockTrain_PINNeAPPle.pdf) |

All eight experiments are now finished and moved here. Only `01_sst_satellite_forecasting` stays in
`PINNeAPPle-Climate` permanently (it is climate work).

## `03_bumper_beam_transolver` is special: raw data is not here

Every other experiment's heavy artifacts (`data/`, `results_smoke/`, `*.pt`) are just gitignored and
regenerable in seconds to minutes. `03`'s raw data — 100 OpenRadioss simulation outputs, the packaged
HDF5 dataset, and the OpenRadioss install itself — is **11 GB** and takes **about 2 days of wall-clock
time** to regenerate (the DoE campaign itself, run via Docker). That stays only on the machine that
produced it (`PINNeAPPle-Climate/experiments/03_bumper_beam_transolver/`, outside any git repo); what's
committed here is the code, the two papers, and the small results (`results.json`, the `.npz` examples the
papers' figures read) needed to rebuild the PDFs without re-running the campaign. The licensed example
deck (`model/`, Altair "Bumper Beam", CC BY-NC 4.0) is small enough to commit and is included as-is.

## What is not committed (every other experiment)

`data/` (raw/downloaded datasets), `results_smoke/` (smoke-test runs) and `*.pt` (model checkpoints) are
gitignored — each experiment's own script regenerates them, and a public git history is the wrong place
for multi-hundred-MB binaries. `results/*.json` and the small `.npz` files a paper's figures read from are
kept, so `paper/make_paper.py` reproduces the PDF from what is here.

## `paperkit/`

The shared PDF-builder used by every paper here. It is also vendored (kept in sync manually) in
`PINNeAPPle-Climate/experiments/paperkit/` for `01_sst_satellite_forecasting`, since that experiment stays
in the other repo — duplicated rather than cross-repo-imported, so this repo has no dependency on Climate.

## Two bugs found and fixed across these experiments, worth knowing if extending any of them

Both are the same root cause, found independently in two different experiments: PINNeAPPle's `ModifiedMLP`
uses Random Fourier Features with a default bandwidth (`sigma=1.0`) tuned for low-dimensional input. On
higher-dimensional inputs (8-D in `04`'s parametric surrogate, 6-D in `06/p_vs_np`'s learned-satisfiability
classifier) that default made training fail outright (far worse than a plain MLP with no RFF, or a trivial
baseline, on the same data) — not a subtle accuracy loss, a broken model. Both were fixed with
`sigma = 1.0 if in_dim <= 2 else 0.1`, found empirically and documented in each experiment's source. The
broken first attempt is kept in each paper's record (`*_first_attempt` keys in `results.json`), not hidden.
If you add a higher-dimensional input to a `ModifiedMLP` anywhere in this project, check this first.
