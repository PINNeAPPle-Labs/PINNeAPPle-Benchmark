# Experiments

Research experiments built on PINNeAPPle, each with a technical paper (PDF) generated from its own
results. Moved here from `PINNeAPPle-Climate/experiments/` (2026-09-28) — none of them are climate work;
`01_sst_satellite_forecasting`, the one that is, stays in `PINNeAPPle-Climate`.

| # | Experiment | Paper(s) |
|---|---|---|
| 05 | Heated 2-D channel digital twin (CFD reference, FNO/DeepONet/POD surrogates, EnKF, control) | [Heated_Channel_Digital_Twin_PINNeAPPle.pdf](05_thermal_channel_twin/paper/Heated_Channel_Digital_Twin_PINNeAPPle.pdf) |
| 06 | Millennium Prize problems (Riemann, Yang–Mills, BSD, Hodge, Poincaré, Navier–Stokes, P vs NP) | see `06_millennium/*/Millennium_*.pdf` |
| 07 | Pedestrian dead reckoning (reproduction + improvement of Ligthart, 2025) | [PDR_Reproduction_PINNeAPPle.pdf](07_pdr_thesis_reproduction/paper/PDR_Reproduction_PINNeAPPle.pdf), [PDR_Improvement_PINNeAPPle.pdf](07_pdr_thesis_reproduction/paper/PDR_Improvement_PINNeAPPle.pdf) |
| 08 | Supersonic shock train in a duct with a fixed-pressure outlet | [Supersonic_ShockTrain_PINNeAPPle.pdf](08_supersonic_shock_train/paper/Supersonic_ShockTrain_PINNeAPPle.pdf) |

Still in `PINNeAPPle-Climate/experiments/` as of this move (running jobs; will move here once finished):
`02_physics_digital_twin`, `03_bumper_beam_transolver`, `04_terramechanics_robust`, and
`06_millennium/p_vs_np` (the rest of `06_millennium` moved; `p_vs_np` shares root-level files
`common_classifier.py`/`relearn.py` with it, so the whole directory waits for that one job).

## What is not committed

`data/` (raw/downloaded datasets), `results_smoke/` (smoke-test runs) and `*.pt` (model checkpoints) are
gitignored — each experiment's own script regenerates them, and a public git history is the wrong place
for multi-hundred-MB binaries. `results/*.json` and the small `.npz` files a paper's figures read from are
kept, so `paper/make_paper.py` reproduces the PDF from what is here.

## `paperkit/`

The shared PDF-builder used by every paper here. It is also vendored (kept in sync manually) in
`PINNeAPPle-Climate/experiments/paperkit/` for `01_sst_satellite_forecasting`, since that experiment stays
in the other repo — duplicated rather than cross-repo-imported, so this repo has no dependency on Climate.
