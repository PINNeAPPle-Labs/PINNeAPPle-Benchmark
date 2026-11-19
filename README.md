# PINNeAPPle Benchmark

Public, reproducible benchmarks for Physics AI, built on
[PINNeAPPle](https://github.com/PINNeAPPle-Labs/PINNeAPPle).

Each case fixes the problem, the reference solution, the data splits, the
metrics and the compute budget, so results from different methods (PINN, FNO,
DeepONet, GNN, classical baselines) can be compared and reproduced with one
command. The execution harness lives in the PINNeAPPle library; this repository
holds the protocols, reference data (or pointers to it) and published results.

## Published experiments

Nine experiments are published under [`experiments/`](experiments), each with its own protocol, its source and a technical paper generated from its own results. [`experiments/README.md`](experiments/README.md) is the authoritative list.

| # | Experiment | Paper |
|---|---|---|
| 02 | Physics digital twin: a thermal mass with PINN parameter identification and anomaly detection | [`02_physics_digital_twin`](experiments/02_physics_digital_twin/paper/Physics_Digital_Twin_PINNeAPPle.pdf) |
| 03 | Bumper-beam crash surrogate: OpenRadioss DoE, a Transolver reproduction and an improvement | [`03_bumper_beam_transolver`](experiments/03_bumper_beam_transolver/paper/Bumper_Transolver_Reproduction_PINNeAPPle.pdf), [improvement](experiments/03_bumper_beam_transolver/paper/Bumper_Transolver_Improvement_PINNeAPPle.pdf) |
| 04 | Terramechanics surrogate robustness: Bekker–Wong constraint audit, hard constraints, 8-D parametric surrogate | [`04_terramechanics_robust`](experiments/04_terramechanics_robust/paper/Terramechanics_Robust_PINNeAPPle.pdf) |
| 05 | Heated 2-D channel digital twin: CFD reference, FNO/DeepONet/POD surrogates, EnKF, control | [`05_thermal_channel_twin`](experiments/05_thermal_channel_twin/paper/Heated_Channel_Digital_Twin_PINNeAPPle.pdf) |
| 06 | Millennium Prize problems: reproduction harness over seven problems | [`06_millennium`](experiments/06_millennium) |
| 07 | Pedestrian dead reckoning: reproduction and improvement of Ligthart (2025) | [`07_pdr_thesis_reproduction`](experiments/07_pdr_thesis_reproduction/paper/PDR_Reproduction_PINNeAPPle.pdf) |
| 08 | Supersonic shock train in a duct with a fixed-pressure outlet | [`08_supersonic_shock_train`](experiments/08_supersonic_shock_train/paper/Supersonic_ShockTrain_PINNeAPPle.pdf) |
| 09 | When PINNs, when data-driven surrogates: equal-budget PINN vs FNO vs DeepONet vs MeshGraphNet vs MLP | [`09_equal_budget_pinn_fno_deeponet`](experiments/09_equal_budget_pinn_fno_deeponet) (results and limitations, no PDF) |
| 10 | What the cheap model buys: a multi-fidelity learning curve on a capsule aerothermodynamics corpus | [`10_multifidelity_capsule`](experiments/10_multifidelity_capsule) |

Experiment `01_sst_satellite_forecasting` is climate work and lives permanently in
[`PINNeAPPle-Climate`](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Climate), which is why the numbering here starts at 02.

## Planned protocol cases

These are the protocol-layer cases this repository is named for: one command per case, fixed splits, standard metrics and a stated compute budget, so a result can be rerun and compared. **None of them is built yet** — that work is tracked in [#3](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark/issues/3), and until it exists the table above is what this repository actually publishes.

| Case | Physics | Reference | Methods |
|---|---|---|---|
| 01 | Heat equation | FDM/FEM | PINN, FNO, DeepONet, MLP |
| 02 | Navier–Stokes | OpenFOAM | FNO, DeepONet, PINN, GNN |
| 03 | Geometry-conditioned surrogate | CFD/FEA | neural operators, GNN |

The published experiments are deliberately kept beside the protocol layer rather than restructured to fit it: each was finished on its own terms, and rewriting them to a common interface that does not exist yet would be a claim about reproducibility this repository cannot make.

## Metrics reported for every case

L2 and relative error, training and inference time, memory, number of
simulations and training samples, extrapolation error, and physical
consistency (VeriPhysics).

## License

Apache 2.0, same as PINNeAPPle.
