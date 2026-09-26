# PINNeAPPle Benchmark

Public, reproducible benchmarks for Physics AI, built on
[PINNeAPPle](https://github.com/PINNeAPPle-Labs/PINNeAPPle).

Each case fixes the problem, the reference solution, the data splits, the
metrics and the compute budget, so results from different methods (PINN, FNO,
DeepONet, GNN, classical baselines) can be compared and reproduced with one
command. The execution harness lives in the PINNeAPPle library; this repository
holds the protocols, reference data (or pointers to it) and published results.

## Planned cases

| Case | Physics | Reference | Methods |
|---|---|---|---|
| 01 | Heat equation | FDM/FEM | PINN, FNO, DeepONet, MLP |
| 02 | Navier–Stokes | OpenFOAM | FNO, DeepONet, PINN, GNN |
| 03 | Geometry-conditioned surrogate | CFD/FEA | neural operators, GNN |

## Metrics reported for every case

L2 and relative error, training and inference time, memory, number of
simulations and training samples, extrapolation error, and physical
consistency (VeriPhysics).

Status: repository created; no case is published yet.

## License

Apache 2.0, same as PINNeAPPle.
