# Experiment 09: equal-budget results

Hardware: Intel(R) Xeon(R) Processor @ 2.10GHz (4 logical CPUs, 15.7 GB), no GPU; torch 2.14.1+cu130. **Every training run uses ONE CPU core** (torch.set_num_threads(1)); four runs execute concurrently, for every method alike. Budgets are small because there is no GPU.

Mean ± standard deviation over 3 seeds (sample standard deviation). Test errors marked *3 inst.* are on the three test instances every PINN was trained on (the operators on exactly the same instances); *all* is the operators' whole test set (200). Time per new instance: operator = one forward pass (batch 1); PINN = its training on that instance (it has to be retrained for every new input) plus evaluation; reference solver = one solve. Break-even: number of new queries above which the operator, including its simulation data and training, costs less than per-instance PINNs (all core-seconds; '0' = always cheaper).

## Burgers 1-D (budget 300 s on one core per training run)

| Method | Rel. L2, test (n = 3) | Rel. L2, test, all (n = 200) | Rel. L2, OOD (n = 2) | Integral QoI rel. err (n = 3) | Training (s) | Training FLOPs | Simulation data (core-s) | Time per new instance | Break-even (queries) |
|---|---|---|---|---|---|---|---|---|---|
| PINN | 0.246 ± 0.081 | - | 0.592 ± 0.051 | 0.118 ± 0.024 | 300 s | 2.19e+12 | - | 300 s | - |
| FNO | 0.00168 ± 0.00034 | 0.00328 ± 0.0003 | 0.0393 ± 0.0017 | 0.00193 ± 0.0011 | 300 s | 2.68e+12 | 1170 s | 1.7 ms | 4.9 |
| DeepONet | 0.0155 ± 0.002 | 0.0416 ± 0.0013 | 0.228 ± 0.033 | 0.0109 ± 0.0059 | 300 s | 1.08e+13 | 1170 s | 2.0 ms | 4.9 |
| Reference solver | reference | reference | reference | reference | - | - | - | 1 s | - |

## Darcy 2-D (budget 300 s on one core per training run)

| Method | Rel. L2, test (n = 3) | Rel. L2, test, all (n = 200) | Rel. L2, OOD (n = 2) | Integral QoI rel. err (n = 3) | Training (s) | Training FLOPs | Simulation data (core-s) | Time per new instance | Break-even (queries) |
|---|---|---|---|---|---|---|---|---|---|
| PINN | 0.873 ± 0.0041 | - | 0.891 ± 0.003 | 0.863 ± 0.004 | 300 s | 1.10e+13 | - | 300 s | - |
| FNO | 0.0167 ± 0.00049 | 0.0153 ± 0.00041 | 0.0138 ± 0.0017 | 0.00963 ± 0.00061 | 300 s | 2.11e+12 | 69 s | 12.6 ms | 1.2 |
| DeepONet | 0.057 ± 0.0032 | 0.0577 ± 0.00082 | 0.0602 ± 0.0032 | 0.0241 ± 0.0048 | 300 s | 1.58e+13 | 69 s | 5.0 ms | 1.2 |
| Reference solver | reference | reference | reference | reference | - | - | - | 63.1 ms | - |

## Navier-Stokes 2-D (budget 300 s on one core per training run)

| Method | Rel. L2, test (n = 3) | Rel. L2, test, all (n = 200) | Rel. L2, OOD (n = 2) | Integral QoI rel. err (n = 3) | Training (s) | Training FLOPs | Simulation data (core-s) | Time per new instance | Break-even (queries) |
|---|---|---|---|---|---|---|---|---|---|
| PINN | 0.16 ± 0.0056 | - | 0.507 ± 0.0053 | 0.248 ± 0.0087 | 300 s | 3.64e+12 | - | 300 s | - |
| FNO | 0.0068 ± 0.00089 | 0.00844 ± 0.0012 | 0.0107 ± 0.0012 | 0.00197 ± 0.00097 | 300 s | 2.21e+12 | 3843 s | 15.3 ms | 13.8 |
| DeepONet | 0.0307 ± 0.0051 | 0.0315 ± 0.00054 | 0.0419 ± 0.0085 | 0.0115 ± 0.0057 | 300 s | 1.58e+13 | 3843 s | 7.1 ms | 13.8 |
| Reference solver | reference | reference | reference | reference | - | - | - | 3 s | - |

## Airfoil 2-D (RANS) (budget 900 s on one core per training run)

OpenFOAM campaign: 436 runs (status {'converged': 330, 'steady-ish': 74, 'unsteady': 32}). One run alone on one core: 56 s (`openfoam_clean_timing.json`, 4 runs re-timed on the idle machine), so the data cost of the data-driven models is 436 x that = 6.8 core-hours (break-even against the potential-flow PINN: MeshGraphNet 28 queries, MLP 28 queries). The campaign's own recorded wall times sum to 10.9 h because it shared the CPU with other jobs. OOD set: 12 extra OpenFOAM runs (`gen_airfoil_ood.py`): NACA 0021, 2421, 4418, 6409, 6412 at 2° and 6° (CST weights outside the training box) and NACA 2412, 0012 at -4° (below the training angles).

### Integral quantities: Cl absolute error / Cd relative error

Mean ± std over seeds of the per-seed mean over the n runs of each set. A cell marked [n=...] has fewer runs than the column (the per-geometry PINNs: 1 NACA case, 4 OOD cases, seed 0 only for OOD). A scalar regressor (the MLP) is trained on exactly these scalars and is expected to beat field models on them.

| Method | held-out test, PINN cases (n = 4) | held-out test, all (n = 60) | NACA polars (n = 36) | OOD, PINN cases (n = 0) | OOD, all (n = 0) | Time per new design |
|---|---|---|---|---|---|---|
| PINN (potential flow) | 0.78 ± 0.0988 / 98.7% ± 0.349% | - | 0.536 ± 0.139 / 98.7% ± 1.38% [n=1] | - | - | 900 s |
| MeshGraphNet | 0.0334 ± 0.00847 / 1.69% ± 0.317% | 0.0319 ± 0.00325 / 2.21% ± 0.509% | 0.0321 ± 0.00219 / 3.32% ± 1.25% | - | - | 105.4 ms |
| MLP (Cl, Cd only) | 0.00107 ± 0.000309 / 0.295% ± 0.15% | 0.00227 ± 0.000377 / 0.567% ± 0.209% | 0.00238 ± 0.000219 / 0.457% ± 0.0129% | - | - | 219 µs |
| Panel method (inviscid, no training) | 0.144 / 100% | 0.124 / 100% | 0.144 / 100% | nan / nan% | nan / nan% | 6.4 ms |
| OpenFOAM (reference) | reference | reference | reference | reference | reference | 56 s |

### Fields on the near-field graph nodes: velocity rel. L2 / pressure rel. L2

The same 2,850 nodes for every method (the MeshGraphNet's graph, ~5 chords, boundary layer included). The MLP predicts only Cl and Cd. The potential-flow PINN and the panel method have no boundary layer, which bounds their velocity error from below.

| Method | held-out test, PINN cases (n = 4) | held-out test, all (n = 60) | NACA polars (n = 36) | OOD, PINN cases (n = 0) | OOD, all (n = 0) |
|---|---|---|---|---|---|
| PINN (potential flow) | 0.833 ± 0.0118 / 1.98 ± 0.00901 | - | 0.78 ± 0.0502 / 2.65 ± 0.0572 [n=1] | - | - |
| MeshGraphNet | 0.155 ± 0.0442 / 0.431 ± 0.135 | 0.156 ± 0.0438 / 0.458 ± 0.0985 | 0.164 ± 0.0438 / 0.478 ± 0.0999 | - | - |
| MLP (Cl, Cd only) | integrals only | integrals only | integrals only | integrals only | integrals only |
| Panel method (inviscid, no training) | 0.651 / 0.435 | 0.65 / 0.427 | 0.652 / 0.406 | nan / nan | nan / nan |

### PINN budget study (seed 0, one core)

| PINN | Case | Budget | Steps | Cl abs. err | Cd rel. err | Velocity rel. L2 | Pressure rel. L2 | Final loss |
|---|---|---|---|---|---|---|---|---|

### PINN setup audit (NACA 2412 at 4°, 900 s, seed 0)

| PINN | Variant | Steps | Cl abs. err | Cd rel. err | Velocity rel. L2 | Loss terms at the end |
|---|---|---|---|---|---|---|

### Did the potential-flow PINN solve its own equations?

Its Kutta-Joukowski Cl against the exact potential-flow (panel method) Cl of the same airfoil and angle, and OpenFOAM's (RANS) Cl.

| Runs | Budget | n | Mean abs. Cl(PINN) - Cl(panel) | Mean abs. Cl(panel) - Cl(OpenFOAM) |
|---|---|---|---|---|
| final | 300 s | 3 | 0.834 | 0.173 |
| search | 300 s | 4 | 0.948 | 0.0726 |
| final | 900 s | 15 | 0.858 | 0.127 |

Collocation (O-grid cell centres, NACA 2412): potential: 12312 points, 24 % within 0.01 c of the wall, 5 % within 0.05 c of the leading edge, nearest 9.1e-04 c; rans: 17100 points, 45 % within 0.01 c of the wall, 7 % within 0.05 c of the leading edge, nearest 2.7e-05 c.

## Inverse heat (h) (budget 120 s on one core per training run)

| Method | h rel. err, in range (n = 3) | h rel. err, out of range (n = 2) | Max field err °C, in range (n = 3) | Max field err °C, out of range (n = 2) | Hot-spot err °C, in range (n = 3) | Training (s) | Simulation data (core-s) | Time per new inverse problem | Break-even (queries) |
|---|---|---|---|---|---|---|---|---|---|
| PINN | 11.4 ± 5.3% | 39.3 ± 1.3% | 1.1 ± 0.31 | 1.71 ± 0.17 | 0.24 ± 0.13 | - | - | 120 s | - |
| MLP | 1.01 ± 0.63% | 4.48 ± 2.9% | 0.235 ± 0.096 | 0.613 ± 0.082 | 0.189 ± 0.09 | 120 s | 14 s | 110.5 ms | 1.1 |
| DeepONet | 1.01 ± 0.29% | 2.06 ± 1.2% | 0.524 ± 0.1 | 0.667 ± 0.22 | 0.129 ± 0.014 | 120 s | 14 s | 619.7 ms | 1.1 |

Failed runs: 0.
