# Benchmark results

Hardware: Intel(R) Xeon(R) Processor @ 2.10GHz (4 logical CPUs), GPU: None; torch 2.14.1+cu130. Every training run uses 1 thread(s); 4 run(s) at a time. Seeds: 3; search: 4 trials per method at 0.333 x the budget, validation data only. Cells: mean ± std over seeds of the per-seed mean over the n instances of the column. A row whose error is no better than a trivial prediction is labelled 'did not converge'; one that failed a gate is INVALID.

## Burgers 1-D (budget per run: 300 s wall clock)

### Fields

| Method | Rel. L2, test (n = 3) | Rel. L2, test, all (n = 200) | Rel. L2, OOD (n = 2) | Rel. L2, OOD, all (n = 100) | Checks |
|---|---|---|---|---|---|
| PINN | 0.246 ± 0.081 | - | 0.592 ± 0.051 | - | ✓ pass (warn) |
| FNO | 0.00168 ± 0.00034 | 0.00328 ± 0.0003 | 0.0393 ± 0.0017 | 0.0199 ± 0.0015 | ✓ pass (warn) |
| DeepONet | 0.0155 ± 0.002 | 0.0416 ± 0.0013 | 0.228 ± 0.033 | 0.142 ± 0.0018 | ✓ pass (warn) |

### Integral quantities

| Method | Energy ∫u² rel. err (n = 3) | Checks |
|---|---|---|
| PINN | 0.118 ± 0.024 | ✓ pass (warn) |
| FNO | 0.00193 ± 0.0011 | ✓ pass (warn) |
| DeepONet | 0.0109 ± 0.0059 | ✓ pass (warn) |

### Cost

| Method | Training (s) | Training FLOPs | Simulation data (core-s) | Time per new instance | Break-even vs per-instance (queries) |
|---|---|---|---|---|---|
| PINN | 300 s | 2.24e+12 | - | 300 s | - |
| FNO | 300 s | 2.68e+12 | 1170 s | 1.7 ms | 4.9 |
| DeepONet | 300 s | 1.08e+13 | 1170 s | 2.0 ms | 4.9 |
| Reference solver | - | - | - | 1 s | - |

### Equal FLOPs (2.16e+12 training FLOPs per run: the fewest any method used in its full wall-clock budget; wall clock capped at 3x the budget)

| Method | Rel. L2, test instance(s) 0 (n = 1) | Rel. L2, test, all (n = 200) | FLOPs used | Wall time | Stopped by |
|---|---|---|---|---|---|
| PINN | 0.252 ± 0.086 | - | 2.16e+12 | 463 s | flops |
| FNO | 0.00186 ± 0.00013 | 0.00336 ± 0.00034 | 2.16e+12 | 281 s | flops |
| DeepONet | 0.058 ± 0.0061 | 0.0561 ± 0.0036 | 2.16e+12 | 60 s | flops |

## Darcy 2-D (budget per run: 300 s wall clock)

### Fields

| Method | Rel. L2, test (n = 3) | Rel. L2, test, all (n = 200) | Rel. L2, OOD (n = 2) | Rel. L2, OOD, all (n = 100) | Checks |
|---|---|---|---|---|---|
| PINN | 0.873 ± 0.0041 | - | 0.891 ± 0.003 | - | ✓ pass (warn) |
| FNO | 0.0167 ± 0.00049 | 0.0153 ± 0.00041 | 0.0138 ± 0.0017 | 0.0172 ± 0.00053 | ✓ pass |
| DeepONet | 0.057 ± 0.0032 | 0.0577 ± 0.00082 | 0.0602 ± 0.0032 | 0.0611 ± 0.00051 | ✓ pass (warn) |

### Integral quantities

| Method | Compliance ∫u rel. err (n = 3) | Checks |
|---|---|---|
| PINN | 0.863 ± 0.004 | ✓ pass (warn) |
| FNO | 0.00963 ± 0.00061 | ✓ pass |
| DeepONet | 0.0241 ± 0.0048 | ✓ pass (warn) |

### Cost

| Method | Training (s) | Training FLOPs | Simulation data (core-s) | Time per new instance | Break-even vs per-instance (queries) |
|---|---|---|---|---|---|
| PINN | 300 s | 1.10e+13 | - | 300 s | - |
| FNO | 300 s | 2.11e+12 | 69 s | 12.6 ms | 1.2 |
| DeepONet | 300 s | 1.58e+13 | 69 s | 5.0 ms | 1.2 |
| Reference solver | - | - | - | 63.1 ms | - |

### Equal FLOPs (2.1e+12 training FLOPs per run: the fewest any method used in its full wall-clock budget; wall clock capped at 3x the budget)

| Method | Rel. L2, test instance(s) 0 (n = 1) | Rel. L2, test, all (n = 200) | FLOPs used | Wall time | Stopped by |
|---|---|---|---|---|---|
| PINN | 0.846 ± 0.013 | - | 2.1e+12 | 74 s | flops |
| FNO | 0.0139 ± 0.00052 | 0.0153 ± 0.00037 | 2.1e+12 | 400 s | flops |
| DeepONet | 0.0521 ± 0.0017 | 0.066 ± 0.00093 | 2.1e+12 | 44 s | flops |

## Navier-Stokes 2-D (budget per run: 300 s wall clock)

### Fields

| Method | Rel. L2, test (n = 3) | Rel. L2, test, all (n = 200) | Rel. L2, OOD (n = 2) | Rel. L2, OOD, all (n = 100) | Checks |
|---|---|---|---|---|---|
| PINN | 0.16 ± 0.0056 | - | 0.507 ± 0.0053 | - | ✓ pass (warn) |
| FNO | 0.0068 ± 0.00089 | 0.00844 ± 0.0012 | 0.0107 ± 0.0012 | 0.0137 ± 0.0013 | ✓ pass (warn) |
| DeepONet | 0.0307 ± 0.0051 | 0.0315 ± 0.00054 | 0.0419 ± 0.0085 | 0.0417 ± 0.0019 | ✓ pass (warn) |

### Integral quantities

| Method | Enstrophy ∫ω² rel. err (n = 3) | Checks |
|---|---|---|
| PINN | 0.248 ± 0.0087 | ✓ pass (warn) |
| FNO | 0.00197 ± 0.00097 | ✓ pass (warn) |
| DeepONet | 0.0115 ± 0.0057 | ✓ pass (warn) |

### Cost

| Method | Training (s) | Training FLOPs | Simulation data (core-s) | Time per new instance | Break-even vs per-instance (queries) |
|---|---|---|---|---|---|
| PINN | 300 s | 3.64e+12 | - | 300 s | - |
| FNO | 300 s | 2.21e+12 | 3843 s | 15.3 ms | 13.8 |
| DeepONet | 300 s | 1.58e+13 | 3843 s | 7.1 ms | 13.8 |
| Reference solver | - | - | - | 3 s | - |

### Equal FLOPs (2.2e+12 training FLOPs per run: the fewest any method used in its full wall-clock budget; wall clock capped at 3x the budget)

| Method | Rel. L2, test instance(s) 0 (n = 1) | Rel. L2, test, all (n = 200) | FLOPs used | Wall time | Stopped by |
|---|---|---|---|---|---|
| PINN | 0.304 ± 0.017 | - | 2.21e+12 | 201 s | flops |
| FNO | 0.0111 ± 0.0021 | 0.00845 ± 0.0012 | 2.2e+12 | 395 s | flops |
| DeepONet | 0.0524 ± 0.0019 | 0.041 ± 0.0007 | 2.2e+12 | 51 s | flops |

## Airfoil 2-D (RANS, Re 4e6) (budget per run: 900 s wall clock)

### Fields

| Method | Velocity rel. L2, held-out test, PINN cases (n = 4) | Pressure rel. L2, held-out test, PINN cases (n = 4) | Velocity rel. L2, held-out test, all (n = 60) | Pressure rel. L2, held-out test, all (n = 60) | Velocity rel. L2, NACA polars (n = 36) | Pressure rel. L2, NACA polars (n = 36) | Velocity rel. L2, OOD, PINN cases (n = 4) | Pressure rel. L2, OOD, PINN cases (n = 4) | Velocity rel. L2, OOD, all (n = 12) | Pressure rel. L2, OOD, all (n = 12) | Checks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PINN (potential flow) | 0.833 ± 0.012 | 1.98 ± 0.009 | - | - | 0.78 ± 0.05 [n=1] | 2.65 ± 0.057 [n=1] | 0.856 | 2.84 | - | - | ✓ pass |
| MeshGraphNet | 0.169 ± 0.049 | 0.479 ± 0.15 | 0.169 ± 0.047 | 0.502 ± 0.11 | 0.177 ± 0.047 | 0.523 ± 0.11 | 0.203 ± 0.024 | 0.746 ± 0.045 | 0.211 ± 0.023 | 0.724 ± 0.049 | ✓ pass (warn) |
| MLP (Cl, Cd only) | integrals only | integrals only | integrals only | integrals only | integrals only | integrals only | integrals only | integrals only | integrals only | integrals only | ✓ pass (warn) |
| POD + MLP | 0.00383 ± 0.0003 | 0.00929 ± 0.0015 | 0.00496 ± 0.00033 | 0.0142 ± 0.0017 | 0.00466 ± 0.00021 | 0.012 ± 0.0016 | 0.0164 ± 0.0035 | 0.0586 ± 0.016 | 0.0179 ± 0.0035 | 0.0581 ± 0.019 | ✓ pass |
| PINN (RANS) | 0.96 ± 0.0043 | 1.02 ± 0.017 | - | - | 0.959 ± 0.022 [n=1] | 1.03 ± 0.022 [n=1] | 0.98 | 1.03 | - | - | ✗ INVALID |
| Panel method (inviscid, no data; reference) | 0.651 | 0.435 | 0.65 | 0.427 | 0.652 | 0.406 | 0.612 | 0.345 | 0.624 | 0.358 | reference |

### Integral quantities

| Method | Cl abs. err, held-out test, PINN cases (n = 4) | Cd rel. err, held-out test, PINN cases (n = 4) | Cl abs. err, held-out test, all (n = 60) | Cd rel. err, held-out test, all (n = 60) | Cl abs. err, NACA polars (n = 36) | Cd rel. err, NACA polars (n = 36) | Cl abs. err, OOD, PINN cases (n = 4) | Cd rel. err, OOD, PINN cases (n = 4) | Cl abs. err, OOD, all (n = 12) | Cd rel. err, OOD, all (n = 12) | Checks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PINN (potential flow) | 0.78 ± 0.099 | 98.7% ± 0.35% | - | - | 0.536 ± 0.14 [n=1] | 98.7% ± 1.4% [n=1] | 0.376 | 99% | - | - | ✓ pass |
| MeshGraphNet | 0.0342 ± 0.016 | 1.87% ± 0.55% | 0.0333 ± 0.0039 | 2.16% ± 0.31% | 0.0339 ± 0.0032 | 2.95% ± 0.82% | 0.0867 ± 0.028 | 4.41% ± 0.68% | 0.0839 ± 0.032 | 6.1% ± 0.85% | ✓ pass (warn) |
| MLP (Cl, Cd only) | 0.00107 ± 0.00031 | 0.295% ± 0.15% | 0.00227 ± 0.00038 | 0.566% ± 0.21% | 0.00238 ± 0.00022 | 0.456% ± 0.013% | 0.00453 ± 0.0012 | 0.535% ± 0.17% | 0.00462 ± 0.00095 | 0.739% ± 0.25% | ✓ pass (warn) |
| POD + MLP | fields only | fields only | fields only | fields only | fields only | fields only | fields only | fields only | fields only | fields only | ✓ pass |
| PINN (RANS) | 0.875 ± 0.0013 | 89.8% ± 4.2% | - | - | 0.662 ± 0.00052 [n=1] | 93% ± 6.3% [n=1] | 0.48 | 87.7% | - | - | ✗ INVALID |
| Panel method (inviscid, no data; reference) | 0.144 | 100% | 0.124 | 100% | 0.146 | 100% | 0.0806 | 100% | 0.126 | 100% | reference |

### Cost

| Method | Training (s) | Training FLOPs | Simulation data (core-s) | Time per new instance | Break-even vs per-instance (queries) |
|---|---|---|---|---|---|
| PINN (potential flow) | 900 s | 1.65e+13 | - | 900 s | - |
| MeshGraphNet | 900 s | 1.86e+13 | 6.8 h | 104.9 ms | 28.2 |
| MLP (Cl, Cd only) | 900 s | 1.72e+13 | 6.8 h | 231 µs | 28.2 |
| POD + MLP | 900 s | 1.23e+13 | 6.8 h | 55 µs | 28.2 |
| PINN (RANS) | 901 s | 1.62e+13 | - | 901 s | - |
| Reference solver | - | - | - | 56 s | - |

_Reading the airfoil tables: the MLP predicts the integrals only (Cl, Cd, Cm from the shape parameters and the angle), so its field columns read "integrals only". A scalar regressor trained directly on the scalars is expected to beat a field model (MeshGraphNet, POD + MLP, the PINNs) on them; compare field models with each other on the field columns, and every method on the integral columns knowing this._

### Airfoil 2-D (RANS, Re 4e6): the potential-flow PINN against the panel method (its own equations)

Over 35 runs, |Cl(PINN) - Cl(panel)| = 0.784 ± 0.31 (both inviscid; OpenFOAM's Cl is the viscous truth).

| Run | Case | Budget | Cl PINN | Cl panel | Cl OpenFOAM |
|---|---|---|---|---|---|
| `diag__airfoil__pinn__lbfgs__NACA2412_a+04` | NACA2412_a+04 | 900 s | 0.236 | 0.723 | 0.662 |
| `diag__airfoil__pinn__lr1e-3__NACA2412_a+04` | NACA2412_a+04 | 900 s | 0.419 | 0.723 | 0.662 |
| `diag__airfoil__pinn__weights_wall100_kutta100__NACA2412_a+04` | NACA2412_a+04 | 900 s | 0.038 | 0.723 | 0.662 |
| `diag__airfoil__pinn__weights_wall1_kutta1__NACA2412_a+04` | NACA2412_a+04 | 900 s | 0.236 | 0.723 | 0.662 |
| `final__airfoil__pinn__L0.111__s0__s003_a1` | s003_a1 | 100 s | 0.197 | 1.002 | 0.829 |
| `final__airfoil__pinn__L0.111__s1__s003_a1` | s003_a1 | 100 s | 0.131 | 1.002 | 0.829 |
| `final__airfoil__pinn__L0.111__s2__s003_a1` | s003_a1 | 100 s | 0.150 | 1.002 | 0.829 |
| `final__airfoil__pinn__L0.333__s0__s003_a1` | s003_a1 | 300 s | 0.223 | 1.002 | 0.829 |
| `final__airfoil__pinn__L0.333__s1__s003_a1` | s003_a1 | 300 s | 0.107 | 1.002 | 0.829 |
| `final__airfoil__pinn__L0.333__s2__s003_a1` | s003_a1 | 300 s | 0.174 | 1.002 | 0.829 |
| `final__airfoil__pinn__L1.000__s0__NACA0021_a+02` | NACA0021_a+02 | 900 s | 0.216 | 0.255 | 0.205 |
| `final__airfoil__pinn__L1.000__s0__NACA2412_a+04` | NACA2412_a+04 | 900 s | 0.220 | 0.723 | 0.662 |
| `final__airfoil__pinn__L1.000__s0__NACA2412_a-04` | NACA2412_a-04 | 900 s | 0.070 | -0.233 | -0.225 |
| `final__airfoil__pinn__L1.000__s0__NACA4418_a+02` | NACA4418_a+02 | 900 s | 0.217 | 0.782 | 0.624 |
| `final__airfoil__pinn__L1.000__s0__NACA6412_a+02` | NACA6412_a+02 | 900 s | 0.075 | 0.972 | 0.865 |
| `final__airfoil__pinn__L1.000__s0__s003_a1` | s003_a1 | 900 s | 0.265 | 1.002 | 0.829 |
| `final__airfoil__pinn__L1.000__s0__s003_a2` | s003_a2 | 900 s | 0.155 | 1.515 | 1.230 |
| `final__airfoil__pinn__L1.000__s0__s007_a1` | s007_a1 | 900 s | 0.180 | 0.675 | 0.627 |
| `final__airfoil__pinn__L1.000__s0__s007_a2` | s007_a2 | 900 s | 0.120 | 0.888 | 0.818 |
| `final__airfoil__pinn__L1.000__s1__NACA2412_a+04` | NACA2412_a+04 | 900 s | -0.033 | 0.723 | 0.662 |
| `final__airfoil__pinn__L1.000__s1__s003_a1` | s003_a1 | 900 s | 0.065 | 1.002 | 0.829 |
| `final__airfoil__pinn__L1.000__s1__s003_a2` | s003_a2 | 900 s | 0.037 | 1.515 | 1.230 |
| `final__airfoil__pinn__L1.000__s1__s007_a1` | s007_a1 | 900 s | -0.114 | 0.675 | 0.627 |
| `final__airfoil__pinn__L1.000__s1__s007_a2` | s007_a2 | 900 s | -0.040 | 0.888 | 0.818 |
| `final__airfoil__pinn__L1.000__s2__NACA2412_a+04` | NACA2412_a+04 | 900 s | 0.194 | 0.723 | 0.662 |
| `final__airfoil__pinn__L1.000__s2__s003_a1` | s003_a1 | 900 s | 0.164 | 1.002 | 0.829 |
| `final__airfoil__pinn__L1.000__s2__s003_a2` | s003_a2 | 900 s | 0.142 | 1.515 | 1.230 |
| `final__airfoil__pinn__L1.000__s2__s007_a1` | s007_a1 | 900 s | 0.101 | 0.675 | 0.627 |
| `final__airfoil__pinn__L1.000__s2__s007_a2` | s007_a2 | 900 s | 0.075 | 0.888 | 0.818 |
| `search__airfoil__pinn__0` | s001_a1 | 300 s | 0.095 | 1.055 | 0.983 |
| `search__airfoil__pinn__1` | s001_a1 | 300 s | 0.134 | 1.055 | 0.983 |
| `search__airfoil__pinn__2` | s001_a1 | 300 s | 0.156 | 1.055 | 0.983 |
| `search__airfoil__pinn__3` | s001_a1 | 300 s | 0.046 | 1.055 | 0.983 |
| `study__airfoil__pinn__B3600__NACA2412_a+04` | NACA2412_a+04 | 3600 s | 0.043 | 0.723 | 0.662 |
| `study__airfoil__pinn__B3600__s003_a1` | s003_a1 | 3600 s | -0.295 | 1.002 | 0.829 |

### Airfoil PINN collocation (NACA 2412)

| PINN | Points | Within 0.01 c of the wall | Within 0.05 c of the leading edge | Closest to the wall |
|---|---|---|---|---|
| potential | 12312 | 24.1 % | 4.6 % | 9.1e-04 c |
| RANS | 17100 | 45.3 % | 6.7 % | 2.7e-05 c |

## Inverse heat (h from 8 sensors) (budget per run: 120 s wall clock)

### Fields

| Method | Max field err °C, in range (n = 3) | Max field err °C, out of range (n = 2) | Checks |
|---|---|---|---|
| PINN | 1.1 ± 0.31 | 1.71 ± 0.17 | ✓ pass (warn) |
| DeepONet surrogate + calibration | 0.524 ± 0.1 | 0.667 ± 0.22 | ✓ pass (warn) |
| MLP surrogate + calibration | 0.235 ± 0.096 | 0.613 ± 0.082 | ✓ pass (warn) |
| FV model + calibration (reference) | 0.168 ± 0.098 | 0.238 ± 0.15 | reference |

### Integral quantities

| Method | h rel. err, in range (n = 3) | h rel. err, out of range (n = 2) | Hot-spot err °C, in range (n = 3) | Checks |
|---|---|---|---|---|
| PINN | 11.4% ± 5.3% | 39.3% ± 1.3% | 0.24 ± 0.13 | ✓ pass (warn) |
| DeepONet surrogate + calibration | 1.01% ± 0.29% | 2.06% ± 1.2% | 0.129 ± 0.014 | ✓ pass (warn) |
| MLP surrogate + calibration | 1.01% ± 0.63% | 4.48% ± 2.9% | 0.189 ± 0.09 | ✓ pass (warn) |
| FV model + calibration (reference) | 0.946% ± 0.55% | 3.31% ± 2% | 0.166 ± 0.097 | reference |

### Cost

| Method | Training (s) | Training FLOPs | Simulation data (core-s) | Time per new instance | Break-even vs per-instance (queries) |
|---|---|---|---|---|---|
| PINN | 120 s | 1.18e+12 | - | 120 s | - |
| DeepONet surrogate + calibration | 120 s | 4.47e+12 | 14 s | 619.7 ms | 1.1 |
| MLP surrogate + calibration | 120 s | 1.27e+12 | 14 s | 110.5 ms | 1.1 |
| Reference solver | - | - | - | 9 s | - |

## Budget study and setup audit

### Budget study: PINN (RANS) on Airfoil 2-D (RANS, Re 4e6) (seed 0)

Instance `s003_a1`:

| Budget | test4/cl | test4/cd | test4/u | test4/p | Run |
|---|---|---|---|---|---|
| 100 s | 0.828 | 1 | 0.959 | 0.992 | `final__airfoil__pinn_rans__L0.111__s0__s003_a1` |
| 300 s | 0.827 | 1 | 0.973 | 0.992 | `final__airfoil__pinn_rans__L0.333__s0__s003_a1` |
| 900 s | 0.828 | 0.728 | 0.97 | 0.977 | `final__airfoil__pinn_rans__L1.000__s0__s003_a1` |
| 3600 s | 0.829 | 1 | 0.994 | 1 | `study__airfoil__pinn_rans__B3600__s003_a1` |
| 4.0 h | 0.829 | 1 | 0.957 | 1.36 | `study__airfoil__pinn_rans__B14400__s003_a1` |

Instance `NACA2412_a+04`:

| Budget | naca/cl | naca/cd | naca/u | naca/p | Run |
|---|---|---|---|---|---|
| 900 s | 0.662 | 0.879 | 0.974 | 1.02 | `final__airfoil__pinn_rans__L1.000__s0__NACA2412_a+04` |
| 3600 s | 0.662 | 1 | 1.07 | 1 | `study__airfoil__pinn_rans__B3600__NACA2412_a+04` |
| 4.0 h | 0.663 | 0.983 | 0.973 | 1.07 | `study__airfoil__pinn_rans__B14400__NACA2412_a+04` |

### Budget study: PINN (potential flow) on Airfoil 2-D (RANS, Re 4e6) (seed 0)

Instance `s003_a1`:

| Budget | test4/cl | test4/cd | test4/u | test4/p | Run |
|---|---|---|---|---|---|
| 100 s | 0.632 | 0.904 | 0.592 | 1.87 | `final__airfoil__pinn__L0.111__s0__s003_a1` |
| 300 s | 0.605 | 0.971 | 0.728 | 2.11 | `final__airfoil__pinn__L0.333__s0__s003_a1` |
| 900 s | 0.564 | 1 | 0.891 | 2.23 | `final__airfoil__pinn__L1.000__s0__s003_a1` |
| 3600 s | 1.12 | 1 | 0.904 | 2.23 | `study__airfoil__pinn__B3600__s003_a1` |

Instance `NACA2412_a+04`:

| Budget | naca/cl | naca/cd | naca/u | naca/p | Run |
|---|---|---|---|---|---|
| 900 s | 0.442 | 0.995 | 0.809 | 2.68 | `final__airfoil__pinn__L1.000__s0__NACA2412_a+04` |
| 3600 s | 0.619 | 0.995 | 0.849 | 2.71 | `study__airfoil__pinn__B3600__NACA2412_a+04` |

### Setup audit: PINN (potential flow) on Airfoil 2-D (RANS, Re 4e6), instance `NACA2412_a+04` (seed 0, full budget)

| Variant | Change | naca/cl | naca/cd | naca/u | naca/p |
|---|---|---|---|---|---|
| best configuration (baseline) | - | 0.442 | 0.995 | 0.809 | 2.68 |
| lr1e-3 | lr = 0.001 | 0.244 | 0.964 | 0.696 | 2.54 |
| lbfgs | lbfgs = 0.3 | 0.426 | 0.984 | 0.743 | 2.61 |
| weights_wall100_kutta100 | wall_w = 100.0, kutta_w = 100.0 | 0.624 | 0.995 | 0.93 | 2.74 |
| weights_wall1_kutta1 | wall_w = 1.0, kutta_w = 1.0 | 0.427 | 0.891 | 0.514 | 1.76 |

### Setup audit: PINN (RANS) on Airfoil 2-D (RANS, Re 4e6), instance `NACA2412_a+04` (seed 0, full budget)

| Variant | Change | naca/cl | naca/cd | naca/u | naca/p |
|---|---|---|---|---|---|
| best configuration (baseline) | - | 0.662 | 0.879 | 0.974 | 1.02 |
| lbfgs | lbfgs = 0.3 | 0.662 | 0.971 | 0.944 | 1.02 |
| lr1e-3 | lr = 0.001 | 0.661 | 1 | 0.955 | 0.996 |

## Diagnostics and suggestions

- **error** [airfoil/pinn_rans] Did not converge: the error is no better than a trivial prediction. Evidence: mean u on test4 = 0.96 (>= 0.9) Suggestion: Check the nondimensionalisation (inputs and outputs O(1)), the loss weights and the collocation density near walls and leading edges; consider more budget or a formulation with fewer stiff terms.
- **warning** [burgers/fno] Still improving at the full budget: the comparison is budget-limited for this method. Evidence: rel_l2 on pinn_test: 0.00261 at 0.333x -> 0.00168 at 1x (36 % lower for 3.0x the budget) Suggestion: Train longer (the next 3.0x) or report it as budget-limited. Edit: `budget.wall_s: 300.0 → 900`.
- **warning** [burgers/deeponet] Still improving at the full budget: the comparison is budget-limited for this method. Evidence: rel_l2 on pinn_test: 0.0322 at 0.333x -> 0.0155 at 1x (52 % lower for 3.0x the budget) Suggestion: Train longer (the next 3.0x) or report it as budget-limited. Edit: `budget.wall_s: 300.0 → 900`.
- **warning** [darcy/fno] Still improving at the full budget: the comparison is budget-limited for this method. Evidence: rel_l2 on pinn_test: 0.034 at 0.333x -> 0.0167 at 1x (51 % lower for 3.0x the budget) Suggestion: Train longer (the next 3.0x) or report it as budget-limited. Edit: `budget.wall_s: 300.0 → 900`.
- **warning** [navier_stokes/pinn] Still improving at the full budget: the comparison is budget-limited for this method. Evidence: rel_l2 on pinn_test: 0.746 at 0.333x -> 0.16 at 1x (78 % lower for 3.0x the budget) Suggestion: Train longer (the next 3.0x) or report it as budget-limited. Edit: `budget.wall_s: 300.0 → 900`.
- **warning** [navier_stokes/fno] Still improving at the full budget: the comparison is budget-limited for this method. Evidence: rel_l2 on pinn_test: 0.111 at 0.333x -> 0.0068 at 1x (94 % lower for 3.0x the budget) Suggestion: Train longer (the next 3.0x) or report it as budget-limited. Edit: `budget.wall_s: 300.0 → 900`.
- **warning** [airfoil/mgn] Still improving at the full budget: the comparison is budget-limited for this method. Evidence: u on test4: 0.368 at 0.333x -> 0.169 at 1x (54 % lower for 3.0x the budget) Suggestion: Train longer (the next 3.0x) or report it as budget-limited. Edit: `problem_budgets.airfoil.wall_s: 900.0 → 2700`.
- **warning** [inverse_heat/pinn] Still improving at the full budget: the comparison is budget-limited for this method. Evidence: h on id: 0.233 at 0.333x -> 0.114 at 1x (51 % lower for 3.0x the budget) Suggestion: Train longer (the next 3.0x) or report it as budget-limited. Edit: `problem_budgets.inverse_heat.wall_s: 120.0 → 360`.
- **warning** [darcy/pinn] The loss fell by less than half in full-budget runs: the optimisation is slow or stalled. Evidence: 16/18 jobs; final__darcy__pinn__L1.000__s2__test2 (1.46 -> 0.806); flops__darcy__pinn__s2 (1.53 -> 0.79) Suggestion: Check the learning rate and the loss scaling.
- **warning** [navier_stokes/pinn] The loss fell by less than half in full-budget runs: the optimisation is slow or stalled. Evidence: 15/18 jobs; final__navier_stokes__pinn__L1.000__s0__ood1 (0.159 -> 0.0798); final__navier_stokes__pinn__L1.000__s1__ood0 (0.159 -> 0.0819); final__navier_stokes__pinn__L1.0 Suggestion: Check the learning rate and the loss scaling.
- **warning** [airfoil/mgn] Loss spikes in the second half of training: the learning rate is likely too high. Evidence: 9 logged losses above 3x the running minimum (final__airfoil__mgn__L1.000__s0) Suggestion: Halve the learning rate (add it to the method's grid). Edit: `method_options.mgn.lr: 0.001 → [0.00025, 0.0005]`.
- **warning** [airfoil/pinn_rans] Loss spikes in the second half of training: the learning rate is likely too high. Evidence: 5 logged losses above 3x the running minimum (final__airfoil__pinn_rans__L1.000__s0__NACA0021_a+02) Suggestion: Halve the learning rate (add it to the method's grid). Edit: `method_options.pinn_rans.lr: 0.003 → [0.00075, 0.0015]`.
- **warning** [airfoil/pinn_rans] Loss spikes in the second half of training: the learning rate is likely too high. Evidence: 12 logged losses above 3x the running minimum (final__airfoil__pinn_rans__L1.000__s0__NACA2412_a+04) Suggestion: Halve the learning rate (add it to the method's grid). Edit: `method_options.pinn_rans.lr: 0.003 → [0.00075, 0.0015]`.
- **warning** [airfoil/pinn_rans] No progress in the first 10 % of the budget: the learning rate may be too low. Evidence: loss 135 -> 325 over the first 10 % (final__airfoil__pinn_rans__L1.000__s0__NACA2412_a+04) Suggestion: Try a 3x larger learning rate. Edit: `method_options.pinn_rans.lr: 0.003 → [0.003, 0.009000000000000001]`.
- **warning** [airfoil/pinn_rans] Loss spikes in the second half of training: the learning rate is likely too high. Evidence: 77 logged losses above 3x the running minimum (final__airfoil__pinn_rans__L1.000__s0__NACA2412_a-04) Suggestion: Halve the learning rate (add it to the method's grid). Edit: `method_options.pinn_rans.lr: 0.003 → [0.00075, 0.0015]`.
- **warning** [airfoil/pinn_rans] No progress in the first 10 % of the budget: the learning rate may be too low. Evidence: loss 114 -> 730 over the first 10 % (final__airfoil__pinn_rans__L1.000__s0__NACA2412_a-04) Suggestion: Try a 3x larger learning rate. Edit: `method_options.pinn_rans.lr: 0.003 → [0.003, 0.009000000000000001]`.
- **warning** [airfoil/pinn_rans] Loss spikes in the second half of training: the learning rate is likely too high. Evidence: 12 logged losses above 3x the running minimum (final__airfoil__pinn_rans__L1.000__s0__s003_a1) Suggestion: Halve the learning rate (add it to the method's grid). Edit: `method_options.pinn_rans.lr: 0.003 → [0.00075, 0.0015]`.
- **warning** [airfoil/pinn_rans] No progress in the first 10 % of the budget: the learning rate may be too low. Evidence: loss 181 -> 268 over the first 10 % (final__airfoil__pinn_rans__L1.000__s0__s003_a1) Suggestion: Try a 3x larger learning rate. Edit: `method_options.pinn_rans.lr: 0.003 → [0.003, 0.009000000000000001]`.
- **warning** [airfoil/pinn_rans] Loss spikes in the second half of training: the learning rate is likely too high. Evidence: 78 logged losses above 3x the running minimum (final__airfoil__pinn_rans__L1.000__s0__s003_a2) Suggestion: Halve the learning rate (add it to the method's grid). Edit: `method_options.pinn_rans.lr: 0.003 → [0.00075, 0.0015]`.
- **warning** [airfoil/pinn_rans] No progress in the first 10 % of the budget: the learning rate may be too low. Evidence: loss 131 -> 341 over the first 10 % (final__airfoil__pinn_rans__L1.000__s0__s007_a1) Suggestion: Try a 3x larger learning rate. Edit: `method_options.pinn_rans.lr: 0.003 → [0.003, 0.009000000000000001]`.
- **warning** [airfoil/pinn_rans] Loss spikes in the second half of training: the learning rate is likely too high. Evidence: 13 logged losses above 3x the running minimum (final__airfoil__pinn_rans__L1.000__s0__s007_a2) Suggestion: Halve the learning rate (add it to the method's grid). Edit: `method_options.pinn_rans.lr: 0.003 → [0.00075, 0.0015]`.
- **warning** [inverse_heat/mlp] Loss spikes in the second half of training: the learning rate is likely too high. Evidence: 4 logged losses above 3x the running minimum (final__inverse_heat__mlp__L1.000__s0) Suggestion: Halve the learning rate (add it to the method's grid). Edit: `method_options.mlp.lr: 0.003 → [0.00075, 0.0015]`.
- **warning** [navier_stokes/fno] No progress in the first 10 % of the budget: the learning rate may be too low. Evidence: loss 0.211 -> 0.201 over the first 10 % (final__navier_stokes__fno__L1.000__s0) Suggestion: Try a 3x larger learning rate. Edit: `method_options.fno.lr: 0.003 → [0.003, 0.009000000000000001]`.
- **warning** [burgers] The OOD set is not clearly outside the training data. Evidence: applicability: OOD inputs outside the training envelope: 20/100 (threshold >= 50 %) Suggestion: Move the OOD inputs further out (amplitude, length scale, geometry, angle).
- **warning** [darcy] The OOD set is not clearly outside the training data. Evidence: OOD set outside the training distribution: 0.09 (threshold 0.5) Suggestion: Move the OOD inputs further out (amplitude, length scale, geometry, angle).
- **warning** [darcy] The OOD set is not clearly outside the training data. Evidence: applicability: OOD inputs outside the training envelope: 1/100 (threshold >= 50 %) Suggestion: Move the OOD inputs further out (amplitude, length scale, geometry, angle).
- **warning** [navier_stokes] The OOD set is not clearly outside the training data. Evidence: applicability: OOD inputs outside the training envelope: 23/100 (threshold >= 50 %) Suggestion: Move the OOD inputs further out (amplitude, length scale, geometry, angle).

## Validation

### Burgers 1-D: prepared artifacts ok

| Check | Status | Value | Detail |
|---|---|---|---|
| Burgers vs Cole-Hopf exact solution | pass | 2.96e-13 | max error at t = 0.5 for dt = 0.02, 0.01, 0.005: ['7.5e-11', '4.7e-12', '3.0e-13'] |
| ETDRK4 observed order in time (>= 3.5 or at round-off) | pass | 3.99 |  |
| finite values | pass |  | all arrays finite |
| split sizes as configured | pass | {"train": 1000, "val": 100, "test": 200, "ood": 100} |  |
| no duplicate inputs within a split | pass | {"train": 0, "val": 0, "test": 0, "ood": 0} |  |
| no input shared between splits (hashes) | pass | {"train&val": 0, "train&test": 0, "train&ood": 0, "val&test… |  |
| OOD set outside the training distribution | pass | 0.97 | fraction of OOD inputs beyond the test set's 99th percentile of the PCA(20) reconstruction error (0.105) or of the Mahalanobis distance in … |
| normalisation from the training split only | pass | train | FieldData computes means and standard deviations on train_x/train_y only |
| applicability: test inputs inside the training envelope | pass | 198/200 | physics_audit.Envelope over amplitude, spectral_centroid |
| applicability: OOD inputs outside the training envelope | warn | 20/100 | an 'OOD' set that sits inside the envelope tests interpolation, not extrapolation |
| grid convergence (GCI) of the energy at T | pass | spread 1.8e-05 across the 3 grids | converged; values (fine to coarse) ['0.036928378', '0.036928405', '0.036927726'] |
| physics audit: viscous Burgers, steep fronts | pass | VALID | viscous Burgers, steep fronts: VALID (compatibility 1.00);   [VALID] advection_dominated: Pe >= 10 (Pe = 172): VALID |

### Darcy 2-D: prepared artifacts ok

| Check | Status | Value | Detail |
|---|---|---|---|
| Darcy manufactured solution, observed order (max error) | pass | 2.0 | max errors ['3.4e-03', '8.6e-04', '2.1e-04'] |
| Darcy grid convergence of the mean (Richardson / GCI) | pass | p = 2.01, GCI_fine = 4.77e-04 | pinneapple_analysis.verification.richardson_extrapolate |
| finite values | pass |  | all arrays finite |
| split sizes as configured | pass | {"train": 1000, "val": 100, "test": 200, "ood": 100} |  |
| no duplicate inputs within a split | pass | {"train": 0, "val": 0, "test": 0, "ood": 0} |  |
| no input shared between splits (hashes) | pass | {"train&val": 0, "train&test": 0, "train&ood": 0, "val&test… |  |
| OOD set outside the training distribution | warn | 0.09 | fraction of OOD inputs beyond the test set's 99th percentile of the PCA(20) reconstruction error (0.715) or of the Mahalanobis distance in … |
| normalisation from the training split only | pass | train | FieldData computes means and standard deviations on train_x/train_y only |
| applicability: test inputs inside the training envelope | pass | 199/200 | physics_audit.Envelope over amplitude, spectral_centroid |
| applicability: OOD inputs outside the training envelope | warn | 1/100 | an 'OOD' set that sits inside the envelope tests interpolation, not extrapolation |
| grid convergence (GCI) of the mean of u | pass | p = 0.41, GCI_fine = 1.48e-02 | fitmesh.gci; values (fine to coarse) ['0.0052829968', '0.0052624712', '0.0052351957']; fine-mesh numerical uncertainty about 1.48% (observe… |
| physics audit: Darcy flow | pass | VALID | Darcy flow: VALID (compatibility 1.00);   [VALID] diffusion_dominated: Pe <= 1 (Pe = 0): VALID |

### Navier-Stokes 2-D: prepared artifacts ok

| Check | Status | Value | Detail |
|---|---|---|---|
| Taylor-Green vortex decays exactly | pass | 1.7e-15 |  |
| forced steady state is steady (this config) | pass | 3.5e-14 | nu = 0.001, dt = 0.005, grid 64^2 |
| finite values | pass |  | all arrays finite |
| split sizes as configured | pass | {"train": 1000, "val": 100, "test": 200, "ood": 100} |  |
| no duplicate inputs within a split | pass | {"train": 0, "val": 0, "test": 0, "ood": 0} |  |
| no input shared between splits (hashes) | pass | {"train&val": 0, "train&test": 0, "train&ood": 0, "val&test… |  |
| OOD set outside the training distribution | pass | 0.92 | fraction of OOD inputs beyond the test set's 99th percentile of the PCA(20) reconstruction error (0.507) or of the Mahalanobis distance in … |
| normalisation from the training split only | pass | train | FieldData computes means and standard deviations on train_x/train_y only |
| applicability: test inputs inside the training envelope | pass | 200/200 | physics_audit.Envelope over amplitude, spectral_centroid |
| applicability: OOD inputs outside the training envelope | warn | 23/100 | an 'OOD' set that sits inside the envelope tests interpolation, not extrapolation |
| grid convergence (GCI) of the enstrophy at T | pass | spread 5.8e-05 across the 3 grids | converged; values (fine to coarse) ['0.52366842', '0.52367235', '0.52364193'] |
| physics audit: 2-D incompressible Navier-Stokes | pass | VALID | 2-D incompressible Navier-Stokes: VALID (compatibility 1.00);   [VALID] laminar_periodic: Re <= 2.3e+03 (Re = 1e+03): VALID;   [VALID] cont… |

### Airfoil 2-D (RANS, Re 4e6): prepared artifacts ok

| Check | Status | Value | Detail |
|---|---|---|---|
| campaign solver convergence (converged or steady-ish) | pass | 93 % of 436 | status counts {'converged': 330, 'steady-ish': 74, 'unsteady': 32}; unsteady runs are averaged |
| NACA 0012 at 0 deg: drag vs the NASA Turbulence Modeling Resource (SST, Re 6e6) | pass | Cd 0.00871 | the app's own validation reference; Re 4e6 here, so drag is slightly higher |
| NACA 0012 lift slope vs the inviscid panel method (viscous slope 85-100 %) | pass | 6.37 /rad vs 6.84 /rad | pinneapple_design.aero.panel |
| no shape shared between training and test | pass | 0 |  |
| no run shared between splits | pass | {"test&train_all": 0, "naca&train_all": 0, "naca&test": 0, … |  |
| finite coefficients | pass |  |  |
| OOD cases outside the training box (CST weights or angle) | pass | 12/12 |  |
| normalisation from the training split only | pass |  | every trainer computes its means and scales on its training runs only |
| simulation preflight (pinneapple_data.preflight) | pass | WARNING | counts {'fail': 0, 'warn': 2, 'info': 0, 'pass': 9};  |
| mesh quality (pinneapple_data.cae.mesh_report) | pass | {"fail": 0, "warn": 0, "checks": 5} |  |
| first cell y+ (mesh_intelligence estimate) | pass | 0.596 | first cell 4.00e-06 vs 6.71e-06 needed for y+ = 1.0 |
| applicability: test inputs inside the training envelope | pass | 56/60 | physics_audit.Envelope over alpha, w0, w1, w2, w3, w4, w5 |
| applicability: OOD inputs outside the training envelope | pass | 12/12 | an 'OOD' set that sits inside the envelope tests interpolation, not extrapolation |
| physics audit: RANS k-omega SST, attached flow | warn | VIOLATED | RANS k-omega SST, attached flow: VIOLATED (compatibility 0.17);   [VALID] fully_turbulent: Re >= 5e+05 (Re = 4e+06): VALID;   [VALID] incom… |
| grid convergence (GCI) of Cd (NACA 2412, 4 deg) | pass | p = 2.79, GCI_fine = 1.23e-02 | fitmesh.gci; values (fine to coarse) ['0.009941089', '0.010142987', '0.010746264']; fine-mesh numerical uncertainty about 1.23% (observed o… |

### Inverse heat (h from 8 sensors): prepared artifacts ok

| Check | Status | Value | Detail |
|---|---|---|---|
| grid convergence (GCI) of the hot-spot temperature (h = 25, 10 W) | pass | spread 3.4e-04 across the 3 grids | converged; values (fine to coarse) ['69.962614', '69.95814', '69.938747'] |
| finite values | pass |  |  |
| split sizes as configured | pass | {"train": 200, "val": 40} |  |
| no (h, power) shared between train and val | pass | 0 |  |
| OOD cases outside the surrogates' h range | pass | [150.0, 200.0] |  |
| in-range cases inside the surrogates' h range | pass | [25.0, 60.0, 10.0] |  |
| normalisation from the training split only | pass |  | Data() scales by the training fields' standard deviation |
| applicability: test inputs inside the training envelope | pass | 3/3 | physics_audit.Envelope over h, power |
| applicability: OOD inputs outside the training envelope | pass | 2/2 | an 'OOD' set that sits inside the envelope tests interpolation, not extrapolation |
| physics audit: thin-plate conduction with face convection | pass | VALID | thin-plate conduction with face convection: VALID (compatibility 1.00);   [VALID] continuum: Kn <= 0.01 (Kn = 0): VALID |

## Trust card

### Burgers 1-D

| Stage | Question | Status | Evidence |
|---|---|---|---|
| Data and geometry | Was the physical problem represented correctly? | partial | PASS finite values (all arrays finite); PASS no duplicate inputs within a split; PASS no input shared between splits (hashes); PASS grid convergence (GCI) of t… |
| Model | Can the model (or solver) represent the relevant dynamics? | answered | PASS p >= rank of the training outputs (99 % energy); PASS autograd derivatives match finite differences (float32, central differences, h = 1e-3); PASS overfit… |
| Physical constraints | Are the equations and the boundary conditions respected? | answered | PASS physics audit: viscous Burgers, steep fronts (viscous Burgers, steep fronts: VALID (compatibility 1.00);   [VALID] advection_dominated: ); PASS PDE residu… |
| Benchmark | How does the result compare with a solver or reference data? | answered | PASS Burgers vs Cole-Hopf exact solution (max error at t = 0.5 for dt = 0.02, 0.01, 0.005: ['7.5e-11', '4.7e-12', '3.0e-13']); PASS ETDRK4 observed order in ti… |
| Uncertainty | Where may the prediction not be reliable? | failed | FAIL applicability: OOD inputs outside the training envelope (an 'OOD' set that sits inside the envelope tests interpolation, not extrapolation); FAIL seed spr… |
| Engineering decision | Is the result adequate for the intended use? | partial | tier B (solid) |

### Darcy 2-D

| Stage | Question | Status | Evidence |
|---|---|---|---|
| Data and geometry | Was the physical problem represented correctly? | partial | PASS finite values (all arrays finite); PASS no duplicate inputs within a split; PASS no input shared between splits (hashes); PASS grid convergence (GCI) of t… |
| Model | Can the model (or solver) represent the relevant dynamics? | failed | FAIL overfits a tiny batch (loss 1.18 -> 0.151 in 1000 Adam steps (at most 1000, stops once below)); PASS p >= rank of the training outputs (99 % energy); PASS… |
| Physical constraints | Are the equations and the boundary conditions respected? | answered | PASS physics audit: Darcy flow (Darcy flow: VALID (compatibility 1.00);   [VALID] diffusion_dominated: Pe <= 1 (Pe = 0): V); PASS PDE residual of an exact solu… |
| Benchmark | How does the result compare with a solver or reference data? | failed | FAIL learned (final loss < half the initial) (final__darcy__pinn__L1.000__s2__test2 (1.46 -> 0.806); flops__darcy__pinn__s2 (1.53 -> 0.7); PASS Darcy manufactu… |
| Uncertainty | Where may the prediction not be reliable? | failed | FAIL OOD set outside the training distribution (fraction of OOD inputs beyond the test set's 99th percentile of the PCA(20) reconstruction); FAIL applicability… |
| Engineering decision | Is the result adequate for the intended use? | partial | tier B (solid) |

### Navier-Stokes 2-D

| Stage | Question | Status | Evidence |
|---|---|---|---|
| Data and geometry | Was the physical problem represented correctly? | partial | PASS finite values (all arrays finite); PASS no duplicate inputs within a split; PASS no input shared between splits (hashes); PASS grid convergence (GCI) of t… |
| Model | Can the model (or solver) represent the relevant dynamics? | failed | FAIL overfits a tiny batch (loss 1.08 -> 0.236 in 1000 Adam steps (at most 1000, stops once below)); PASS p >= rank of the training outputs (99 % energy); PASS… |
| Physical constraints | Are the equations and the boundary conditions respected? | answered | PASS physics audit: 2-D incompressible Navier-Stokes (2-D incompressible Navier-Stokes: VALID (compatibility 1.00);   [VALID] laminar_periodic: ); PASS PDE res… |
| Benchmark | How does the result compare with a solver or reference data? | failed | FAIL loss decreasing (final__navier_stokes__fno__L0.111__s0 (0.211 -> 0.223)); FAIL learned (final loss < half the initial) (final__navier_stokes__pinn__L1.000… |
| Uncertainty | Where may the prediction not be reliable? | failed | FAIL applicability: OOD inputs outside the training envelope (an 'OOD' set that sits inside the envelope tests interpolation, not extrapolation); FAIL seed spr… |
| Engineering decision | Is the result adequate for the intended use? | partial | tier B (solid) |

### Airfoil 2-D (RANS, Re 4e6)

| Stage | Question | Status | Evidence |
|---|---|---|---|
| Data and geometry | Was the physical problem represented correctly? | partial | PASS finite coefficients; PASS mesh quality (pinneapple_data.cae.mesh_report); PASS first cell y+ (mesh_intelligence estimate) (first cell 4.00e-06 vs 6.71e-06… |
| Model | Can the model (or solver) represent the relevant dynamics? | answered | PASS forward shapes; PASS parameter count; PASS message passing honours the graph (node permutation equivariance) (fields of the permuted graph = permuted fiel… |
| Physical constraints | Are the equations and the boundary conditions respected? | failed | FAIL physics audit: RANS k-omega SST, attached flow (RANS k-omega SST, attached flow: VIOLATED (compatibility 0.17);   [VALID] fully_turbulent:); PASS PDE resi… |
| Benchmark | How does the result compare with a solver or reference data? | failed | FAIL no divergence (final__airfoil__pinn_rans__L1.000__s0__NACA2412_a-04 (final 14.5, min 0.699); final__airfo); PASS campaign solver convergence (converged or… |
| Uncertainty | Where may the prediction not be reliable? | failed | FAIL seed spread of u on test4 within the veriphysics tolerance; FAIL seed spread of cd on test4 within the veriphysics tolerance; FAIL seed spread of u on tes… |
| Engineering decision | Is the result adequate for the intended use? | open | tier C (exploratory) |

### Inverse heat (h from 8 sensors)

| Stage | Question | Status | Evidence |
|---|---|---|---|
| Data and geometry | Was the physical problem represented correctly? | partial | PASS finite values; PASS in-range cases inside the surrogates' h range |
| Model | Can the model (or solver) represent the relevant dynamics? | answered | PASS forward shape; PASS parameter count; PASS overfits a tiny batch (loss 1.63 -> 0.138 in 28 Adam steps (at most 1000, stops once below)); PASS forward shape… |
| Physical constraints | Are the equations and the boundary conditions respected? | answered | PASS physics audit: thin-plate conduction with face convection (thin-plate conduction with face convection: VALID (compatibility 1.00);   [VALID] continuu) |
| Benchmark | How does the result compare with a solver or reference data? | answered | PASS grid convergence (GCI) of the hot-spot temperature (h = 25, 10 W) (converged; values (fine to coarse) ['69.962614', '69.95814', '69.938747']); PASS loss f… |
| Uncertainty | Where may the prediction not be reliable? | failed | FAIL seed spread of h on id within the veriphysics tolerance; FAIL seed spread of h on id within the veriphysics tolerance; FAIL seed spread of h on id within … |
| Engineering decision | Is the result adequate for the intended use? | partial | tier B (solid) |

## Checks

| Stage | Subject | Check | Status | Value | Threshold |
|---|---|---|---|---|---|
| problem | burgers | Burgers vs Cole-Hopf exact solution | pass | 2.96e-13 | 1e-06 |
| problem | burgers | ETDRK4 observed order in time (>= 3.5 or at round-off) | pass | 3.99 | 3.5 |
| data | burgers | finite values | pass |  |  |
| data | burgers | split sizes as configured | pass | {"train": 1000, "val": 100, "test": 200, "ood": 100} | {"train": 1000, "val": 100, "test": 200, "ood": 100} |
| data | burgers | no duplicate inputs within a split | pass | {"train": 0, "val": 0, "test": 0, "ood": 0} | 0 |
| data | burgers | no input shared between splits (hashes) | pass | {"train&val": 0, "train&test": 0, "train&ood": 0, "val&test… | 0 |
| data | burgers | OOD set outside the training distribution | pass | 0.97 | 0.5 |
| data | burgers | normalisation from the training split only | pass | train |  |
| validation | burgers | applicability: test inputs inside the training envelope | pass | 198/200 | >= 90 % |
| validation | burgers | applicability: OOD inputs outside the training envelope | warn | 20/100 | >= 50 % |
| validation | burgers | grid convergence (GCI) of the energy at T | pass | spread 1.8e-05 across the 3 grids | spread < 1e-3 or GCI <= 0.05 |
| validation | burgers | physics audit: viscous Burgers, steep fronts | pass | VALID |  |
| problem | darcy | Darcy manufactured solution, observed order (max error) | pass | 2.0 | 1.8 |
| problem | darcy | Darcy grid convergence of the mean (Richardson / GCI) | pass | p = 2.01, GCI_fine = 4.77e-04 | p > 1.5 |
| data | darcy | finite values | pass |  |  |
| data | darcy | split sizes as configured | pass | {"train": 1000, "val": 100, "test": 200, "ood": 100} | {"train": 1000, "val": 100, "test": 200, "ood": 100} |
| data | darcy | no duplicate inputs within a split | pass | {"train": 0, "val": 0, "test": 0, "ood": 0} | 0 |
| data | darcy | no input shared between splits (hashes) | pass | {"train&val": 0, "train&test": 0, "train&ood": 0, "val&test… | 0 |
| data | darcy | OOD set outside the training distribution | warn | 0.09 | 0.5 |
| data | darcy | normalisation from the training split only | pass | train |  |
| validation | darcy | applicability: test inputs inside the training envelope | pass | 199/200 | >= 90 % |
| validation | darcy | applicability: OOD inputs outside the training envelope | warn | 1/100 | >= 50 % |
| validation | darcy | grid convergence (GCI) of the mean of u | pass | p = 0.41, GCI_fine = 1.48e-02 | GCI <= 0.05 |
| validation | darcy | physics audit: Darcy flow | pass | VALID |  |
| problem | navier_stokes | Taylor-Green vortex decays exactly | pass | 1.7e-15 | 1e-08 |
| problem | navier_stokes | forced steady state is steady (this config) | pass | 3.5e-14 | 1e-08 |
| data | navier_stokes | finite values | pass |  |  |
| data | navier_stokes | split sizes as configured | pass | {"train": 1000, "val": 100, "test": 200, "ood": 100} | {"train": 1000, "val": 100, "test": 200, "ood": 100} |
| data | navier_stokes | no duplicate inputs within a split | pass | {"train": 0, "val": 0, "test": 0, "ood": 0} | 0 |
| data | navier_stokes | no input shared between splits (hashes) | pass | {"train&val": 0, "train&test": 0, "train&ood": 0, "val&test… | 0 |
| data | navier_stokes | OOD set outside the training distribution | pass | 0.92 | 0.5 |
| data | navier_stokes | normalisation from the training split only | pass | train |  |
| validation | navier_stokes | applicability: test inputs inside the training envelope | pass | 200/200 | >= 90 % |
| validation | navier_stokes | applicability: OOD inputs outside the training envelope | warn | 23/100 | >= 50 % |
| validation | navier_stokes | grid convergence (GCI) of the enstrophy at T | pass | spread 5.8e-05 across the 3 grids | spread < 1e-3 or GCI <= 0.05 |
| validation | navier_stokes | physics audit: 2-D incompressible Navier-Stokes | pass | VALID |  |
| problem | airfoil | campaign solver convergence (converged or steady-ish) | pass | 93 % of 436 | >= 85 % |
| problem | airfoil | NACA 0012 at 0 deg: drag vs the NASA Turbulence Modeling Resource (SST, Re 6e6) | pass | Cd 0.00871 | 0.0081 +- 12 % |
| problem | airfoil | NACA 0012 lift slope vs the inviscid panel method (viscous slope 85-100 %) | pass | 6.37 /rad vs 6.84 /rad | 0.85..1.0 |
| data | airfoil | no shape shared between training and test | pass | 0 | 0 |
| data | airfoil | no run shared between splits | pass | {"test&train_all": 0, "naca&train_all": 0, "naca&test": 0, … |  |
| data | airfoil | finite coefficients | pass |  |  |
| data | airfoil | OOD cases outside the training box (CST weights or angle) | pass | 12/12 | all |
| data | airfoil | normalisation from the training split only | pass |  |  |
| validation | airfoil | simulation preflight (pinneapple_data.preflight) | pass | WARNING |  |
| validation | airfoil | mesh quality (pinneapple_data.cae.mesh_report) | pass | {"fail": 0, "warn": 0, "checks": 5} |  |
| validation | airfoil | first cell y+ (mesh_intelligence estimate) | pass | 0.596 | <= 1.5 |
| validation | airfoil | applicability: test inputs inside the training envelope | pass | 56/60 | >= 90 % |
| validation | airfoil | applicability: OOD inputs outside the training envelope | pass | 12/12 | >= 50 % |
| validation | airfoil | physics audit: RANS k-omega SST, attached flow | warn | VIOLATED |  |
| validation | airfoil | grid convergence (GCI) of Cd (NACA 2412, 4 deg) | pass | p = 2.79, GCI_fine = 1.23e-02 | GCI <= 0.05 |
| problem | inverse_heat | grid convergence (GCI) of the hot-spot temperature (h = 25, 10 W) | pass | spread 3.4e-04 across the 3 grids | spread < 1e-3 or GCI <= 0.02 |
| data | inverse_heat | finite values | pass |  |  |
| data | inverse_heat | split sizes as configured | pass | {"train": 200, "val": 40} | {"train": 200, "val": 40} |
| data | inverse_heat | no (h, power) shared between train and val | pass | 0 |  |
| data | inverse_heat | OOD cases outside the surrogates' h range | pass | [150.0, 200.0] | [5.0, 100.0] |
| data | inverse_heat | in-range cases inside the surrogates' h range | pass | [25.0, 60.0, 10.0] |  |
| data | inverse_heat | normalisation from the training split only | pass |  |  |
| validation | inverse_heat | applicability: test inputs inside the training envelope | pass | 3/3 | >= 90 % |
| validation | inverse_heat | applicability: OOD inputs outside the training envelope | pass | 2/2 | >= 50 % |
| validation | inverse_heat | physics audit: thin-plate conduction with face convection | pass | VALID |  |
| architecture | burgers/pinn | autograd derivatives match finite differences | pass | 7.93e-03 | 0.01 |
| architecture | burgers/pinn | PDE residual of an exact solution ~ 0 | pass | 3.64e-18 | 1e-06 |
| architecture | burgers/pinn | overfits a tiny batch | pass | 0.09929 | < 0.1 (error at >= 0.5) |
| architecture | burgers/pinn | forward shape and dtype | pass | (256,) vs data (256,) |  |
| architecture | burgers/pinn | parameter count | pass | 12929 |  |
| architecture | burgers/fno | forward shape and dtype | pass | (4, 256) -> (4, 256) torch.float32 |  |
| architecture | burgers/fno | parameter count | pass | 102657 |  |
| architecture | burgers/fno | retained Fourier modes <= Nyquist | pass | 24 | 129 |
| architecture | burgers/fno | output invariant to input resolution (2x grid, untrained) | pass | 0.0001 | 0.1 |
| architecture | burgers/fno | overfits a tiny batch | pass | 0.08453 | < 0.1 (error at >= 0.5) |
| architecture | burgers/deeponet | forward shape and dtype | pass | (4, 256) -> (4, 256) torch.float32 |  |
| architecture | burgers/deeponet | parameter count | pass | 132225 |  |
| architecture | burgers/deeponet | branch/trunk shapes | pass | branch 256->(3, 128), trunk 1->(256, 128) |  |
| architecture | burgers/deeponet | p >= rank of the training outputs (99 % energy) | pass | 128 | 8 |
| architecture | burgers/deeponet | overfits a tiny batch | pass | 0.09837 | < 0.1 (error at >= 0.5) |
| architecture | darcy/pinn | autograd derivatives match finite differences | pass | 8.37e-05 | 0.01 |
| architecture | darcy/pinn | overfits a tiny batch | pass | 0.0956 | < 0.1 (error at >= 0.5) |
| architecture | darcy/pinn | forward shape and dtype | pass | (64, 64) vs data (64, 64) |  |
| architecture | darcy/pinn | parameter count | pass | 12737 |  |
| architecture | darcy/pinn | PDE residual of an exact solution ~ 0 | pass | 5.62e-16 | 1e-06 |
| architecture | darcy/fno | forward shape and dtype | pass | (4, 64, 64) -> (4, 64, 64) torch.float32 |  |
| architecture | darcy/fno | parameter count | pass | 1182169 |  |
| architecture | darcy/fno | retained Fourier modes <= Nyquist | pass | 16 | 32 |
| architecture | darcy/fno | overfits a tiny batch | pass | 0.09749 | < 0.1 (error at >= 0.5) |
| architecture | darcy/deeponet | forward shape and dtype | pass | (4, 64, 64) -> (4, 64, 64) torch.float32 |  |
| architecture | darcy/deeponet | parameter count | pass | 623873 |  |
| architecture | darcy/deeponet | branch/trunk shapes | pass | branch 4096->(3, 128), trunk 2->(4096, 128) |  |
| architecture | darcy/deeponet | p >= rank of the training outputs (99 % energy) | pass | 128 | 45 |
| architecture | darcy/deeponet | overfits a tiny batch | warn | 0.12748 | < 0.1 (error at >= 0.5) |
| architecture | navier_stokes/pinn | autograd derivatives match finite differences | pass | 1.02e-03 | 0.01 |
| architecture | navier_stokes/pinn | PDE residual of an exact solution ~ 0 | pass | 4.78e-14 | 1e-06 |
| architecture | navier_stokes/pinn | overfits a tiny batch | pass | 0.09296 | < 0.1 (error at >= 0.5) |
| architecture | navier_stokes/pinn | forward shape and dtype | pass | (64, 64) vs data (64, 64) |  |
| architecture | navier_stokes/pinn | parameter count | pass | 13250 |  |
| architecture | navier_stokes/fno | forward shape and dtype | pass | (4, 64, 64) -> (4, 64, 64) torch.float32 |  |
| architecture | navier_stokes/fno | parameter count | pass | 1182169 |  |
| architecture | navier_stokes/fno | retained Fourier modes <= Nyquist | pass | 16 | 32 |
| architecture | navier_stokes/fno | overfits a tiny batch | pass | 0.09509 | < 0.1 (error at >= 0.5) |
| architecture | navier_stokes/deeponet | forward shape and dtype | pass | (4, 64, 64) -> (4, 64, 64) torch.float32 |  |
| architecture | navier_stokes/deeponet | parameter count | pass | 623873 |  |
| architecture | navier_stokes/deeponet | branch/trunk shapes | pass | branch 4096->(3, 128), trunk 2->(4096, 128) |  |
| architecture | navier_stokes/deeponet | p >= rank of the training outputs (99 % energy) | pass | 128 | 12 |
| architecture | navier_stokes/deeponet | overfits a tiny batch | warn | 0.21784 | < 0.1 (error at >= 0.5) |
| architecture | airfoil/pinn | PDE residual of an exact solution ~ 0 | pass | 0.0e+00 (sign-flipped: 1.6e+01) | < 1e-12, flipped > 1e-3 |
| architecture | airfoil/mgn | forward shapes | pass | fields (1, 2850, 4), coefficients (1, 3) |  |
| architecture | airfoil/mgn | parameter count | pass | 108551 |  |
| architecture | airfoil/mgn | message passing honours the graph (node permutation equivariance) | pass | 0.0e+00 | 0.0001 |
| architecture | airfoil/mlp | forward shape | pass | (5, 6, 3) |  |
| architecture | airfoil/mlp | parameter count | pass | 172815 |  |
| architecture | airfoil/mlp | overfits a tiny batch | pass | 0.09814 | < 0.1 (error at >= 0.5) |
| architecture | airfoil/mlp_fields | POD basis orthonormal | pass | 2.1e-15 | 1e-08 |
| architecture | airfoil/mlp_fields | POD reconstruction error = sqrt(1 - energy kept) | pass | 0.0091 vs 0.0091 |  |
| architecture | airfoil/mlp_fields | forward shape | pass | (6, 16) |  |
| architecture | airfoil/mlp_fields | parameter count | pass | 36240 |  |
| architecture | airfoil/mlp_fields | overfits a tiny batch | pass | 0.09833 | < 0.1 (error at >= 0.5) |
| architecture | airfoil/pinn_rans | PDE residual of an exact solution ~ 0 | pass | 2.4e-30 (sign-flipped: 2.2e+00) | < 1e-12, flipped > 1e-3 |
| architecture | inverse_heat/pinn | PDE residual of an exact solution ~ 0 | n/a |  |  |
| architecture | inverse_heat/deeponet | forward shape | pass | (3, 60, 100) |  |
| architecture | inverse_heat/deeponet | parameter count | pass | 25345 |  |
| architecture | inverse_heat/deeponet | overfits a tiny batch | pass | 0.0844 | < 0.1 (error at >= 0.5) |
| architecture | inverse_heat/mlp | forward shape | pass | (3, 60, 100) |  |
| architecture | inverse_heat/mlp | parameter count | pass | 402672 |  |
| architecture | inverse_heat/mlp | overfits a tiny batch | pass | 0.0823 | < 0.1 (error at >= 0.5) |
| training | airfoil/mgn | loss finite | pass | 13/13 jobs |  |
| training | airfoil/mgn | loss decreasing | pass | 13/13 jobs |  |
| training | airfoil/mgn | no divergence | pass | 13/13 jobs |  |
| training | airfoil/mgn | learned (final loss < half the initial) | pass | 3/3 jobs |  |
| training | airfoil/mgn | budget enforced | pass | 13/13 jobs |  |
| training | airfoil/mgn | evaluation off the clock | pass | 13/13 jobs |  |
| training | airfoil/mgn | search used validation data only | pass | 4/4 jobs |  |
| training | airfoil/mlp | loss finite | pass | 13/13 jobs |  |
| training | airfoil/mlp | loss decreasing | pass | 13/13 jobs |  |
| training | airfoil/mlp | no divergence | pass | 13/13 jobs |  |
| training | airfoil/mlp | learned (final loss < half the initial) | pass | 3/3 jobs |  |
| training | airfoil/mlp | budget enforced | pass | 13/13 jobs |  |
| training | airfoil/mlp | evaluation off the clock | pass | 13/13 jobs |  |
| training | airfoil/mlp | search used validation data only | pass | 4/4 jobs |  |
| training | airfoil/mlp_fields | loss finite | pass | 13/13 jobs |  |
| training | airfoil/mlp_fields | loss decreasing | pass | 13/13 jobs |  |
| training | airfoil/mlp_fields | no divergence | pass | 13/13 jobs |  |
| training | airfoil/mlp_fields | learned (final loss < half the initial) | pass | 3/3 jobs |  |
| training | airfoil/mlp_fields | budget enforced | pass | 13/13 jobs |  |
| training | airfoil/mlp_fields | evaluation off the clock | pass | 13/13 jobs |  |
| training | airfoil/mlp_fields | search used validation data only | pass | 4/4 jobs |  |
| training | airfoil/pinn | loss finite | pass | 29/29 jobs |  |
| training | airfoil/pinn | loss decreasing | pass | 29/29 jobs |  |
| training | airfoil/pinn | no divergence | pass | 29/29 jobs |  |
| training | airfoil/pinn | learned (final loss < half the initial) | pass | 19/19 jobs |  |
| training | airfoil/pinn | budget enforced | pass | 29/29 jobs |  |
| training | airfoil/pinn | evaluation off the clock | pass | 29/29 jobs |  |
| training | airfoil/pinn | search used validation data only | pass | 4/4 jobs |  |
| training | airfoil/pinn_rans | loss finite | pass | 29/29 jobs |  |
| training | airfoil/pinn_rans | loss decreasing | pass | 29/29 jobs |  |
| training | airfoil/pinn_rans | no divergence | FAIL | 27/29 jobs |  |
| training | airfoil/pinn_rans | learned (final loss < half the initial) | pass | 19/19 jobs |  |
| training | airfoil/pinn_rans | budget enforced | pass | 29/29 jobs |  |
| training | airfoil/pinn_rans | evaluation off the clock | pass | 29/29 jobs |  |
| training | airfoil/pinn_rans | search used validation data only | pass | 4/4 jobs |  |
| training | burgers/deeponet | loss finite | pass | 16/16 jobs |  |
| training | burgers/deeponet | loss decreasing | pass | 16/16 jobs |  |
| training | burgers/deeponet | no divergence | pass | 16/16 jobs |  |
| training | burgers/deeponet | learned (final loss < half the initial) | pass | 6/6 jobs |  |
| training | burgers/deeponet | budget enforced | pass | 16/16 jobs |  |
| training | burgers/deeponet | evaluation off the clock | pass | 16/16 jobs |  |
| training | burgers/deeponet | search used validation data only | pass | 4/4 jobs |  |
| training | burgers/fno | loss finite | pass | 16/16 jobs |  |
| training | burgers/fno | loss decreasing | pass | 16/16 jobs |  |
| training | burgers/fno | no divergence | pass | 16/16 jobs |  |
| training | burgers/fno | learned (final loss < half the initial) | pass | 6/6 jobs |  |
| training | burgers/fno | budget enforced | pass | 16/16 jobs |  |
| training | burgers/fno | evaluation off the clock | pass | 16/16 jobs |  |
| training | burgers/fno | search used validation data only | pass | 4/4 jobs |  |
| training | burgers/pinn | loss finite | pass | 40/40 jobs |  |
| training | burgers/pinn | loss decreasing | pass | 40/40 jobs |  |
| training | burgers/pinn | no divergence | pass | 40/40 jobs |  |
| training | burgers/pinn | learned (final loss < half the initial) | pass | 18/18 jobs |  |
| training | burgers/pinn | budget enforced | pass | 40/40 jobs |  |
| training | burgers/pinn | evaluation off the clock | pass | 40/40 jobs |  |
| training | burgers/pinn | search used validation data only | pass | 4/4 jobs |  |
| training | darcy/deeponet | loss finite | pass | 16/16 jobs |  |
| training | darcy/deeponet | loss decreasing | pass | 16/16 jobs |  |
| training | darcy/deeponet | no divergence | pass | 16/16 jobs |  |
| training | darcy/deeponet | learned (final loss < half the initial) | pass | 6/6 jobs |  |
| training | darcy/deeponet | budget enforced | pass | 16/16 jobs |  |
| training | darcy/deeponet | evaluation off the clock | pass | 16/16 jobs |  |
| training | darcy/deeponet | search used validation data only | pass | 4/4 jobs |  |
| training | darcy/fno | loss finite | pass | 16/16 jobs |  |
| training | darcy/fno | loss decreasing | pass | 16/16 jobs |  |
| training | darcy/fno | no divergence | pass | 16/16 jobs |  |
| training | darcy/fno | learned (final loss < half the initial) | pass | 6/6 jobs |  |
| training | darcy/fno | budget enforced | pass | 16/16 jobs |  |
| training | darcy/fno | evaluation off the clock | pass | 16/16 jobs |  |
| training | darcy/fno | search used validation data only | pass | 4/4 jobs |  |
| training | darcy/pinn | loss finite | pass | 40/40 jobs |  |
| training | darcy/pinn | loss decreasing | pass | 40/40 jobs |  |
| training | darcy/pinn | no divergence | pass | 40/40 jobs |  |
| training | darcy/pinn | learned (final loss < half the initial) | warn | 16/18 jobs |  |
| training | darcy/pinn | budget enforced | pass | 40/40 jobs |  |
| training | darcy/pinn | evaluation off the clock | pass | 40/40 jobs |  |
| training | darcy/pinn | search used validation data only | pass | 4/4 jobs |  |
| training | inverse_heat/deeponet | loss finite | pass | 13/13 jobs |  |
| training | inverse_heat/deeponet | loss decreasing | pass | 13/13 jobs |  |
| training | inverse_heat/deeponet | no divergence | pass | 13/13 jobs |  |
| training | inverse_heat/deeponet | learned (final loss < half the initial) | pass | 3/3 jobs |  |
| training | inverse_heat/deeponet | budget enforced | pass | 13/13 jobs |  |
| training | inverse_heat/deeponet | evaluation off the clock | pass | 13/13 jobs |  |
| training | inverse_heat/deeponet | search used validation data only | pass | 4/4 jobs |  |
| training | inverse_heat/mlp | loss finite | pass | 13/13 jobs |  |
| training | inverse_heat/mlp | loss decreasing | pass | 13/13 jobs |  |
| training | inverse_heat/mlp | no divergence | pass | 13/13 jobs |  |
| training | inverse_heat/mlp | learned (final loss < half the initial) | pass | 3/3 jobs |  |
| training | inverse_heat/mlp | budget enforced | pass | 13/13 jobs |  |
| training | inverse_heat/mlp | evaluation off the clock | pass | 13/13 jobs |  |
| training | inverse_heat/mlp | search used validation data only | pass | 4/4 jobs |  |
| training | inverse_heat/pinn | loss finite | pass | 49/49 jobs |  |
| training | inverse_heat/pinn | loss decreasing | pass | 49/49 jobs |  |
| training | inverse_heat/pinn | no divergence | pass | 49/49 jobs |  |
| training | inverse_heat/pinn | learned (final loss < half the initial) | pass | 15/15 jobs |  |
| training | inverse_heat/pinn | budget enforced | pass | 49/49 jobs |  |
| training | inverse_heat/pinn | evaluation off the clock | pass | 49/49 jobs |  |
| training | inverse_heat/pinn | search used validation data only | pass | 4/4 jobs |  |
| training | navier_stokes/deeponet | loss finite | pass | 16/16 jobs |  |
| training | navier_stokes/deeponet | loss decreasing | pass | 16/16 jobs |  |
| training | navier_stokes/deeponet | no divergence | pass | 16/16 jobs |  |
| training | navier_stokes/deeponet | learned (final loss < half the initial) | pass | 6/6 jobs |  |
| training | navier_stokes/deeponet | budget enforced | pass | 16/16 jobs |  |
| training | navier_stokes/deeponet | evaluation off the clock | pass | 16/16 jobs |  |
| training | navier_stokes/deeponet | search used validation data only | pass | 4/4 jobs |  |
| training | navier_stokes/fno | loss finite | pass | 16/16 jobs |  |
| training | navier_stokes/fno | loss decreasing | warn | 15/16 jobs |  |
| training | navier_stokes/fno | no divergence | pass | 16/16 jobs |  |
| training | navier_stokes/fno | learned (final loss < half the initial) | pass | 6/6 jobs |  |
| training | navier_stokes/fno | budget enforced | pass | 16/16 jobs |  |
| training | navier_stokes/fno | evaluation off the clock | pass | 16/16 jobs |  |
| training | navier_stokes/fno | search used validation data only | pass | 4/4 jobs |  |
| training | navier_stokes/pinn | loss finite | pass | 40/40 jobs |  |
| training | navier_stokes/pinn | loss decreasing | pass | 40/40 jobs |  |
| training | navier_stokes/pinn | no divergence | pass | 40/40 jobs |  |
| training | navier_stokes/pinn | learned (final loss < half the initial) | warn | 15/18 jobs |  |
| training | navier_stokes/pinn | budget enforced | pass | 40/40 jobs |  |
| training | navier_stokes/pinn | evaluation off the clock | pass | 40/40 jobs |  |
| training | navier_stokes/pinn | search used validation data only | pass | 4/4 jobs |  |
| training | all | identical accounting across methods (hardware, threads, timer, FLOP counter) | pass | [["Intel(R) Xeon(R) Processor @ 2.10GHz", "2.14.1+cu130", 1… |  |
| training | burgers/pinn | seed spread of rel_l2 on pinn_test within the veriphysics tolerance | warn | 0.331 | 0.1 |
| training | burgers/fno | seed spread of rel_l2 on pinn_test within the veriphysics tolerance | warn | 0.203 | 0.1 |
| training | burgers/deeponet | seed spread of rel_l2 on pinn_test within the veriphysics tolerance | warn | 0.126 | 0.1 |
| training | darcy/pinn | seed spread of rel_l2 on pinn_test within the veriphysics tolerance | pass | 0.005 | 0.1 |
| training | darcy/fno | seed spread of rel_l2 on pinn_test within the veriphysics tolerance | pass | 0.029 | 0.1 |
| training | darcy/deeponet | seed spread of rel_l2 on pinn_test within the veriphysics tolerance | pass | 0.056 | 0.1 |
| training | navier_stokes/pinn | seed spread of rel_l2 on pinn_test within the veriphysics tolerance | pass | 0.035 | 0.1 |
| training | navier_stokes/fno | seed spread of rel_l2 on pinn_test within the veriphysics tolerance | warn | 0.131 | 0.1 |
| training | navier_stokes/deeponet | seed spread of rel_l2 on pinn_test within the veriphysics tolerance | warn | 0.167 | 0.1 |
| training | airfoil/pinn | seed spread of u on test4 within the veriphysics tolerance | pass | 0.014 | 0.1 |
| training | airfoil/mgn | seed spread of u on test4 within the veriphysics tolerance | warn | 0.292 | 0.1 |
| training | airfoil/mlp | seed spread of cd on test4 within the veriphysics tolerance | warn | 0.506 | 0.1 |
| training | airfoil/mlp_fields | seed spread of u on test4 within the veriphysics tolerance | pass | 0.079 | 0.1 |
| training | airfoil/pinn_rans | seed spread of u on test4 within the veriphysics tolerance | pass | 0.004 | 0.1 |
| training | inverse_heat/pinn | seed spread of h on id within the veriphysics tolerance | warn | 0.467 | 0.1 |
| training | inverse_heat/deeponet | seed spread of h on id within the veriphysics tolerance | warn | 0.29 | 0.1 |
| training | inverse_heat/mlp | seed spread of h on id within the veriphysics tolerance | warn | 0.618 | 0.1 |
| training | burgers/pinn | determinism (same seed, same steps, same loss) | pass | 17.4283 vs 17.4283 | 1e-05 |
