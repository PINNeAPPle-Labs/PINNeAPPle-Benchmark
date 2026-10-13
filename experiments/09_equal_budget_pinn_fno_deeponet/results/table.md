# Experiment 09: equal-budget results

Hardware: Intel(R) Xeon(R) Processor @ 2.10GHz (4 logical CPUs, 15.7 GB), no GPU; torch 2.14.1+cu130. **Every training run uses ONE CPU core** (torch.set_num_threads(1)); four runs execute concurrently, for every method alike. Budgets are small because there is no GPU.

Mean ± standard deviation over 3 seeds (sample standard deviation). Test errors marked *3 inst.* are on the three test instances every PINN was trained on (the operators on exactly the same instances); *all* is the operators' whole test set (200). Time per new instance: operator = one forward pass (batch 1); PINN = its training on that instance (it has to be retrained for every new input) plus evaluation; reference solver = one solve. Break-even: number of new queries above which the operator, including its simulation data and training, costs less than per-instance PINNs (all core-seconds; '0' = always cheaper).

## Burgers 1-D (budget 300 s on one core per training run)

| Method | Rel. L2, test (3 inst.) | Rel. L2, test (all) | Rel. L2, OOD (2 inst.) | Integral QoI rel. err (3 inst.) | Training (s) | Training FLOPs | Simulation data (core-s) | Time per new instance | Break-even (queries) |
|---|---|---|---|---|---|---|---|---|---|
| PINN | 0.246 ± 0.081 | - | 0.592 ± 0.051 | 0.118 ± 0.024 | 300 s | 2.19e+12 | - | 300 s | - |
| FNO | 0.00168 ± 0.00034 | 0.00328 ± 0.0003 | 0.0393 ± 0.0017 | 0.00193 ± 0.0011 | 300 s | 2.68e+12 | 1170 s | 1.7 ms | 4.9 |
| DeepONet | 0.0155 ± 0.002 | 0.0416 ± 0.0013 | 0.228 ± 0.033 | 0.0109 ± 0.0059 | 300 s | 1.08e+13 | 1170 s | 2.0 ms | 4.9 |
| Reference solver | reference | reference | reference | reference | - | - | - | 1 s | - |

## Darcy 2-D (budget 300 s on one core per training run)

| Method | Rel. L2, test (3 inst.) | Rel. L2, test (all) | Rel. L2, OOD (2 inst.) | Integral QoI rel. err (3 inst.) | Training (s) | Training FLOPs | Simulation data (core-s) | Time per new instance | Break-even (queries) |
|---|---|---|---|---|---|---|---|---|---|
| Reference solver | reference | reference | reference | reference | - | - | - | 63.1 ms | - |

## Navier-Stokes 2-D (budget 300 s on one core per training run)

| Method | Rel. L2, test (3 inst.) | Rel. L2, test (all) | Rel. L2, OOD (2 inst.) | Integral QoI rel. err (3 inst.) | Training (s) | Training FLOPs | Simulation data (core-s) | Time per new instance | Break-even (queries) |
|---|---|---|---|---|---|---|---|---|---|
| Reference solver | reference | reference | reference | reference | - | - | - | 3 s | - |

## Airfoil 2-D (RANS) (budget 900 s on one core per training run)

OpenFOAM campaign: 436 runs, 90 s per run on one core, 10.9 core-hours in total; status {'unsteady': 32, 'steady-ish': 74, 'converged': 330}.

| Method | Cl abs. err (4 test runs) | Cd rel. err (4 test runs) | Velocity rel. L2 (4 test runs) | Pressure rel. L2 (4 test runs) | Cl abs. err (all test) | Cd rel. err (all test) | Cl abs. err, NACA 2412 α=4° | Cd rel. err, NACA 2412 α=4° | Training (s) | Time per new design | Break-even (queries) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PINN | 0.78 ± 0.099 | 98.7 ± 0.35% | 0.833 ± 0.012 | 1.98 ± 0.009 | - | - | 0.536 ± 0.14 | 98.7 ± 1.4% | 900 s | 900 s | - |
| MeshGraphNet | 0.0334 ± 0.0085 | 1.69 ± 0.32% | 0.155 ± 0.044 | 0.431 ± 0.13 | 0.0319 ± 0.0033 | 2.21 ± 0.51% | 0.026 ± 0.023 | 1.1 ± 0.7% | 901 s | 105.4 ms | 44.7 |
| MLP | 0.00107 ± 0.00031 | 0.295 ± 0.15% | - | - | 0.00227 ± 0.00038 | 0.567 ± 0.21% | 0.000341 ± 1.9e-05 | 0.0377 ± 0.012% | 900 s | 219 µs | 44.7 |
| OpenFOAM (reference) | reference | reference | reference | reference | reference | reference | reference | reference | - | 90 s | - |

## Inverse heat (h) (budget 120 s on one core per training run)

| Method | h rel. err, in range (3 cases) | h rel. err, out of range (2 cases) | Max field err °C, in range | Max field err °C, out of range | Hot-spot err °C, in range | Training (s) | Simulation data (core-s) | Time per new inverse problem | Break-even (queries) |
|---|---|---|---|---|---|---|---|---|---|

Failed runs: 0.
