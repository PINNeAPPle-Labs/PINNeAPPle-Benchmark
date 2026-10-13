# 09 — When PINNs, when data-driven surrogates: an equal-budget benchmark

PINN vs FNO vs DeepONet (plus MeshGraphNet, MLP and POD + MLP where they apply) on five problems, every method with
**the same compute budget, the same search effort, the same hardware and the same seeds**, built on
[PINNeAPPle](https://github.com/PINNeAPPle-Labs/PINNeAPPle) and its benchmark API (`pp.benchmark`). It answers
[PINNeAPPle issue #299](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/299).

![Summary](figures/fig0_summary_table.png)

> **CPU-only budgets.** Everything ran on one 4-core Intel Xeon @ 2.10 GHz without a GPU, one core per training run:
> **300 s** per run (Burgers, Darcy, Navier–Stokes), **900 s** (airfoil), **120 s** (inverse heat). These are small
> budgets; the error-versus-budget curves show which methods were still improving at the end.

## Headline

<!-- HEADLINE -->

## Problems

| Problem | Equation and setting | Input → output | Reference solver (library) | OOD set |
|---|---|---|---|---|
| Burgers 1-D | u_t + (u²/2)_x = ν u_xx, periodic [0, 1), ν = 0.01, T = 1 | u(·, 0) → u(·, 1), 256 points | `operator_benchmarks.burgers_1d` (pseudo-spectral, ETDRK4, 1024 points, dt 1e-4) | initial amplitude ×2 (σ = 50) |
| Darcy 2-D | −∇·(a∇u) = 1, u = 0 on ∂[0, 1]², a ∈ {3, 12} | a → u, 64² | `operator_benchmarks.darcy_2d` (finite volumes, 128², pooled to 64²) | finer coefficient (τ = 6); see Limitations |
| Navier–Stokes 2-D | vorticity form, periodic unit square, ν = 1e-3, Li et al. forcing, T = 10 | ω(·, 0) → ω(·, 10), 64² | `operator_benchmarks.navier_stokes_2d_vorticity` (pseudo-spectral, ETDRK4, dt 5e-3) | smaller-scale initial vorticity (τ = 14) |
| Airfoil 2-D | RANS, k-ω SST, Re 4·10⁶, −2° to 14°, CST shapes | shape + α → Cl, Cd, fields | OpenFOAM v1912 simpleFoam (the aero app's 436-run campaign) | 12 cases: thick (NACA 0021, 2421, 4418) and high-camber (6409, 6412) sections outside the CST box, and α = −4° outside the angle range |
| Inverse heat | thin plate, 8 thermocouples, 0.5 °C noise | readings → h | finite volumes (`heat_fv_reference.plate_2d`) | h = 150, 200 W/m²K (outside the surrogates' 5–100) |

Input distributions follow Li et al. (2021), "Fourier Neural Operator for Parametric Partial Differential Equations",
ICLR ([arXiv:2010.08895](https://arxiv.org/abs/2010.08895)): Gaussian random fields with covariance
σ²(−Δ + τ²I)^(−α) (`periodic_grf`); α = 2, τ = 5, σ = 25 (Burgers); α = 2, τ = 3 thresholded to 12/3 (Darcy);
α = 2.5, τ = 7 (Navier–Stokes). All data are generated here with the library's own solvers; no external dataset is
used. Burgers uses ν = 0.01 rather than Li et al.'s ν = 0.1: on the unit interval ν = 0.1 decays the solution to
about 1.4 % of its initial amplitude by t = 1 (`results/verification.json`: max |u(0)| = 1.069, max |u(1)| = 0.0146),
which makes the operator nearly trivial.

## Methods

| Method | Library entry point | Trained on | Notes |
|---|---|---|---|
| FNO | `pinneapple_neural.architectures.neural_operators.fno` (`FourierNeuralOperator`, `FNO2d`) | simulation data | 4 layers, grid channels, relative-L2 loss, pointwise normalisation |
| DeepONet | `pinneapple_neural.architectures.neural_operators.deeponet.DeepONet` | simulation data | branch = the input field at every grid point, trunk = coordinates |
| PINN | `pinneapple_physics.pinn_solver.factory.pinn_factory` (`PINNFactory`, `PINN`) | the equations only, one instance at a time | hard initial/boundary conditions (`FourierInterpolant` for gridded initial conditions), Adam |
| MeshGraphNet | `pinneapple_design.aero.surrogate.AeroGNN` (on `pinneapple_neural`'s `MeshGraphNet`) | the OpenFOAM runs | fields + pooled Cl, Cd, Cm head (the aero app's model) |
| MLP (integrals only) | `pinneapple_design.aero.surrogate.MLPEnsemble` | the OpenFOAM runs | 5 bootstrapped MLPs on shape + α → Cl, Cd, Cm |
| POD + MLP (fields) | `pp.benchmark` built-in `mlp_fields` | the OpenFOAM runs | POD of (p, Ux, Uy) on the graph nodes, MLP from shape + α to the POD coefficients |
| PINN, potential flow (airfoil) | `PINNFactory` with Γ and the wall stream value as inverse parameters | the equations only, per geometry | Kutta condition on the trailing-edge bisector |
| PINN, RANS (airfoil) | `PINNFactory`, incompressible RANS with a mixing-length closure | the equations only, per geometry | van Driest damping, eddy viscosity frozen per step; hard no-slip |
| PINN (inverse heat) | `pinneapple_analysis.inverse_problems.heat_transfer.run_plate(budget=...)` | readings + equations, per case | h as an inverse parameter |
| Surrogate + calibration | MLP / DeepONet + golden-section search of h | finite-volume runs | the same readings as the PINN for each seed |
| FV + calibration (reference) | golden-section search through `plate_2d` itself | — | the classical approach |

## Protocol (fixed before any run)

**Hardware.** One 4-core Intel Xeon @ 2.10 GHz VM, 15 GB RAM, no GPU, torch 2.14.1 (CPU), Python 3.13 (the record
from `hardware_record()` is in every run file and in `results/hardware.json`). Every training run uses one CPU core
(`torch.set_num_threads(1)`); four runs execute concurrently whatever the method, so every method sees the same
contention.

**Budget per training run** (wall clock on one core, the primary budget): 300 s for Burgers, Darcy and
Navier–Stokes, 900 s for the airfoil, 120 s for the inverse heat problem. The loop is
`pinneapple_analysis.cost.train_within_budget`: it stops when the budget is spent, drives a cosine learning-rate
schedule by the fraction of the budget used (so a shorter budget is not a truncated longer run), and pauses the clock
during evaluation. FLOPs are counted on the first step with `count_flops` (PyTorch's per-operator formulas plus FFTs;
elementwise operations are not counted) and reported with the wall clock.

**Equal-FLOPs arm.** For the three standard problems every method is also trained to the same FLOP count: the fewest
any method used in its full wall-clock budget (median over seeds), with the wall clock capped at 3× the budget.

**Hyper-parameter search, equal for every method.** `equal_budget_search` draws 4 configurations from each method's
grid (the `space_*` of each built-in problem in `pinneapple_analysis.benchmark.builtin`), each trained for 1/3 of the
full budget with seed 0 and scored on **validation data only** (operators: the validation split; PINNs: one
validation instance that is not a test instance). The best configuration is then trained with seeds 0, 1, 2 at 1/9,
1/3 and 1× the budget (the error-versus-budget curves).

**Data-generation cost, counted separately.** The CPU time of every reference solve is measured
(`time.process_time()`, one thread) and summed over the training and validation sets. The OpenFOAM campaign is costed
at its clean per-run time (56 s per design, `results/openfoam_clean_timing.json`: 4 runs re-timed alone) because the
campaign's recorded wall times were inflated by sharing the CPU. The amortised cost per new query is
`AmortizedCost`: an operator pays the data and its training once and then one forward pass per query; a per-instance
PINN pays its training for every new query. The break-even column is the number of queries above which the operator
is cheaper in total.

**Test sets.** Standard problems: 200 held-out test inputs and 100 out-of-distribution inputs. Per-instance PINNs
are trained on test instances 0, 1, 2 and OOD instances 0, 1 only; the tables report every method on **those same
instances** (n = 3 and n = 2) next to the operators on the whole sets (n = 200 and n = 100). Airfoil: the aero app's
split (15 % of the shapes held out with all their angles, 60 runs), the four NACA polars as an independent set, and
the 12 OOD cases; the PINNs on 4 held-out runs (2 shapes × 2 angles), NACA 2412 at 4° and 4 of the OOD cases.
Inverse heat: 3 cases inside the surrogates' training range and 2 outside it.

**Metrics.** Relative L2 of the output field per instance; an integral quantity (Burgers: energy ∫u², Darcy: ∫u, the
compliance for f = 1; Navier–Stokes: enstrophy ∫ω²); Cl absolute and Cd relative error, velocity and pressure
relative L2 on the near-field graph nodes (airfoil); h relative error, maximum temperature error and hot-spot error
(inverse heat). Inference time per new instance is the median of repeated batch-1 calls on one core. Cells are the
mean ± sample standard deviation over 3 seeds of the per-seed mean over the column's n instances.

**Correctness gates and validation (library).** Before training, every (problem, method) pair passes
`pp.benchmark`'s architecture gates (shapes, FNO modes against the Nyquist limit and resolution invariance, DeepONet
shapes, MeshGraphNet permutation equivariance, PINN autograd against finite differences and the PDE residual of an
exact solution, overfitting a tiny batch); every run is checked afterwards (finite and decreasing loss, budget
enforced, identical accounting, evaluation off the clock, search on validation data only, seed spread, a determinism
spot-check). The prepared artifacts are hashed and validated with the library's validation stack (solver
verification, grid convergence, data sanity and leakage, applicability envelope, physics audit, mesh quality and y+
for the airfoil). Results: `results/table.md` (sections Validation, Trust card, Checks) and `results/validation/`.

## Results

All numbers below are in `results/table.md` / `results/table.csv` (the library's report) and come from the run files
in `results/runs/`.

### Burgers 1-D (300 s per run)

| Method | Rel. L2, test (n = 3) | Rel. L2, test, all (n = 200) | Rel. L2, OOD (n = 2) | Rel. L2, OOD, all (n = 100) | Energy rel. err (n = 3) | Time per new instance |
|---|---|---|---|---|---|---|
| PINN | 0.246 ± 0.081 | – | 0.592 ± 0.051 | – | 0.118 ± 0.024 | 300 s (training) |
| FNO | **0.00168 ± 0.00034** | 0.00328 ± 0.0003 | **0.0393 ± 0.0017** | 0.0199 ± 0.0015 | 0.00193 ± 0.0011 | 1.7 ms |
| DeepONet | 0.0155 ± 0.002 | 0.0416 ± 0.0013 | 0.228 ± 0.033 | 0.142 ± 0.0018 | 0.0109 ± 0.0059 | 2.0 ms |

### Darcy 2-D (300 s per run)

| Method | Rel. L2, test (n = 3) | Rel. L2, test, all (n = 200) | Rel. L2, OOD (n = 2) | Rel. L2, OOD, all (n = 100) | ∫u rel. err (n = 3) | Time per new instance |
|---|---|---|---|---|---|---|
| PINN | 0.873 ± 0.0041 | – | 0.891 ± 0.003 | – | 0.863 ± 0.004 | 300 s (training) |
| FNO | **0.0167 ± 0.00049** | 0.0153 ± 0.00041 | **0.0138 ± 0.0017** | 0.0172 ± 0.00053 | 0.00963 ± 0.00061 | 12.6 ms |
| DeepONet | 0.057 ± 0.0032 | 0.0577 ± 0.00082 | 0.0602 ± 0.0032 | 0.0611 ± 0.00051 | 0.0241 ± 0.0048 | 5.0 ms |

### Navier–Stokes 2-D (300 s per run)

| Method | Rel. L2, test (n = 3) | Rel. L2, test, all (n = 200) | Rel. L2, OOD (n = 2) | Rel. L2, OOD, all (n = 100) | Enstrophy rel. err (n = 3) | Time per new instance |
|---|---|---|---|---|---|---|
| PINN | 0.16 ± 0.0056 | – | 0.507 ± 0.0053 | – | 0.248 ± 0.0087 | 300 s (training) |
| FNO | **0.0068 ± 0.00089** | 0.00844 ± 0.0012 | **0.0107 ± 0.0012** | 0.0137 ± 0.0013 | 0.00197 ± 0.00097 | 15.3 ms |
| DeepONet | 0.0307 ± 0.0051 | 0.0315 ± 0.00054 | 0.0419 ± 0.0085 | 0.0417 ± 0.0019 | 0.0115 ± 0.0057 | 7.1 ms |

### Equal FLOPs (the same training FLOPs for every method)

Rel. L2 on test instance 0 (the one the PINN was trained on, n = 1) and, for the operators, on the whole test set
(n = 200); mean ± std over 3 seeds.

| Problem (FLOPs per run) | PINN | FNO | DeepONet |
|---|---|---|---|
| Burgers (2.16e12) | 0.252 ± 0.086 | **0.00186 ± 0.00013** (all: 0.00336) | 0.058 ± 0.0061 (all: 0.0561) |
| Darcy (2.10e12) | 0.846 ± 0.013 | **0.0139 ± 0.00052** (all: 0.0153) | 0.0521 ± 0.0017 (all: 0.066) |
| Navier–Stokes (2.20e12) | 0.304 ± 0.017 | **0.0111 ± 0.0021** (all: 0.00845) | 0.0524 ± 0.0019 (all: 0.041) |

The FLOP budget is the fewest FLOPs any method used in its 300 s (the PINN's on Burgers, the FNO's on Darcy and
Navier–Stokes). DeepONet executes 4–7× more counted FLOPs per second than the FNO here, so at equal FLOPs it trains
for only 44–60 s: 3.7× worse than at equal wall clock on Burgers, 1.7× on Navier–Stokes, about the same on Darcy
(0.052 vs 0.057). The FNO gets 281–400 s and stays within a factor 1.6 of
its wall-clock error (better on Darcy, worse on Navier–Stokes). The Navier–Stokes PINN gets 201 s instead of 300 s and
its error doubles (0.304 vs 0.16). The ranking (FNO, then DeepONet, then PINN) is the same as at equal wall clock on
all three problems.

### Cost: data, training, break-even

| Problem | Simulation data (core-s, train + val) | Reference solver per instance | Operator break-even vs the PINN (queries) |
|---|---|---|---|
| Burgers | 1170 | 1 s | 4.9 |
| Darcy | 69 | 63 ms | 1.2 |
| Navier–Stokes | 3843 | 3 s | 13.8 |
| Inverse heat | 14 | 9 s (calibration through the FV model) | 1.1 |

The break-even counts the operator's data **and** training against the PINN's training per new instance. Above about
5, 1 and 14 queries the operator is cheaper in total on these three problems, and it is also 1–2 orders of magnitude
more accurate. Against the classical solver itself, the PINN is not competitive on any of them in these budgets: the
reference solver is faster (63 ms to 3 s) and exact to its discretisation.

<!-- AIRFOIL -->

### Inverse heat (h from 8 sensors, 120 s per run)

| Method | h rel. err, in range (n = 3) | h rel. err, out of range (n = 2) | Hot-spot err °C (n = 3) | Max field err °C, in range (n = 3) | Time per new case |
|---|---|---|---|---|---|
| PINN | 11.4 % ± 5.3 % | 39.3 % ± 1.3 % | 0.24 ± 0.13 | 1.1 ± 0.31 | 120 s (training) |
| DeepONet surrogate + calibration | 1.01 % ± 0.29 % | **2.06 % ± 1.2 %** | 0.129 ± 0.014 | 0.524 ± 0.1 | 620 ms |
| MLP surrogate + calibration | 1.01 % ± 0.63 % | 4.48 % ± 2.9 % | 0.189 ± 0.09 | 0.235 ± 0.096 | 111 ms |
| FV model + calibration (reference) | **0.946 % ± 0.55 %** | 3.31 % ± 2 % | 0.166 ± 0.097 | 0.168 ± 0.098 | 9 s |

In range, both surrogates match the classical calibration through the finite-volume model itself (about 1 % in h,
the level the 0.5 °C sensor noise allows), at 1–7 % of its cost per case; outside their training range they stay
within a few per cent. The PINN, which must learn the field and h
for every case in 120 s, is an order of magnitude worse and was still improving at the full budget (0.233 at 1/3 of
the budget, 0.114 at the full budget).

## Where each method wins and where it loses

<!-- WINS -->

## What the library's gates and advisor said

<!-- DIAGNOSTICS -->

## Limitations

<!-- LIMITATIONS -->

## Reproduce

```bash
bash reproduce.sh            # everything: OpenFOAM campaign, data, runs, library prepare/validation/gates, report
EXP09_SMOKE=1 bash reproduce.sh   # tiny sizes and budgets: checks the pipeline in minutes
```

It needs the PINNeAPPle library checked out next to this repository at the commit in `env/PINNEAPPLE_COMMIT`, the
Python packages in `env/requirements.lock`, and OpenFOAM v1912 (Ubuntu 24.04: `apt-get install openfoam`) for the
airfoil. Runs are cached in `results/runs/`: an interrupted run resumes, and finished jobs are not repeated.
The library part alone, on the same artifacts:

```bash
cd src && python bench09.py prepare && python bench09.py run    # results/table.md, table.csv, summary.json, figures/
```

## Files

| Path | Content |
|---|---|
| `src/bench09.py` | this experiment as a `pp.benchmark` configuration (prepare, run, report) |
| `src/run_all.py`, `src/jobs.py`, `src/methods_std.py`, `src/airfoil.py`, `src/inverse_heat.py` | the job runner and trainers used for the runs (the library's built-ins were moved from them) |
| `src/gen_data.py`, `src/gen_airfoil_ood.py` | data generation with measured cost |
| `src/report.py`, `src/report_airfoil.py` | this experiment's figures and airfoil tables |
| `results/runs/` | one JSON per training job (spec, result, hardware, wall time) |
| `results/table.md`, `table.csv`, `summary.json` | the library's report |
| `results/prepared/`, `results/validation/` | artifact manifests (hashes, cost), checks and validation |
| `figures/` | the figures shown here |
| `env/` | environment lock and the library commit |

## References

- Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar (2021). Fourier Neural Operator for
  Parametric Partial Differential Equations. ICLR. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895).
- Lu, Jin, Pang, Zhang, Karniadakis (2021). Learning nonlinear operators via DeepONet based on the universal
  approximation theorem of operators. Nature Machine Intelligence 3, 218–229.
- Krishnapriyan, Gholami, Zhe, Kirby, Mahoney (2021). Characterizing possible failure modes in physics-informed
  neural networks. NeurIPS. [arXiv:2109.01050](https://arxiv.org/abs/2109.01050).
- Wang, Teng, Perdikaris (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural
  networks. SIAM Journal on Scientific Computing 43(5). [doi:10.1137/20M1318043](https://doi.org/10.1137/20M1318043).
- Eivazi, Tahani, Schlatter, Vinuesa (2022). Physics-informed neural networks for solving Reynolds-averaged
  Navier–Stokes equations. Physics of Fluids 34, 075117. [doi:10.1063/5.0095270](https://doi.org/10.1063/5.0095270).
