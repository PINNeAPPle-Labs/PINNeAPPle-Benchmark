## Protocol (fixed before any run)

**Hardware.** One 4-core Intel Xeon @ 2.10 GHz VM, 15 GB RAM, **no GPU** (the exact line from
`hardware_record()` is in every run file). Every training run uses **one CPU core**
(`torch.set_num_threads(1)`); four runs execute concurrently, whatever the method, so every method sees the same
contention. Budgets are therefore small; they are stated in every table header.

**Budget per training run** (wall clock on one core, the primary budget): 300 s for Burgers, Darcy and
Navier–Stokes, 900 s for the airfoil, 120 s for the inverse heat problem. The training loop is
`pinneapple_analysis.cost.train_within_budget`: it stops when the budget is spent, drives a cosine learning-rate
schedule by the fraction of the budget used (so a shorter budget is not a truncated longer run), and pauses the
clock during evaluation. FLOPs are counted on the first step with `count_flops` (PyTorch's per-operator formulas
plus FFTs; elementwise operations are not counted) and reported with the wall clock.

**Equal-FLOPs arm.** For the three standard problems every method is also trained to the same FLOP count: the
smallest any method used in its full wall-clock budget (median over seeds), capped at 3x the wall budget.

**Hyper-parameter search, equal for every method.** `equal_budget_search` draws 4 configurations from each
method's grid (`SPACES` in `src/methods_std.py`, `src/airfoil.py`, `src/inverse_heat.py`), each trained for 1/3 of
the full budget with seed 0, scored on **validation data only** (operators: the validation split; PINNs: one
validation instance that is not a test instance). The best configuration is then trained with seeds 0, 1, 2 at
1/9, 1/3 and 1 x the budget (the error-versus-budget curves).

**Data-generation cost, counted separately.** The CPU time of every reference solve is measured
(`time.process_time()`, one thread) and summed over the training and validation sets. The OpenFOAM campaign
records the wall time of each run. The amortised cost per new query is `AmortizedCost`: an operator pays the data
and its training once and then one forward pass per query; a per-instance PINN pays its training for every new
query. The break-even column is the number of queries above which the operator is cheaper in total.

**Test sets.** Standard problems: 200 held-out test inputs and 100 out-of-distribution inputs (below). Per-instance
PINNs are trained on test instances 0, 1, 2 (and OOD instances 0, 1) only; the table reports the operators on
**those same instances** next to their error on the whole test set. Airfoil: the aero app's split (15 % of the
shapes held out with all their angles, 60 runs), the four NACA polars as an independent set; the PINN on 4
held-out runs (2 shapes x 2 angles) and NACA 2412 at 4°. Inverse heat: 3 cases inside the surrogates' training
range and 2 outside it.

**Metrics.** Relative L2 of the output field per instance; an integral quantity (Burgers: energy ∫u², Darcy: ∫u,
the compliance for f = 1, Navier–Stokes: enstrophy ∫ω²); Cl absolute and Cd relative error, velocity and pressure
relative L2 on the near-field graph nodes (airfoil); h relative error, maximum temperature error and hot-spot error
(inverse heat). Inference time per new instance is the median of repeated batch-1 calls on one core. Mean ±
sample standard deviation over 3 seeds.

## Problems

| Problem | Equation and setting | Input → output | Reference solver (library) | OOD set |
|---|---|---|---|---|
| Burgers 1-D | u_t + (u²/2)_x = ν u_xx, periodic [0, 1), ν = 0.01, T = 1 | u(·, 0) → u(·, 1), 256 points | `operator_benchmarks.burgers_1d` (pseudo-spectral, ETDRK4, 1024 points, dt 1e-4) | initial amplitude x2 (σ = 50) |
| Darcy 2-D | −∇·(a∇u) = 1, u = 0 on ∂[0, 1]², a ∈ {3, 12} | a → u, 64² | `operator_benchmarks.darcy_2d` (finite volumes, 128², pooled to 64²) | finer coefficient (τ = 6) |
| Navier–Stokes 2-D | vorticity form, periodic unit square, ν = 1e-3, Li et al. forcing, T = 10 | ω(·, 0) → ω(·, 10), 64² | `operator_benchmarks.navier_stokes_2d_vorticity` (pseudo-spectral, ETDRK4, dt 5e-3) | smaller-scale initial vorticity (τ = 14) |
| Airfoil 2-D | RANS, k-ω SST, Re 4·10⁶, −2° to 14°, CST shapes | shape + α → Cl, Cd, fields | OpenFOAM v1912 simpleFoam (`apps/aero_optimizer/tools/generate_dataset.py`) | NACA 2412/4412/0012/4415 |
| Inverse heat | thin plate, 8 thermocouples, 0.5 °C noise | readings → h | finite volumes (`heat_fv_reference.plate_2d`) | h = 150, 200 W/m²K (outside the surrogates' 5–100) |

Input distributions follow Li et al. (2021) ("Fourier Neural Operator for Parametric PDEs", ICLR): GRFs with
covariance σ²(−Δ + τ²I)^(−α) (`periodic_grf`), α = 2, τ = 5, σ = 25 (Burgers), α = 2, τ = 3 thresholded to 12/3
(Darcy), α = 2.5, τ = 7 (Navier–Stokes). The data are generated here with the library's own solvers; no external
dataset is used. Burgers uses ν = 0.01 (as in PDEBench's Burgers setting) rather than Li et al.'s ν = 0.1 because on
the unit interval ν = 0.1 decays the solution to about 1.4 % of its initial amplitude by t = 1
(`results/verification.json`: max |u(0)| and max |u(1)|), which makes the operator nearly trivial.

## Methods

| Method | Library entry point | Trained on | Notes |
|---|---|---|---|
| FNO | `pinneapple_neural.architectures.neural_operators.fno.FourierNeuralOperator` / `FNO2d` | simulation data | 4 layers, grid channels, relative-L2 loss, pointwise normalisation |
| DeepONet | `pinneapple_neural.architectures.neural_operators.deeponet.DeepONet` | simulation data | branch = the input field at every grid point, trunk = coordinates |
| PINN | `pinneapple_physics.pinn_solver.factory.pinn_factory.PINNFactory`, `PINN`, `NeuralNetwork` | the equations only, one instance at a time | hard initial/boundary conditions (`FourierInterpolant` for gridded initial conditions), periodic input features, Adam |
| MeshGraphNet | `pinneapple_design.aero.surrogate.AeroGNN` (on `pinneapple_neural` `MeshGraphNet`) | the OpenFOAM runs | fields + pooled Cl, Cd, Cm head (the aero app's model) |
| MLP | `pinneapple_design.aero.surrogate.MLPEnsemble`; `pinneapple_neural ... deeponet.mlp` (heat) | simulation data | airfoil: 5 bootstrapped MLPs on shape + α |
| PINN (airfoil) | `PINNFactory` with an inverse parameter | the equations only, per geometry | incompressible potential flow; see below |
| PINN (inverse heat) | `pinneapple_analysis.inverse_problems.heat_transfer.run_plate(budget=...)` | readings + equations, per case | the Inverse Heat Lab's PINN, h as an inverse parameter |
| Surrogate + calibration | MLP / DeepONet + golden-section search of h | finite-volume runs | readings identical to the PINN's for each seed |
| FV + calibration | golden-section search through `plate_2d` itself | — | the classical reference |
