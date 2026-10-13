# FAILED_APPROACHES — PINNeAPPle-Benchmark

Nothing failed in the creation/population of this repo itself (it is new; the move that populated it
merged cleanly on the first try). The failures relevant to its *content* — bugs found and fixed inside the
three moved experiments — are documented where they happened, in
`PINNeAPPle-Climate/.agent/FAILED_APPROACHES.md` (that file covers the same code, since it moved here
unchanged; not duplicated here to avoid two copies drifting apart).

If real failures happen in *this* repo specifically going forward (a bad merge, a benchmark protocol
attempt that didn't pan out, a data-inclusion decision that had to be reverted), record them here.

## Experiment 09 (equal-budget PINN vs FNO vs DeepONet vs MeshGraphNet), 2026-10-13

1. **Operators with Adam `weight_decay=1e-4`.** The first search stage used it for FNO and DeepONet on the three
   standard problems. On Navier-Stokes every FNO configuration sat at the mean prediction (validation rel. L2
   0.185, identical to three digits); the same FNO without weight decay reached 0.010 in 240 s (four-variant check:
   no decay 0.010, MSE loss 0.018, no grid channels 0.208, with decay 0.185). Coupled L2 in Adam swamps the small
   gradients of the FNO's spectral weights (initial scale 1/(width^2)). Fix: `OPERATOR_WD = 0.0` for both operators on
   all three problems and their whole search re-run. The first attempt's 24 search runs are kept in
   `experiments/09_equal_budget_pinn_fno_deeponet/results/first_attempt_wd1e-4/`.
2. **Airfoil potential-flow PINN, three formulations before the one used.** All on NACA 2412 at 4 deg
   (OpenFOAM Cl 0.66), checked before any reported run:
   - raw (x, y) inputs over a 20-chord disc and a Kutta condition matching |u| on both surfaces: Cl 0.017 after
     300 s. Two faults: tanh saturation on +-20 inputs, and |u| equality admits the zero-circulation flow (equal
     speeds, opposite directions);
   - bounded inputs and a vector Kutta condition: Gamma moved only with a separate learning rate (Cl 0.22 at 150 s,
     0.38 at 600 s); the network met the condition with a local distortion near the trailing edge;
   - wall condition made exact with a distance function: velocities evaluated on the wall are 0/eps there
     (gradient of the distance undefined), so the Kutta points on the wall were trivially satisfied; Gamma stayed 0;
   - used: soft wall condition (psi = C, C an inverse parameter) and the Kutta condition as a value condition,
     psi = C on 8 points of the trailing-edge bisector (the dividing streamline leaves at the trailing edge):
     Cl 0.43 at 300 s. The wall-pressure integral of the PINN does not converge in these budgets (blunt-TE corner
     singularities), so Cl is reported from Kutta-Joukowski (2 Gamma).
3. **The library's correctness gates, first pass on this experiment's artifacts** (`pp.benchmark`, before
   publishing). What they caught and what changed:
   - *Darcy OOD set is not out of distribution*: the data gate (PCA reconstruction error and Mahalanobis distance
     of the coefficient fields, against the training split) put only 9 % of the OOD inputs outside the training
     envelope. A finer GRF length scale (tau 6 vs 3), once thresholded to the same two values {3, 12}, is not
     distinguishable from the training inputs by these features. Not regenerated (the runs are already done with it):
     reported as a limitation, and the Darcy OOD column is read as "a second test set", not as extrapolation.
   - *GCI of Burgers energy and Navier-Stokes enstrophy reported "oscillatory, the GCI formula does not apply"* and
     the first version of the gate counted that as a pass. The three grids agree to better than 1e-3 there, so the
     gate now passes on the spread itself and records an oscillation with a real spread as a warning.
   - *DeepONet "overfits a tiny batch" blocked Darcy and Navier-Stokes* (loss to 13 % and 22 % of its initial value
     in 1000 steps; threshold 10 %), although the same DeepONet trains to 0.057 and 0.031 relative L2. A model that
     cannot overfit three samples is a broken setup only when it cannot reduce the loss at all: the gate now errors
     at >= 50 % and warns between 10 % and 50 %.
   - *"Did not learn" (loss not halved) flagged the 1/9-budget runs and the search runs* as errors. Those are short by
     design; the check now applies to the full-budget final and equal-FLOPs runs, and a loss that halves too slowly is
     a warning (the error item is "did not converge": an error no better than a trivial prediction).
4. **Airfoil RANS PINN collapses to the zero-flow solution** (found in the 1 h budget-study runs, 2026-10-13). After
   3600 s (2277 Adam steps) on NACA 2412 at 4 deg its velocity on the graph nodes is ~0.01 everywhere (true mean
   0.71): velocity rel. L2 1.07, pressure 1.00, Cl from the wall 0.00 (OpenFOAM 0.66). The checkpoint of the 4 h run at
   2.1 h is the same (mean Ux 0.013). Not a units bug: OpenFOAM's fields are U_inf = 1 and kinematic gauge pressure,
   as the PINN's. u = v = 0, p = const satisfies the RANS equations exactly, and only the far-field penalty (256 points
   at r = 19) opposes it. Two fixes tried, 300 s each on NACA 2412 at 4 deg (velocity rel. L2):
   - far field exact (u = U_inf + (1 - r^2/19^2) N): 0.98 (lr 3e-3), 0.96 (lr 1e-3), 1.47 (lr 1e-2, diverged); PDE
     loss falls to 1e-3, so the network finds a slowly varying near-solution instead, in the outer flow that the
     wall-clustered O-grid collocation barely samples;
   - plus half the collocation points even in log r over 0.55 < r < 19: 0.98 (lr 3e-3), 0.97 (lr 1e-3); 80 % of them:
     1.02; with the soft far field: 0.96. Without either: 0.96.
   Neither changes the outcome, so the reported runs keep the original formulation; both are options in the library
   (`airfoil_impl.RANS_FAR`, `RANS_MIX`, or cfg `far`, `colloc_mix`), off by default. Literature on the same failure:
   Krishnapriyan et al. 2021 (arXiv:2109.01050), Wang, Teng & Perdikaris 2021 (doi:10.1137/20M1318043); the RANS PINNs
   that work (Eivazi et al. 2022, doi:10.1063/5.0095270) use data on the boundaries, which this benchmark's data-free
   PINN does not have.
