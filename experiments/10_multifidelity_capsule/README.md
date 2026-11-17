# 10 — What the cheap model buys: a multi-fidelity learning curve on an aerothermodynamics corpus

How many expensive runs a cheap model makes unnecessary, measured on
[capsule-aeroheating](https://huggingface.co/datasets/bijective/capsule-aeroheating) (Bijective Systems, CC BY 4.0):
1.024 parametric entry capsules with the laminar forebody heat flux from a modified-Newtonian panel method — an
engineering model that costs seconds — and 64 of them also with a second-order laminar Navier–Stokes solution from
Eilmer, which cost a median of 1.9 h each on six cores (1.2 h to 5.4 h). Mach 6–15, altitude 40–80 km, diameter 0.8 m
to 5.5 m, zero incidence, a 300 K isothermal wall, calorically perfect air.

What makes this corpus a multi-fidelity benchmark rather than two datasets: every wall has the same 2.361 nodes and
4.640 triangles, and node *i* is the same point of the heat shield on every capsule and at both fidelities. The
difference between the fidelities is therefore a subtraction at each node, with no interpolation between meshes — a
property no other public CFD corpus we read has.

## Protocol

**Data.** One file, `vehicle_capsule_multifidelity.gamut` (77 MB). The 64 high-fidelity runs are a subset of the
1.024 members; the low-fidelity run of the same member supplies the cheap value at every expensive design. No design is
interpolated to: a cheap value that is not in the file is refused rather than taken from the nearest member.

**Inputs.** The six design and flight variables, scaled to `[-1, 1]` by the bounds the file declares
(`diameter` m, `nose_radius` and `shoulder_radius` as fractions of the diameter, `forebody_angle` in degrees, `mach`,
`altitude` in km). Scaling by the declared design space rather than by the sample's range is what keeps a 5.5 m capsule
from dominating a kernel.

**Splits.** For a budget `N < 64`, `N` of the expensive runs are drawn at random (20 draws, one seed) and the score is
on the remaining `64 - N`. At `N = 64` there is nothing to hold out, so each run is predicted from the other 63
(leave-one-out) — a different question, and the only honest way to put the full budget on the same axis. Every
strategy of a draw is scored on the same held-out runs, so comparisons between rows are paired.

**Metrics.** Relative L2 against the high-fidelity value, per target and per run, then the median over the held-out
runs and draws; R² is recorded next to it. For the field, the relative L2 is over a run's whole 2.361-node field, so a
run contributes one number.

**Regressor.** Kernel ridge with an RBF kernel, in closed form: no optimiser state, no epochs, and it behaves at `N = 8`
where a network has nothing to learn from. The length scale is the median pairwise distance of the training designs.
One seed, 20 draws; the spread over draws is reported next to every median.

## Strategies

| per capsule | what it is |
|---|---|
| `low_fidelity` | the cheap model's own value. No fit. The baseline a correction has to beat |
| `geometry` | a surrogate of the target from the six inputs alone — "what if there is no cheap model" |
| `difference` | a surrogate of `y_high - y_low` from the six inputs: cheap model everywhere plus a learned correction (the two-fidelity co-kriging model) |
| `residual_feature` | the same difference, with the cheap model's value among the inputs |

| per field | what it is |
|---|---|
| `low_fidelity` | the cheap field on the same nodes |
| `mean_correction` | the cheap field plus the average *relative* residual of the training runs: one shape, nothing learned from the design |
| `pod_correction` | the cheap field plus `k = 3` modes of the relative residual with amplitudes learned from the six inputs (POD + regression on its coefficients) |

The relative residual divides each run's residual by the magnitude of its own cheap field, and the prediction multiplies
back by the cheap run's magnitude — legitimate at any design, because the cheap model runs everywhere. This is not
cosmetic: the heat flux spans 175× across designs, so the raw decomposition's coefficients span 175× too, and a
regression on six inputs cannot learn a quantity whose scale the design sets. The first version of this case, without
it, reported 96 % and 5494 % error.

## Headline

The cheap model as it stands, and what a correction fitted on `N` Navier–Stokes runs does to it. Relative L2, median
over 20 draws (spread over draws in the JSON):

| target | cheap model | corrected, N=8 | corrected, N=64 | design only, N=64 |
|---|---|---|---|---|
| heat load | 5.85 % | 5.04 % | 3.55 % | 54.86 % |
| stagnation heat flux | 18.28 % | 17.87 % | 14.60 % | 26.85 % |
| **shoulder heat flux** | **27.43 %** | **16.41 %** | **7.01 %** | 25.55 % |

| the heat-flux field, 2.361 nodes | relative L2 per run |
|---|---|
| cheap field | 17.91 % |
| + average correction, nothing learned | **8.94 %** |
| + 3 modes, amplitudes learned, N=8 | 5.88 % |
| + 3 modes, amplitudes learned, N=64 | **3.57 %** |

The energy of the relative residual sits **81.0 %** in one mode, **92.4 %** in two and **97.4 %** in three.

## What the numbers say

* **The cheap model is worth less where it is already good.** On the heat load it is 5.85 % off, and eight CFD runs buy
  0.8 points. On the shoulder it is 27.4 % off, and eight runs buy 11 points. A study that reports one number for "the
  multi-fidelity gain" is hiding this.
* **The shoulder is where the panel method is structurally wrong** — it cannot see how the shoulder changes the flow
  over the face — and it is where the correction earns its cost.
* **The field's error is a shape, not a table.** One mode carries 81 % of it, and the *average* correction with no
  design dependence at all halves the field error (17.9 % → 8.9 %). What the design buys on top of that is the
  amplitude, and that is what the eight runs pay for.
* **The design alone never gets close** (25 % to 55 %): knowing the diameter, the nose radius and the Mach number does
  not tell you the heating. This is the measurement behind an engineering model existing at all.

## Limitations, stated

* **64 expensive runs is a small sample.** The 20 draws give the spread of the *split*, not an uncertainty of the
  corpus: the high-fidelity values themselves carry no error bars, and the corpus says its grids were not refined case
  by case.
* **The field curve is low rank.** It is not a comparison against a network free to choose every node's correction.
  That is a choice, not a claim of superiority: with 64 fields over 2.361 nodes such a model is fitting noise, and the
  mode structure above says how little there is to gain.
* **The scalar targets here are three summaries of the same field.** They are correlated; they are not three
  independent problems.
* **No architecture search, no hyperparameter tuning.** One regressor, one kernel, one length-scale heuristic, so the
  rows compare *strategies*, not models. A comparison of architectures is experiment 09's job.
* **No extrapolation test.** Every strategy is scored inside the sampled design box. The corpus has a sparse corner
  (the largest capsules near Mach 15 and 80 km could not be run), so an extrapolation split would be measuring the
  corpus's gap as much as the model.
* **The cheap model is the corpus's own panel method.** Its own biases are the thing being corrected, not an
  independent implementation.

## Reproduce

```bash
bash reproduce.sh            # downloads 77 MB, then ~2 min on a laptop CPU
bash reproduce.sh smoke      # 16 expensive runs, 2 draws: a pipeline check in seconds
```

Writes `results/curve.json` (every median and spread, plus the protocol: seed, draws, budgets, channel scaling,
residual energy shares). The case needs the PINNeAPPle checkout named in `env/PINNEAPPLE_COMMIT` next to this
repository (or `$PINNEAPPLE_REPO`), and `pip install bijective-gamut` for the `.gamut` reader.

The same two curves, with the same numbers and one seed, are also an example in the library
(`examples/capsule_aeroheating/`), which is where the reader and the regression live:
`pinneapple_data.datasets.capsule_aeroheating` and `pinneapple_data.multifidelity`.

## Licence and citation

Data and figures CC BY 4.0; the reader (`bijective-gamut`) Apache-2.0; this case Apache-2.0 like the repository.

```bibtex
@misc{bijective_capsule_aeroheating_2026_v4,
  title        = {Capsule forebody aeroheating, Newtonian and Navier--Stokes {CFD}},
  author       = {{Bijective Systems}},
  year         = {2026},
  version      = {4},
  howpublished = {\url{https://huggingface.co/datasets/bijective/capsule-aeroheating}}
}
```
