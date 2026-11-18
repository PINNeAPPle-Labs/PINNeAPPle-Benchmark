"""Experiment 10: the multi-fidelity learning curve on the capsule-aeroheating corpus, one command.

What it measures, and what it deliberately does not
---------------------------------------------------
The corpus gives two fidelities of the same wall: 1.024 entry capsules with the laminar forebody heat flux from a
Newtonian panel method (seconds per run), and 64 of those also from second-order laminar Navier-Stokes in Eilmer
(1.2 h to 5.4 h on six cores). Node *i* is the same point of the heat shield on every capsule and at both fidelities,
so the difference between the fidelities is a subtraction, not an interpolation.

The benchmark is a **learning curve**: fit on N of the expensive runs, score on the ones held out, for four strategies
per scalar target and three for the field. The point is not to crown a model -- it is to price the cheap model, which
is the thing a study of this kind actually decides.

Everything is deterministic in the splits (one seed, `repeats` draws); the regressor is kernel ridge in closed form, so
there is no optimiser state and no epoch budget to tune. That is a choice with a cost stated in the README: this case
does not compare neural architectures, and the field curve is a low-rank model rather than a network per node.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
ROOT = CASE.parent.parent                    # the benchmark repository
REPO = os.environ.get("PINNEAPPLE_REPO") or str(ROOT / "PINNEAPPle")
sys.path.insert(0, REPO)

from pinneapple_data.datasets import capsule_aeroheating as cap
from pinneapple_data.multifidelity import (
    FidelityPair,
    FieldPair,
    field_learning_curve,
    learning_curve,
    residual_rank,
)

TARGETS = ("heat_load", "stagnation_heat_flux", "shoulder_heat_flux")
SCALAR_BUDGETS = (8, 16, 32, 64)
FIELD_BUDGETS = (8, 16, 32, 64)
MODES = 3
REPEATS = 20
SEED = 0


def load(study) -> dict:
    """The two pairs, from the file -- the only place the dataset's layout is read."""
    high, low = cap.pair_fidelities(study)
    _, channels, scaling = cap.field_channels(study)
    fidelity = np.asarray(study.table(["fidelity"])["fidelity"], dtype=str)
    cheap = np.flatnonzero(fidelity == cap.LOW_FIDELITY)
    table = study.table(list(TARGETS))
    scalars = {target: FidelityPair(design_low=channels[cheap], values_low=np.asarray(table[target])[cheap],
                                    design_high=channels[high], values_high=np.asarray(table[target])[high],
                                    low_at_high=np.asarray(table[target])[low])
               for target in TARGETS}
    field = FieldPair(design_high=channels[high], values_high=np.asarray(
        [cap.wall(study, int(row)).fields["heat_flux"] for row in high]),
        values_low=np.asarray([cap.wall(study, int(row)).fields["heat_flux"] for row in low]),
        position=np.asarray(cap.wall(study, int(high[0])).fields["arc_length"]))
    return {"scalars": scalars, "field": field, "scaling": scaling,
            "n_cheap": int(cheap.size), "n_expensive": int(high.size)}


def curve_as_dict(curve) -> dict:
    return {"budgets": list(curve.budgets), "repeats": curve.repeats,
            "strategies": {name: {"relative_l2_median": np.median(values["relative_l2"], axis=1).tolist(),
                                  "relative_l2_spread": np.ptp(values["relative_l2"], axis=1).tolist(),
                                  "r2_median": np.median(values["r2"], axis=1).tolist()}
                           for name, values in curve.scores.items()}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(CASE / "data"), help="where the 77 MB wall file is kept")
    parser.add_argument("--smoke", action="store_true", help="16 expensive runs, 2 repeats, small budgets")
    parser.add_argument("--out", default=str(CASE / "results" / "curve.json"))
    args = parser.parse_args()

    path = cap.download("wall", args.data)
    with cap.open_study(path) as study:
        loaded = load(study)
    field = loaded["field"]
    scalars = loaded["scalars"]
    if args.smoke:
        keep = 16
        scalars = {t: FidelityPair(design_low=p.design_low, values_low=p.values_low, design_high=p.design_high[:keep],
                                   values_high=p.values_high[:keep], low_at_high=p.low_at_high[:keep])
                   for t, p in scalars.items()}
        field = FieldPair(design_high=field.design_high[:keep], values_high=field.values_high[:keep],
                          values_low=field.values_low[:keep], position=field.position)
    scalar_budgets = (4, 8, 16) if args.smoke else SCALAR_BUDGETS
    repeats = 2 if args.smoke else REPEATS

    result = {"case": "10_multifidelity_capsule", "seed": SEED, "repeats": repeats,
              "dataset": {"id": cap.HUGGINGFACE, "file": cap.FILES["wall"], "licence": cap.LICENSE,
                          "n_cheap_runs": loaded["n_cheap"], "n_expensive_runs": loaded["n_expensive"],
                          "nodes_per_run": field.n_nodes, "cite": cap.CITATION},
              "inputs": {"channels": list(cap.PARAMETERS), "scaling": loaded["scaling"]},
              "scalars": {}, "field": {}}
    for target, pair in scalars.items():
        curve = learning_curve(pair, budgets=scalar_budgets, repeats=repeats, seed=SEED)
        result["scalars"][target] = {"cheap_model_error": float(np.linalg.norm(pair.low_at_high - pair.values_high)
                                                                / np.linalg.norm(pair.values_high)),
                                     **curve_as_dict(curve)}
        print(f"=== {target}\n{curve.table()}\n")

    relative = field.residual() / field.run_scale().reshape(-1, 1)
    curve = field_learning_curve(field, budgets=(4, 8, 16) if args.smoke else FIELD_BUDGETS, repeats=repeats,
                                 seed=SEED, components=MODES)
    result["field"] = {"nodes": field.n_nodes, "components": MODES,
                       "residual_energy_share": {str(k): float(residual_rank(relative, components=k)["share"])
                                                 for k in range(1, MODES + 1)},
                       **curve_as_dict(curve)}
    print(f"=== heat_flux field\n{curve.table()}\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
