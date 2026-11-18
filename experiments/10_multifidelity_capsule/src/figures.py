"""The figures of case 10, drawn from the corpus and from results/curve.json -- never from a table typed by hand.

    python src/figures.py            # writes figures/fig1..fig5 and rewrites nothing else

Every panel is computed here: the design space of the 1.024 capsules, the two fidelities along one heat shield, the
shape of the cheap model's error and its first modes, and the two learning curves. If a number in the report and a
number in a figure disagree, the figure is right and the report is stale.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
ROOT = CASE.parent.parent
REPO = os.environ.get("PINNEAPPLE_REPO") or str(ROOT / "PINNEAPPLE")
sys.path.insert(0, REPO)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pinneapple_data.datasets import capsule_aeroheating as cap
from pinneapple_data.multifidelity import (
    FidelityPair,
    FieldPair,
    field_learning_curve,
    learning_curve,
    residual_rank,
)

FIGURES = CASE / "figures"
TARGETS = ("heat_load", "stagnation_heat_flux", "shoulder_heat_flux")
LABELS = {"heat_load": "heat load (W)", "stagnation_heat_flux": "stagnation heat flux (W m$^{-2}$)",
          "shoulder_heat_flux": "shoulder heat flux (W m$^{-2}$)"}
GREY = "#8a8a8a"
BLUE = "#1f4e79"
RED = "#c0392b"
ORANGE = "#e08214"
GREEN = "#2e7d32"


def _style() -> None:
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "legend.fontsize": 8})


def _save(fig, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")
    return str(path)


# ---------------------------------------------------------------------------
# the dataset
# ---------------------------------------------------------------------------

def fig_wall_3d(study) -> str:
    """One capsule's heat shield, coloured by each fidelity -- the corpus's own answer to "what is the data"."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    high, low = cap.pair_fidelities(study)
    expensive = cap.wall(study, int(high[0]))
    cheap = cap.wall(study, int(low[0]))
    peak = max(expensive.fields["heat_flux"].max(), cheap.fields["heat_flux"].max()) / 1e3
    cmap = plt.get_cmap("inferno")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=peak)
    fig = plt.figure(figsize=(9.6, 3.8))
    for i, (one, title) in enumerate(((expensive, "Navier-Stokes (Eilmer)"), (cheap, "panel method (Newtonian)"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        triangles = one.points[one.cells]
        per_face = one.fields["heat_flux"][one.cells].mean(axis=1) / 1e3
        collection = Poly3DCollection(triangles, facecolors=cmap(norm(per_face)), linewidths=0, antialiased=False)
        ax.add_collection3d(collection)
        bounds = np.array([one.points.min(axis=0), one.points.max(axis=0)])
        ax.set_xlim(bounds[0, 0], bounds[1, 0])
        ax.set_ylim(bounds[0, 1], bounds[1, 1])
        ax.set_zlim(bounds[0, 2], bounds[1, 2])
        ax.set_box_aspect((bounds[1] - bounds[0]) / np.max(bounds[1] - bounds[0]))
        ax.set_title(f"{title}\n{one.member}: D = {one.parameters['diameter']:.2f} m, "
                     f"M = {one.parameters['mach']:.1f}, h = {one.parameters['altitude']:.0f} km", fontsize=9)
        ax.set_xlabel("x (m)", labelpad=-6)
        ax.set_ylabel("y (m)", labelpad=-8)
        ax.set_zlabel("z (m)", labelpad=-8)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.zaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.tick_params(pad=-2, labelsize=7)
        ax.view_init(elev=14, azim=-72)
        ax.text2D(0.0, 0.04, f"peak {per_face.max():.0f} kW m$^{{-2}}$", transform=ax.transAxes, fontsize=8,
                  color="0.25")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=fig.axes, fraction=0.025, pad=0.02)
    bar.set_label("$q$ (kW m$^{-2}$)")
    fig.suptitle("Figure 1 -- one wall, two fidelities: the corpus's 64 Navier-Stokes runs and 1.024 panel runs share "
                 "these 2.361 nodes", y=1.03, fontsize=10)
    return _save(fig, "fig1_two_fidelities.png")


def fig_design_space(study) -> str:
    """The 1.024 capsules in the six inputs, with the 64 that have a CFD solution marked."""
    table = study.table(["diameter", "mach", "altitude", "forebody_angle", "nose_radius", "shoulder_radius",
                         "fidelity", "member"])
    fidelity = np.asarray(table["fidelity"], dtype=str)
    high, _ = cap.pair_fidelities(study)
    is_high = np.zeros(fidelity.size, dtype=bool)
    is_high[high] = True
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    panels = (("diameter", "mach", "diameter (m)", "Mach"),
              ("forebody_angle", "altitude", "forebody half-angle (deg)", "altitude (km)"),
              ("nose_radius", "shoulder_radius", "nose radius (x D)", "shoulder radius (x D)"))
    for ax, (xname, yname, xlabel, ylabel) in zip(axes, panels, strict=True):
        x = np.asarray(table[xname], dtype=float)
        y = np.asarray(table[yname], dtype=float)
        ax.scatter(x[~is_high], y[~is_high], s=6, c=GREY, alpha=0.45, linewidths=0, label="panel method (1024)")
        ax.scatter(x[is_high], y[is_high], s=22, c=RED, alpha=0.9, linewidths=0, label="+ Navier-Stokes (64)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="lower right")
    fig.suptitle("Figure 2 -- the design space: 1.024 Latin-hypercube capsules, 64 of them with a CFD solution",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    return _save(fig, "fig2_design_space.png")


def fig_cheap_against_cfd(study) -> str:
    """Three capsules along the heat shield: the two fidelities on the same nodes, the shoulder shaded."""
    high, low = cap.pair_fidelities(study)
    picks = [0, len(high) // 3, len(high) // 2]
    fig, axes = plt.subplots(len(picks), 1, figsize=(7.4, 6.4), sharex=True)
    for ax, k in zip(axes, picks, strict=True):
        expensive = cap.wall(study, int(high[k]))
        cheap = cap.wall(study, int(low[k]))
        s = expensive.fields["arc_length"]
        order = np.argsort(s)
        q_high = expensive.fields["heat_flux"][order] / 1e3
        q_low = cheap.fields["heat_flux"][order] / 1e3
        delta = q_low - q_high
        ax.axvspan(0.85 * s.max(), s.max(), color="0.92", zorder=0)     # the shoulder
        ax.plot(s[order], q_high, color="0.15", lw=1.5, label="Navier-Stokes")
        ax.plot(s[order], q_low, color=BLUE, lw=1.2, ls="--", label="panel method")
        ax.set_ylabel("$q$ (kW m$^{-2}$)")
        drop = 100 * (np.linalg.norm(delta) / np.linalg.norm(q_high))
        ax.set_title(f"{expensive.member}: D = {expensive.parameters['diameter']:.2f} m, "
                     f"M = {expensive.parameters['mach']:.1f}, h = {expensive.parameters['altitude']:.0f} km "
                     f"-- cheap model {drop:.1f}% off (relative L2 over the wall)", loc="left")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("arc length from the stagnation point (m)   --   shoulder shaded")
    fig.suptitle("Figure 3 -- the two fidelities on the same nodes, node for node, no interpolation", y=1.0,
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, "fig3_cheap_against_cfd.png")


def fig_error_shape(study) -> dict:
    """What the cheap model's error looks like: the mean relative residual along the arc, and its first modes."""
    pair = cap.field_pair(study)
    relative = pair.residual() / pair.run_scale().reshape(-1, 1)
    arc = np.asarray(pair.position, dtype=float)
    order = np.argsort(arc)
    rank = residual_rank(relative, components=3)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for k, colour in zip(range(3), (BLUE, ORANGE, GREEN), strict=True):
        mode = rank["modes"][k][order]
        ax.plot(arc[order], mode / np.abs(mode).max(), color=colour, lw=1.3, label=f"mode {k + 1} "
                f"({rank['singular'][k] ** 2 / (rank['singular'] ** 2).sum():.0%} of the energy)")
    ax.set_xlabel("arc length from the stagnation point (m)")
    ax.set_ylabel("mode, scaled to its peak")
    ax.set_title("The error's shape: modes of the relative residual", loc="left")
    ax.legend(loc="lower left")
    ax = axes[1]
    energy = rank["singular"] ** 2
    cumulative = np.cumsum(energy) / energy.sum()
    ax.bar(np.arange(1, 11), 100 * energy[:10] / energy.sum(), color=BLUE, alpha=0.85, label="this mode")
    ax.plot(np.arange(1, 11), 100 * cumulative[:10], color=RED, marker="o", ms=3, lw=1.2, label="cumulative")
    ax.set_xlabel("mode")
    ax.set_ylabel("share of the residual energy (%)")
    ax.set_title(f"Deep, not wide: {100 * cumulative[2]:.1f}% in three modes", loc="left")
    ax.legend(loc="lower right")
    fig.suptitle("Figure 4 -- the cheap model is wrong the same way on every capsule; only the size depends on the "
                 "design", y=1.02, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig4_error_shape.png")
    return {"cumulative": cumulative[:3].tolist()}


# ---------------------------------------------------------------------------
# the outputs
# ---------------------------------------------------------------------------

def fig_scalar_curves(pairs) -> dict:
    """The learning curve per target: relative L2 against the number of expensive runs."""
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4), sharey=False)
    summary = {}
    for ax, target in zip(axes, TARGETS, strict=True):
        pair = pairs[target]
        curve = learning_curve(pair, budgets=(8, 16, 32, 64), repeats=20, seed=0)
        budgets = np.array(curve.budgets)
        for name, colour, style, label in (("low_fidelity", GREY, ":", "cheap model, no fit"),
                                           ("geometry", RED, "-.", "design only"),
                                           ("difference", BLUE, "-", "cheap + learned correction"),
                                           ("residual_feature", GREEN, "--", "correction reading the cheap value")):
            median = np.median(curve.scores[name]["relative_l2"], axis=1)
            spread = np.ptp(curve.scores[name]["relative_l2"], axis=1)
            ax.plot(budgets, 100 * median, style, color=colour, marker="o", ms=3.5, lw=1.3, label=label)
            if name in ("difference", "geometry"):
                ax.fill_between(budgets, 100 * (median - spread / 2), 100 * (median + spread / 2), color=colour,
                                alpha=0.12, linewidth=0)
        ax.set_xlabel("expensive (Navier-Stokes) runs, N")
        ax.set_ylabel("relative L2 against the CFD (%)")
        ax.set_title(LABELS[target], loc="left")
        if target == TARGETS[0]:
            ax.legend(loc="upper right")
        summary[target] = float(np.median(curve.scores["difference"]["relative_l2"][0]))
    fig.suptitle("Figure 5 -- how many CFD runs the cheap model buys, per target (median over 20 draws; band = "
                 "spread over draws)", y=1.03, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig5_scalar_curves.png")
    return summary


def fig_field_curve(field: FieldPair, members: list[str]) -> dict:
    """The field curve, and what the corrections actually do to a held-out heat shield."""
    curve = field_learning_curve(field, budgets=(8, 16, 32, 64), repeats=20, seed=0, components=3)
    fig = plt.figure(figsize=(10.4, 3.6))
    ax = fig.add_subplot(1, 2, 1)
    budgets = np.array(curve.budgets)
    for name, colour, style, label in (("low_fidelity", GREY, ":", "cheap field"),
                                       ("mean_correction", ORANGE, "--", "+ average correction (nothing learned)"),
                                       ("pod_correction", BLUE, "-", "+ 3 modes, amplitudes learned")):
        median = np.median(curve.scores[name]["relative_l2"], axis=1)
        ax.plot(budgets, 100 * median, style, color=colour, marker="o", ms=3.5, lw=1.3, label=label)
    ax.set_xlabel("expensive runs, N")
    ax.set_ylabel("relative L2 per run over 2.361 nodes (%)")
    ax.set_title("The field's learning curve", loc="left")
    ax.legend(loc="upper right")

    # one held-out capsule: cheap, the average correction, and three modes, against the CFD
    ax = fig.add_subplot(1, 2, 2)
    relative = field.residual() / field.run_scale().reshape(-1, 1)
    modes = residual_rank(relative, components=3)["modes"]
    coefficients = relative @ modes.T
    run = int(np.argmax(np.linalg.norm(relative, axis=1)))     # the capsule where the cheap model is worst
    arc = np.asarray(field.position, dtype=float)
    order = np.argsort(arc)
    scale = field.run_scale()[run]
    mean_correction = relative.mean(axis=0) * scale
    pod_correction = (coefficients[run] @ modes) * scale
    truth = field.values_high[run] / 1e3
    cheap = field.values_low[run] / 1e3
    ax.plot(arc[order], truth[order], color="0.15", lw=1.8, label="Navier-Stokes")
    ax.plot(arc[order], cheap[order], color=GREY, lw=1.2, ls=":", label="cheap model")
    ax.plot(arc[order], (cheap + mean_correction / 1e3)[order], color=ORANGE, lw=1.2, ls="--",
            label="+ average correction")
    ax.plot(arc[order], (cheap + pod_correction / 1e3)[order], color=BLUE, lw=1.4, label="+ 3 modes (own amplitudes)")
    ax.set_xlabel("arc length from the stagnation point (m)")
    ax.set_ylabel("$q$ (kW m$^{-2}$)")
    ax.set_title(f"{members[run]}: the correction along the heat shield", loc="left")
    ax.legend(loc="upper right")
    fig.suptitle("Figure 6 -- the same question asked of the whole field", y=1.03, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig6_field_curve.png")
    return {"cheap": float(np.median(curve.scores["low_fidelity"]["relative_l2"][0])),
            "pod_at_8": float(np.median(curve.scores["pod_correction"]["relative_l2"][0])),
            "pod_at_64": float(np.median(curve.scores["pod_correction"]["relative_l2"][-1])),
            "mean_correction": float(np.median(curve.scores["mean_correction"]["relative_l2"][0]))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(CASE / "data"))
    args = parser.parse_args()
    _style()
    path = cap.download("wall", args.data)
    with cap.open_study(path) as study:
        print("figures:")
        fig_wall_3d(study)
        fig_design_space(study)
        fig_cheap_against_cfd(study)
        shape = fig_error_shape(study)
        high, low = cap.pair_fidelities(study)
        _, channels, _ = cap.field_channels(study)
        table = study.table(list(TARGETS))
        pairs = {target: FidelityPair(design_low=channels[low], values_low=np.asarray(table[target])[low],
                                      design_high=channels[high], values_high=np.asarray(table[target])[high],
                                      low_at_high=np.asarray(table[target])[low])
                 for target in TARGETS}
        scalars = fig_scalar_curves(pairs)
        members = [cap.wall(study, int(row)).member for row in high]
        fields = fig_field_curve(cap.field_pair(study), members)
    print("\nnumbers behind the figures:")
    print(f"  residual energy in 1/2/3 modes: {[round(100 * v, 1) for v in shape['cumulative']]}")
    print(f"  correction at N=8 per target:  { {k: round(100 * v, 2) for k, v in scalars.items()} }")
    print(f"  field: cheap {100 * fields['cheap']:.2f}%, mean correction {100 * fields['mean_correction']:.2f}%, "
          f"three modes {100 * fields['pod_at_8']:.2f}% at N=8 and {100 * fields['pod_at_64']:.2f}% at N=64")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
