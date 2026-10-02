"""Bumper-beam Transolver papers: (1) reproduction and (2) improvement -- numbers read from results/."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "paperkit"))
from paperkit import CAT, DIV, INK2, MUTED, SEQ, Paper, fmt, savefig, setup_mpl  # noqa: E402

setup_mpl()
R = json.loads((ROOT / "results" / "results.json").read_text())
FIG = ROOT / "figures"
D = R["data"]
A, B, Cc = R["reproduction"], R["improved"], R["classical"]
H = np.load(ROOT / "results" / "test_node_histories.npz")
Fz = np.load(ROOT / "results" / "test_fields_final.npz")
t = H["t"]
design = np.array(D["design_train"]); dtest = np.array(D["design_test"])

# ---- Fig: DoE
fig, ax = plt.subplots(figsize=(4.4, 3.4))
ax.scatter(design[:, 0], design[:, 1], s=12, color=CAT[0], label="train + validation")
ax.scatter(dtest[:, 0], dtest[:, 1], s=30, color=CAT[1], marker="s", label="held-out test (write-up IDs)")
for (x, y), i in zip(dtest, D["test_ids"]):
    ax.annotate(str(i), (x, y), fontsize=6.5, xytext=(3, 3), textcoords="offset points", color=INK2)
ax.set_xlabel("DV1: DP1000 gauge (mm)"); ax.set_ylabel("DV2: DP600 gauge (mm)"); ax.set_title("100-point Latin hypercube"); ax.legend(fontsize=6.8)
f_doe = savefig(fig, FIG / "fig_doe.png")

# ---- Fig: node 1806 histories, 10 test runs
def hist_fig(keys, fname, title):
    fig, axs = plt.subplots(2, 5, figsize=(8.8, 3.8), sharex=True)
    for j, ax in enumerate(axs.ravel()):
        ax.plot(t, H["truth"][j], color=INK2, lw=1.8, label="OpenRadioss")
        for c, (k, lab) in zip((CAT[1], CAT[0], CAT[2]), keys):
            if k in H.files:
                ax.plot(t, H[k][j], "--", color=c, lw=1.2, label=lab)
        ax.set_title(f"Exp_{int(H['ids'][j])}", fontsize=8); ax.tick_params(labelsize=6.5)
    axs[1, 0].set_xlabel("t (ms)"); axs[0, 0].set_ylabel("u$_x$ node 1806 (mm)"); axs[0, 0].legend(fontsize=5.8)
    fig.suptitle(title, fontsize=9.5)
    return savefig(fig, FIG / fname)


f_repro = hist_fig([("Reproduction", "TransolverLite (reproduction)")], "fig_repro_hist.png", "Reproduction: node 1806 on the 10 held-out runs")
f_impr = hist_fig([("Reproduction", "reproduction"), ("Improved", "PINNeAPPle Transolver ens."), ("POD__Gaussian_process_DV1_DV2", "POD + GP")],
                  "fig_impr_hist.png", "Improvement: node 1806 on the 10 held-out runs")

# ---- Fig: final-frame field error (one test run)
xyz, mat = Fz["xyz"], Fz["mat"]
shell = mat[:, 2] == 0
fig, axs = plt.subplots(1, 3, figsize=(8.8, 3.0))
j = 0
tr, li, im = Fz["truth"][j][:, -1], Fz["lite"][j][:, -1], Fz["improved"][j][:, -1]
v = np.percentile(np.abs(tr[shell]), 99)
for ax, f, lab in zip(axs, (tr, li - tr, im - tr), ("OpenRadioss u$_x$ at 100 ms", "reproduction − truth", "improved − truth")):
    sc = ax.scatter(xyz[shell, 0], xyz[shell, 2], c=f[shell], s=0.6, cmap=DIV, vmin=-v, vmax=v)
    ax.set_title(lab, fontsize=8.5); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.colorbar(sc, ax=axs, shrink=0.8, label="mm")
f_field = savefig(fig, FIG / "fig_field.png")


def score_rows(entries):
    rows = [["Model", "node RMSE (mm)", "node max |err| (mm)", "node R²", "peak err (mm)", "final err (mm)", "field rel. L2", "shell rel. L2"]]
    for lab, s in entries:
        rows.append([lab, fmt(s["node_rmse_mm"]), fmt(s["node_max_abs_mm"]), fmt(s["node_r2"]), fmt(s["node_peak_err_mm"]),
                     fmt(s["node_final_err_mm"]), fmt(s["field_rel_l2"]), fmt(s["shell_rel_l2"])])
    return rows


# ======================================================== paper 1: reproduction
P = Paper("Reproducing a Weekend Transolver for OpenRadioss Bumper-Beam Crash Data with PINNeAPPle",
          "100 OpenRadioss simulations, a Latin-hypercube DoE over two gauges, and a slice-attention neural operator — on a laptop",
          report_id="Experiment 03a · reproduction")
sa = A["test"]
P.abstract(
    f"We reproduce, with the PINNeAPPle toolchain, a public write-up in which a Transformer-based neural operator (a simplified Transolver) "
    f"is trained on OpenRadioss simulations of a bumper beam to predict the displacement time history of every node. We re-ran the full "
    f"pipeline: the OpenRadioss 'Bumper Beam' example, two linked shell gauges as design variables (DP1000 and DP600, ±30% of nominal), a "
    f"100-point Latin hypercube with the same seed and bounds, 100 explicit simulations (100 ms, ANIM output converted to VTK) run in Docker on "
    f"Apple silicon, consolidation into HDF5 with the write-up's schema, and the write-up's architecture and loss (field MSE + 2× the history "
    f"of node 1806). New code needed for this — an OpenRadioss bridge (deck editing, Docker runner, ANIM/VTK reader) and the slice-attention "
    f"model — was added to PINNeAPPle. On the write-up's ten held-out runs the reproduced model reaches a node-1806 RMSE of "
    f"{fmt(sa['node_rmse_mm'])} mm (R² = {fmt(sa['node_r2'])}) and a full-field relative L2 error of {fmt(sa['field_rel_l2'])}. "
    + ("This is consistent with the write-up's claim that the predicted curves track the solver closely."
       if sa["node_r2"] > 0.95 else
       "We therefore do not reproduce the write-up's claim that predictions are 'nearly indistinguishable' from the solver; the discrepancy "
       "and its likely causes are analysed."),
    ["OpenRadioss", "crash simulation", "surrogate model", "Transolver", "neural operator", "design of experiments", "reproducibility", "PINNeAPPle"])
P.box("Reproduction scope", [
    "Reproduced exactly: model, solver, deck, design variables, bounds, LHS seed (SciPy qmc, seed 42), held-out test IDs, network block and loss weighting.",
    "Not specified in the write-up and chosen here: optimiser settings, epochs/early stopping, validation split, displacement component "
    "(u<sub>x</sub>, inferred from the node-1806 curves described), normalisation constants. These are the most likely sources of difference.",
    "Our OpenRadioss build (latest-20260728, aarch64) and scipy version may differ from the write-up's."])
P.section("Pipeline")
P.bullets([
    "Model: OpenRadioss 'Bumper Beam' example (Altair, CC BY-NC 4.0): DP600 crash boxes and DP1000 beam, shell elements, 5 m/s initial velocity "
    "against a rigid pole, 100 ms. The engine deck was changed only to request ANIM displacement output every 1 ms (101 frames).",
    "DoE: DV1 (DP1000, /PROP/SHELL/2 and /7) ∈ [1.26, 2.34] mm and DV2 (DP600, /PROP/SHELL/1 and /6) ∈ [1.54, 2.86] mm, LHS n = 100, seed 42, "
    "via PINNeAPPle <i>sample_parameters</i>. Thickness fields edited in place with a byte-preserving deck editor (verified by diff).",
    f"Solves: <i>pinneapple_simulation.external_solvers.openradioss</i> runs starter, engine and anim_to_vtk in an Ubuntu container, eight in parallel. "
    f"Energy-balance error reported by the solver: median {fmt(D['energy_error_pct_max']['median'], 2)}%, worst {fmt(D['energy_error_pct_max']['max'], 2)}%.",
    f"Data: {D['n_runs']} runs, {D['nodes']:,} nodes, {D['frames']} frames, HDF5 with inputs/Exp_i (nodes, elements, gauges) and outputs/Exp_i "
    "(displacement field, node-1806 history).",
    f"Split: test = the write-up's IDs {', '.join(map(str, D['test_ids']))}; validation = 10 further fixed runs; training = the remaining {len(D['train_ids'])}."])
P.figure(f_doe, "The Latin-hypercube design; squares are the ten held-out test runs of the write-up.", 8.5)
P.section("Model")
P.p("TransolverLite (now in PINNeAPPle as <i>transolver_lite</i>): input projection 5→64, three slice-attention blocks (8 slices, 4 heads, "
    "dropout 0.1), per-node output of 101 values. Node features [x, y, z normalised to [0,1], t<sub>DP600</sub>, t<sub>DP1000</sub>]; targets "
    "standardised with training-set statistics. Loss: MSE over all nodes and frames + 2 × MSE on node 1806. Adam, lr 10<super>−3</super>, "
    f"batch of 2 runs, early stopping on validation. Parameters: {A['info']['n_params']:,}; training {A['info']['train_s'] / 60:.0f} min "
    f"({A['info']['epochs']} epochs) on Apple-silicon MPS; inference {1000 * A['inference_s_per_run']:.0f} ms per run.")
P.section("Results")
P.table(score_rows([("TransolverLite (reproduction)", sa)]), "Held-out test runs (u<sub>x</sub>).", [4.0, 1.7, 1.9, 1.3, 1.6, 1.6, 1.6, 1.6])
P.figure(f_repro, "Node-1806 displacement history on the ten held-out runs: OpenRadioss (solid) vs reproduced model (dashed).", 16)
per = sa["per_run_node_rmse_mm"]
worst = max(per, key=per.get)
P.p(f"Per-run node RMSE ranges from {min(per.values()):.2f} to {max(per.values()):.2f} mm (worst: Exp_{worst}). The write-up singled out Exp_30 "
    f"as its only visible deviation; in our reproduction Exp_30 has {per.get('30', per.get(30, float('nan'))):.2f} mm.")
P.section("Sources of error in the reproduced model")
P.bullets([
    "Scale mismatch in the target: the field contains parts moving by hundreds of millimetres, while node 1806 moves by a few millimetres. With one "
    f"global normalisation (σ = {fmt(D['ux_scale_mm'], 1)} mm) the node-1806 signal is a small fraction of the loss; the 2× weighting does not fix this.",
    "Both gauges are broadcast to every node; the network must infer which nodes belong to which gauge from coordinates alone.",
    "With only 80 training designs and a two-dimensional design space, a high-capacity point-cloud model is data-starved; a classical surrogate on "
    "(DV1, DV2) is a strong baseline (see the companion improvement report)."])
P.references([
    "Chavare, S. Building a ground-up Transolver on OpenRadioss data. Medium (accessed September 2026).",
    "OpenRadioss, https://openradioss.org and https://github.com/OpenRadioss/OpenRadioss (Bumper Beam example model, CC BY-NC 4.0).",
    "Wu, H., Luo, H., Wang, H., Wang, J., Long, M. (2024). Transolver: a fast Transformer solver for PDEs on general geometries. <i>ICML</i>, arXiv:2402.02366.",
    "McKay, M. D., Beckman, R. J., Conover, W. J. (1979). A comparison of three methods for selecting values of input variables. <i>Technometrics</i> 21, 239–245.",
])
P.build(str(ROOT / "paper" / "Bumper_Transolver_Reproduction_PINNeAPPle.pdf"))

# ======================================================== paper 2: improvement
Q = Paper("Improving the Bumper-Beam Crash Surrogate with PINNeAPPle: Physics-Attention Transolver, Node-wise Scaling and Classical Baselines",
          "From a weekend neural operator to a validated, uncertainty-aware surrogate", report_id="Experiment 03b · improvement")
sb, sg, sn = B["test"], Cc["pod_gp"], Cc["nearest"]
best = min((("PINNeAPPle Transolver ensemble", sb), ("POD + Gaussian process", sg)), key=lambda x: x[1]["node_rmse_mm"])
Q.abstract(
    f"Starting from our reproduction of a weekend Transolver for OpenRadioss bumper-beam data (node-1806 RMSE {fmt(sa['node_rmse_mm'])} mm on the "
    f"ten held-out runs), we improve the surrogate with PINNeAPPle: (i) the full Physics-Attention Transolver (per-head slice weights, "
    f"normalised slice tokens, learnable temperature), added natively to the library; (ii) node-wise output scaling, so that small-amplitude "
    f"structural nodes are not drowned by large rigid motions; (iii) per-node physical features (own gauge and material) instead of broadcast "
    f"gauges; (iv) the exact initial condition u(t = 0) = 0 imposed by construction; (v) a three-member deep ensemble for uncertainty. The "
    f"improved model reaches a node-1806 RMSE of {fmt(sb['node_rmse_mm'])} mm (R² = {fmt(sb['node_r2'])}) and a field relative L2 of "
    f"{fmt(sb['field_rel_l2'])}. We also report what the write-up did not: a classical POD + Gaussian-process surrogate on the two design "
    f"variables reaches {fmt(sg['node_rmse_mm'])} mm, and a leave-one-out study over all 100 runs gives its error distribution. The best model on "
    f"this held-out set is the {best[0]}.",
    ["OpenRadioss", "crash surrogate", "Transolver", "physics attention", "Gaussian process", "POD", "uncertainty", "PINNeAPPle", "VeriPhysics"])
Q.box("Main findings", [
    "Node-wise output scaling removes the scale mismatch identified in the reproduction; its effect was not ablated separately from the architecture change.",
    "With two design variables and 100 runs, a classical POD + GP surrogate is a baseline any neural operator must be compared with.",
    "Mesh-native neural operators earn their cost when geometry or topology varies — not in this two-gauge study, where only thicknesses change."])
Q.section("Improvements")
Q.bullets([
    "<b>Physics-Attention Transolver</b> (<i>pinneapple_neural…transolver.Transolver</i>, registered as <i>transolver_native</i>): 4 blocks, width 128, "
    "8 heads × 32 slices, MLP ratio 2; pure PyTorch, no research-only dependency (the existing Noether bridge is licence-restricted).",
    "<b>Node-wise normalisation</b> of u<sub>x</sub> with training-set mean and standard deviation per node.",
    "<b>Features</b>: normalised coordinates, the node's own shell gauge (0 for non-shell/rigid nodes), material one-hot (DP600 / DP1000 / other), both design variables.",
    "<b>Hard initial condition</b>: the first frame is fixed to zero displacement.",
    "<b>Deep ensemble</b> (3 seeds) for predictive uncertainty; AdamW, cosine schedule, early stopping on the same validation runs.",
    "<b>Classical baselines</b>: POD of the training fields (99.99% energy) with a Matérn-ARD Gaussian process per coefficient; nearest design point."])
Q.section("Results")
Q.table(score_rows([("Reproduction (TransolverLite)", sa), ("PINNeAPPle Transolver ensemble", sb), ("POD + Gaussian process", sg),
                    ("Nearest design point", sn)]), "Held-out test runs (write-up IDs), u<sub>x</sub>.", [4.0, 1.7, 1.9, 1.3, 1.6, 1.6, 1.6, 1.6])
Q.figure(f_impr, "Node-1806 history on the ten held-out runs: OpenRadioss vs reproduction, improved ensemble and POD + GP.", 16)
Q.figure(f_field, "u<sub>x</sub> at 100 ms on the shell structure (x–z view) for one test run, and the errors of the two neural models.", 15.5)
if "pod_gp_loo_node_rmse_mm" in Cc:
    lo = Cc["pod_gp_loo_node_rmse_mm"]
    Q.p(f"Leave-one-out over all 100 runs (POD + GP, node 1806): median RMSE {fmt(lo['median'])} mm, 95th percentile {fmt(lo['p95'])} mm, "
        f"maximum {fmt(lo['max'])} mm — a distribution, not one split.")
Q.subsection("Uncertainty and VeriPhysics")
V = R["veriphysics"]
rows = [["Model", "Trust score", "Coverage", "Components"]]
for k, v in V.items():
    if k == "not_run":
        continue
    rows.append([k, fmt(v["overall_score"]), f"{v['coverage']:.1f}", "; ".join(f"{c['name']} {c['score']:.2f}" for c in v["components"])])
Q.table(rows, "PINNeAPPle compute_physics_confidence against the held-out node-1806 histories (registered benchmark). uq_calibration uses the "
              "ensemble spread (Transolver) or the GP posterior (POD + GP). Not run: " + "; ".join(f"{k} ({v})" for k, v in V["not_run"].items()) + ".",
        [5.0, 1.8, 1.6, 7.6])
Q.section("Discussion")
Q.p("The design space here is two-dimensional and the mesh never changes, which is the regime where interpolation in design space (POD + GP) "
    "is hard to beat. A mesh-native operator is justified when the geometry varies between designs or when fields must be predicted for unseen "
    "meshes; that is the natural next experiment, together with more runs near the design-space edges where all models degrade.")
Q.references([
    "Wu, H., et al. (2024). Transolver: a fast Transformer solver for PDEs on general geometries. <i>ICML</i>, arXiv:2402.02366.",
    "Rasmussen, C. E., Williams, C. K. I. (2006). <i>Gaussian Processes for Machine Learning</i>. MIT Press.",
    "Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. <i>NeurIPS</i>.",
    "OpenRadioss, https://openradioss.org (Bumper Beam example, CC BY-NC 4.0).",
    "Chavare, S. Building a ground-up Transolver on OpenRadioss data. Medium (accessed September 2026).",
])
Q.build(str(ROOT / "paper" / "Bumper_Transolver_Improvement_PINNeAPPle.pdf"))
print("ok")
