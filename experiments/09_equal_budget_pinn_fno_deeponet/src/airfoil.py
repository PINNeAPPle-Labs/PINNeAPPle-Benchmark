"""The aero app's 2-D airfoil (issue #299, case 1): MeshGraphNet and MLP trained on the 436 OpenFOAM runs against a
PINN trained per geometry, under the same compute budget.

Data: apps/aero_optimizer/tools/generate_dataset.py (OpenFOAM v1912 simpleFoam, k-omega SST, Re 4e6, the app's
O-grid), re-run here; the split is the app's (tools/train_surrogate.py ``split``: 15 % of the LHS shapes held out
with all their angles; the four NACA airfoils as a second, independent test). For the search, 10 of the training
shapes are held out as validation (the same for every method).

Methods:
* MeshGraphNet: pinneapple_design.aero.surrogate.AeroGNN (pinneapple_neural MeshGraphNet on the near-field cells,
  fields p, Ux, Uy, nut and a pooled Cl, Cd, Cm head), loss as the app's.
* MLP: pinneapple_design.aero.surrogate.MLPEnsemble (5 bootstrapped MLPs on shape + alpha -> Cl, log Cd, Cm).
* PINN (per geometry, no data): the equations a PINN can be given here without a turbulence model or simulation
  data: incompressible potential flow. Stream function psi = U(y cos a - x sin a) + Gamma/(2 pi) ln r + (1 - r^2/R^2)
  N(x, y); lap psi = 0 (PINNFactory residual), psi = C on the wall (C trainable, PINNFactory condition with an
  inverse parameter), Kutta condition |u| equal on both surfaces at x = 0.995 (extra loss), Gamma trainable.
  Cl = 2 Gamma (Kutta-Joukowski); Cd from the wall-pressure integral (zero in exact potential flow, d'Alembert).
"""
from __future__ import annotations

import math
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from common import DATA, LIB
from pinneapple_analysis.cost import ComputeBudget, cosine_lr, set_lr, time_per_query, train_within_budget
from pinneapple_design.aero.geometry import surfaces, wall_loop
from pinneapple_design.aero.mesh import GridSpec, _distances, cell_centres, grid_points
from pinneapple_design.aero.surrogate import (FIELDS, AeroGNN, Layout, MLPEnsemble, coef_from_targets, coef_targets,
                                              edge_features, node_features)
from pinneapple_physics.pinn_solver.factory.pinn_factory import PINN, NeuralNetwork, PINNFactory, PINNProblemSpec

sys.path.insert(0, str(LIB / "apps" / "aero_optimizer" / "tools"))
from train_surrogate import load_runs, split  # noqa: E402

AIR = DATA / "airfoil"
SPACES = {
    "mlp": {"width": [64, 96, 128], "lr": [1e-3, 3e-3], "wd": [1e-6, 1e-4]},
    "mgn": {"hidden": [32, 48], "mp": [6, 8], "lr": [5e-4, 1e-3]},
    "pinn": {"width": [64, 128], "depth": [4, 6], "lr": [1e-3, 3e-3], "n_colloc": [2048, 4096]},
}
R_FAR = 20.0


def splits():
    recs = load_runs([str(AIR)])
    tr, te, ref = split(recs)
    shapes = sorted({r["shape_id"] for r in tr})
    rng = np.random.default_rng(5)
    val_shapes = set(rng.choice(shapes, 10, replace=False))
    return {"train": [r for r in tr if r["shape_id"] not in val_shapes],
            "val": [r for r in tr if r["shape_id"] in val_shapes], "train_all": tr, "test": te, "naca": ref}


def coefs(recs):
    return np.array([[r["Cl"], r["Cd"], r["Cm"]] for r in recs])


def coef_scores(true: np.ndarray, pred: np.ndarray) -> Dict:
    return {"cl_abs_err": np.abs(pred[:, 0] - true[:, 0]).tolist(),
            "cd_rel_err": (np.abs(pred[:, 1] - true[:, 1]) / true[:, 1]).tolist(),
            "cm_abs_err": np.abs(pred[:, 2] - true[:, 2]).tolist(),
            "cl_abs_err_mean": float(np.abs(pred[:, 0] - true[:, 0]).mean()),
            "cd_rel_err_mean": float((np.abs(pred[:, 1] - true[:, 1]) / true[:, 1]).mean()),
            "cm_abs_err_mean": float(np.abs(pred[:, 2] - true[:, 2]).mean())}


def field_scores(true: np.ndarray, pred: np.ndarray) -> Dict:
    """Relative L2 per run of the velocity (Ux, Uy together) and of the pressure, on the graph nodes."""
    u = np.linalg.norm((pred[..., 1:3] - true[..., 1:3]).reshape(len(true), -1), axis=1) / np.linalg.norm(
        true[..., 1:3].reshape(len(true), -1), axis=1)
    p = np.linalg.norm(pred[..., 0] - true[..., 0], axis=1) / np.linalg.norm(true[..., 0], axis=1)
    return {"u_rel_l2": u.tolist(), "p_rel_l2": p.tolist(), "u_rel_l2_mean": float(u.mean()),
            "p_rel_l2_mean": float(p.mean())}


# ------------------------------------------------------------------------------------------------- MLP
def train_mlp(cfg: Dict, seed: int, budget: ComputeBudget, *, final: bool) -> Dict:
    S = splits()
    tr = S["train_all"] if final else S["train"]
    X = MLPEnsemble.inputs(np.array([r["shape"] for r in tr]), np.array([r["alpha"] for r in tr]))
    Y = coef_targets(*coefs(tr).T)
    mean, std = Y.mean(0), Y.std(0)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(((Y - mean) / std).astype(np.float32))
    torch.manual_seed(seed)
    model = MLPEnsemble(5, cfg["width"])
    rng = np.random.default_rng(seed)
    boots = [torch.from_numpy(rng.integers(0, len(X), len(X))) for _ in range(5)]
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    w = torch.tensor([1.0, 2.0, 1.0])

    def step(progress):
        set_lr(opt, cosine_lr(progress, cfg["lr"]))
        loss = sum(((net(Xt[b]) - Yt[b]) ** 2 * w).mean() for net, b in zip(model.nets, boots))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    run = train_within_budget(step, budget)

    def predict(recs):
        Xq = torch.from_numpy(MLPEnsemble.inputs(np.array([r["shape"] for r in recs]), np.array([r["alpha"] for r in recs])))
        with torch.no_grad():
            y = model(Xq).numpy() * std + mean
        cl, cd, cm = coef_from_targets(y)
        return np.stack([cl.mean(0), cd.mean(0), cm.mean(0)], 1)

    out = {"train": run.to_dict(), "n_params": sum(p.numel() for p in model.parameters()),
           "n_train_runs": len(tr)}
    for name in (("test", "naca") if final else ("val",)):
        out[name] = coef_scores(coefs(S[name]), predict(S[name]))
        out[name]["ids"] = [r["id"] for r in S[name]]
    x1 = torch.from_numpy(MLPEnsemble.inputs(np.array([S["val"][0]["shape"]]), np.array([S["val"][0]["alpha"]])))
    out["infer_batch1"] = time_per_query(lambda: model(x1), repeats=50, warmup=5)
    return out


# ------------------------------------------------------------------------------------------------- MeshGraphNet
def graph_data(lay, recs):
    X, E, Yf, Yc = [], [], [], []
    for r in recs:
        s = np.array(r["shape"])
        pos = lay.positions(s)
        X.append(node_features(lay, s, r["alpha"], pos))
        E.append(edge_features(lay, pos))
        z = np.load(r["npz"])
        nu = 1.0 / r["reynolds"]
        Yf.append(np.stack([z["p"], z["Ux"], z["Uy"], np.log1p(z["nut"] / nu)], 1)[lay.cell_index])
        Yc.append(coef_targets(r["Cl"], r["Cd"], r["Cm"]))
    return np.stack(X), np.stack(E), np.stack(Yf).astype(np.float32), np.stack(Yc).astype(np.float32)


def train_mgn(cfg: Dict, seed: int, budget: ComputeBudget, *, final: bool, batch: int = 4) -> Dict:
    S = splits()
    tr = S["train_all"] if final else S["train"]
    lay = Layout()
    X, E, Yf, Yc = graph_data(lay, tr)
    fm, fs = Yf.reshape(-1, 4).mean(0), Yf.reshape(-1, 4).std(0)
    cm_, cs = Yc.mean(0), Yc.std(0)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))      # noqa: E731
    Xt, Et, Yfn, Ycn = T(X), T(E), T((Yf - fm) / fs), T((Yc - cm_) / cs)
    torch.manual_seed(seed)
    model = AeroGNN(X.shape[-1], hidden=cfg["hidden"], mp=cfg["mp"])
    ei, wall = torch.from_numpy(lay.edge_index), torch.arange(len(lay.i_sel))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(seed)
    cw = torch.tensor([1.0, 2.0, 1.0])
    state = {"perm": rng.permutation(len(X)), "pos": 0}

    def step(progress):
        set_lr(opt, cosine_lr(progress, cfg["lr"]))
        if state["pos"] + batch > len(X):
            state["perm"], state["pos"] = rng.permutation(len(X)), 0
        idx = torch.from_numpy(state["perm"][state["pos"]:state["pos"] + batch])
        state["pos"] += batch
        y, c = model(Xt[idx], ei, Et[idx], wall)
        loss = ((y - Yfn[idx]) ** 2).mean() + (((c - Ycn[idx]) ** 2) * cw).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    model.train()
    run = train_within_budget(step, budget)
    model.eval()
    out = {"train": run.to_dict(), "n_params": sum(p.numel() for p in model.parameters()), "n_train_runs": len(tr)}
    for name in (("test", "naca") if final else ("val",)):
        Xq, Eq, Yfq, _ = graph_data(lay, S[name])
        C, F = [], []
        with torch.no_grad():
            for b in range(0, len(Xq), 8):
                y, c = model(T(Xq[b:b + 8]), ei, T(Eq[b:b + 8]), wall)
                C.append(c.numpy() * cs + cm_)
                F.append(y.numpy() * fs + fm)
        cl, cd, cm = coef_from_targets(np.concatenate(C))
        out[name] = coef_scores(coefs(S[name]), np.stack([cl, cd, cm], 1))
        out[name].update(field_scores(Yfq, np.concatenate(F)))
        out[name]["ids"] = [r["id"] for r in S[name]]
    r0 = S["val"][0]
    s0 = np.array(r0["shape"])

    def one_design():                                             # features from the shape, then the network
        pos = lay.positions(s0)
        x = torch.from_numpy(node_features(lay, s0, r0["alpha"], pos))[None]
        e = torch.from_numpy(edge_features(lay, pos))[None]
        return model(x, ei, e, wall)
    out["infer_batch1"] = time_per_query(one_design, repeats=20, warmup=3)
    return out


# ------------------------------------------------------------------------------------------------- PINN
class PotentialNet(nn.Module):
    def __init__(self, cfg: Dict, alpha_deg: float):
        super().__init__()
        self.mlp = NeuralNetwork(num_inputs=3, num_outputs=1, num_layers=cfg["depth"], num_neurons=cfg["width"],
                                 activation=nn.Tanh())
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.a = math.radians(alpha_deg)

    def forward(self, X):
        x, y = X[:, :1], X[:, 1:2]
        r2 = (x - 0.25) ** 2 + y ** 2
        base = y * math.cos(self.a) - x * math.sin(self.a) + self.gamma / (4 * math.pi) * torch.log(r2)
        feats = torch.cat([x - 0.5, y, torch.log(r2) / 6], 1)
        return base + (1 - r2 / R_FAR ** 2) * self.mlp(feats)


def _velocity(model, x, y):
    psi = model(x, y)
    gx, gy = torch.autograd.grad(psi.sum(), (x, y), create_graph=True)
    return gy, -gx                                                # u = psi_y, v = -psi_x


def train_pinn(cfg: Dict, seed: int, budget: ComputeBudget, *, case: str) -> Dict:
    S = splits()
    rec = next(r for k in ("val", "test", "naca") for r in S[k] if r["id"] == case)
    shape, alpha = np.array(rec["shape"]), float(rec["alpha"])
    P = grid_points(shape, GridSpec())
    cc = cell_centres(P)                                          # (nj, ni, 2)
    d = _distances(GridSpec().first, GridSpec().radius, GridSpec().nj)
    dc = 0.5 * (d[:-1] + d[1:])
    pts = cc[(dc > 1e-3) & (dc < R_FAR - 1)].reshape(-1, 2)
    pts = pts[np.linalg.norm(pts - [0.25, 0], axis=1) < R_FAR - 0.5]
    pool = torch.tensor(pts, dtype=torch.float32)
    wl = torch.tensor(wall_loop(shape, 200, 4).xy, dtype=torch.float32)
    xk = np.array([0.995])
    yu, yl = surfaces(shape, xk)
    kutta = torch.tensor([[0.995, yu[0]], [0.995, yl[0]]], dtype=torch.float32)

    spec = PINNProblemSpec(pde_residuals=["Derivative(psi(x, y), (x, 2)) + Derivative(psi(x, y), (y, 2))"],
                           conditions=[{"name": "wall", "equation": "psi(x, y) - C", "weight": 10.0}],
                           independent_vars=["x", "y"], dependent_vars=["psi"], inverse_params=["C"])
    torch.manual_seed(seed)
    model = PINN(PotentialNet(cfg, alpha), inverse_params_names=["C"], initial_guesses={"C": 0.0})
    loss_fn = PINNFactory(spec).generate_loss_function()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    g = torch.Generator().manual_seed(1000 + seed)

    def step(progress):
        set_lr(opt, cosine_lr(progress, cfg["lr"]))
        c = pool[torch.randint(0, len(pool), (cfg["n_colloc"],), generator=g)]
        batch = {"collocation": (c[:, :1].clone().requires_grad_(True), c[:, 1:].clone().requires_grad_(True)),
                 "conditions": [(wl[:, :1].clone().requires_grad_(True), wl[:, 1:].clone().requires_grad_(True))]}
        loss, _ = loss_fn(model, batch)
        kx, ky = kutta[:, :1].clone().requires_grad_(True), kutta[:, 1:].clone().requires_grad_(True)
        u, v = _velocity(model, kx, ky)
        q = u ** 2 + v ** 2
        loss = loss + 10.0 * (q[0] - q[1]).pow(2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    run = train_within_budget(step, budget)
    gamma = float(model.net.gamma.detach())
    # wall pressure integral (as the app's forces(): p n over the wall faces); kinematic p = (1 - |u|^2) / 2
    loop = wall_loop(shape, 200, 4).xy
    a_, b_ = loop, np.roll(loop, -1, 0)
    mid = torch.tensor(0.5 * (a_ + b_), dtype=torch.float32)
    e = b_ - a_
    L = np.linalg.norm(e, axis=1)
    n_body = np.stack([e[:, 1], -e[:, 0]], 1) / L[:, None]
    mx, my = mid[:, :1].clone().requires_grad_(True), mid[:, 1:].clone().requires_grad_(True)
    u, v = _velocity(model, mx, my)
    p = (0.5 * (1 - (u ** 2 + v ** 2))).detach().numpy().ravel()
    F = ((p[:, None] * n_body) * L[:, None]).sum(0)
    a = math.radians(alpha)
    cd_p = float(F @ np.array([math.cos(a), math.sin(a)]) / 0.5)
    cl_p = float(F @ np.array([-math.sin(a), math.cos(a)]) / 0.5)
    # fields on the same graph nodes as the MeshGraphNet
    lay = Layout()
    pos = torch.tensor(lay.positions(shape), dtype=torch.float32)
    px, py = pos[:, :1].clone().requires_grad_(True), pos[:, 1:].clone().requires_grad_(True)
    u, v = _velocity(model, px, py)
    u, v = u.detach().numpy().ravel(), v.detach().numpy().ravel()
    pf = 0.5 * (1 - (u ** 2 + v ** 2))
    z = np.load(rec["npz"])
    true_f = np.stack([z["p"], z["Ux"], z["Uy"]], 1)[lay.cell_index][None]
    pred_f = np.stack([pf, u, v], 1)[None]
    true_c = coefs([rec])
    pred_c = np.array([[2 * gamma, max(cd_p, 1e-12), np.nan]])
    out = {"train": run.to_dict(), "n_params": sum(p.numel() for p in model.parameters()), "case": case,
           "gamma": gamma, "cl_kutta_joukowski": 2 * gamma, "cl_pressure_integral": cl_p, "cd_pressure_integral": cd_p,
           "wall_constant": float(model.inverse_params["C"].detach()), "true": true_c[0].tolist(),
           "result": {**coef_scores(true_c, pred_c), **field_scores(true_f, pred_f)}}

    def eval_fields():
        xx, yy = pos[:, :1].clone().requires_grad_(True), pos[:, 1:].clone().requires_grad_(True)
        return _velocity(model, xx, yy)
    out["infer_eval_fields"] = time_per_query(eval_fields, repeats=5, no_grad=False)
    return out
