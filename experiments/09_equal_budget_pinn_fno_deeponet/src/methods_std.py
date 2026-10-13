"""FNO, DeepONet (trained once on simulation data) and PINNs (trained per instance, no data) on Burgers, Darcy and
2-D Navier-Stokes, each under a ComputeBudget (pinneapple_analysis.cost.equal_budget).

Library entry points used: pinneapple_neural.architectures.neural_operators.{fno.FourierNeuralOperator, fno.FNO2d,
deeponet.DeepONet}; pinneapple_physics.pinn_solver.factory.pinn_factory.{PINNFactory, PINNProblemSpec, PINN,
NeuralNetwork}; pinneapple_simulation.numerical_solvers.operator_benchmarks.FourierInterpolant;
pinneapple_analysis.cost.{ComputeBudget, train_within_budget, cosine_lr, relative_l2, time_per_query}.
"""
from __future__ import annotations

import math
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from common import DATA, PINN_VAL_ID
from pinneapple_analysis.cost import ComputeBudget, cosine_lr, relative_l2, set_lr, time_per_query, train_within_budget
from pinneapple_neural.architectures.neural_operators.deeponet import DeepONet
from pinneapple_neural.architectures.neural_operators.fno import FNO2d, FourierNeuralOperator
from pinneapple_physics.pinn_solver.factory.pinn_factory import PINN, NeuralNetwork, PINNFactory, PINNProblemSpec
from pinneapple_simulation.numerical_solvers.operator_benchmarks import FourierInterpolant, li_forcing

NU_BURGERS, NU_NS, T_NS = 0.01, 1e-3, 10.0

# hyper-parameter grids: each method gets N_TRIALS random draws from its grid (same number for every method)
SPACES = {
    ("burgers", "fno"): {"width": [32, 64], "modes": [16, 24], "lr": [1e-3, 3e-3]},
    ("darcy", "fno"): {"width": [24, 32], "modes": [12, 16], "lr": [1e-3, 3e-3]},
    ("navier_stokes", "fno"): {"width": [24, 32], "modes": [12, 16], "lr": [1e-3, 3e-3]},
    "deeponet": {"hidden": [128, 256], "depth": [3, 4], "p": [64, 128], "lr": [1e-3, 3e-3]},
    "pinn": {"width": [64, 128], "depth": [4, 6], "lr": [1e-3, 3e-3], "n_colloc": [2048, 8192]},
}
BATCH = 20
# No weight decay for the operators. The first attempt used Adam(weight_decay=1e-4) for FNO and DeepONet; it held the
# FNO at the mean prediction on Navier-Stokes (validation rel. L2 0.185 for all 4 configurations, 0.010 without it in
# the same time). That attempt is kept in results/first_attempt_wd1e-4/ and described in the README.
OPERATOR_WD = 0.0


def space(problem: str, method: str) -> Dict:
    return SPACES.get((problem, method)) or SPACES[method]


# ------------------------------------------------------------------------------------------------- data
class Data:
    def __init__(self, problem: str):
        z = np.load(DATA / f"{problem}.npz")
        self.problem = problem
        self.raw = {k: z[k].astype(np.float32) for k in z.files}
        x, y = self.raw["train_x"], self.raw["train_y"]
        self.xm, self.xs = x.mean(0), x.std(0) + 1e-6          # pointwise Gaussian normalisers (train set only)
        self.ym, self.ys = y.mean(0), y.std(0) + 1e-6
        self.shape = x.shape[1:]

    def x(self, split):
        return torch.from_numpy((self.raw[f"{split}_x"] - self.xm) / self.xs)

    def y(self, split):
        return torch.from_numpy(self.raw[f"{split}_y"])

    def decode(self, yn: torch.Tensor) -> torch.Tensor:
        return yn * torch.from_numpy(self.ys) + torch.from_numpy(self.ym)


def qoi(problem: str, field: np.ndarray) -> np.ndarray:
    """The integral quantity reported next to the field error: Burgers energy int u^2, Darcy int u (the compliance
    for f = 1), Navier-Stokes enstrophy int w^2 (mean over the grid = integral over the unit domain)."""
    f = field.reshape(field.shape[0], -1).astype(np.float64)
    return (f ** 2).mean(1) if problem in ("burgers", "navier_stokes") else f.mean(1)


def scores(problem: str, pred: np.ndarray, true: np.ndarray) -> Dict:
    e = relative_l2(pred, true)
    q = np.abs(qoi(problem, pred) - qoi(problem, true)) / np.abs(qoi(problem, true))
    return {"rel_l2": e.tolist(), "rel_l2_mean": float(e.mean()), "qoi_rel_err": q.tolist(),
            "qoi_rel_err_mean": float(q.mean())}


# ------------------------------------------------------------------------------------------------- operators
def coords(shape) -> torch.Tensor:
    if len(shape) == 1:
        return (torch.arange(shape[0], dtype=torch.float32) / shape[0]).reshape(-1, 1)
    g = torch.arange(shape[0], dtype=torch.float32) / shape[0]
    X, Y = torch.meshgrid(g, g, indexing="ij")
    return torch.stack([X.ravel(), Y.ravel()], 1)


class OperatorModel(nn.Module):
    """field (B, *shape) normalised -> field (B, *shape) normalised, for FNO and DeepONet alike."""

    def __init__(self, problem: str, method: str, cfg: Dict, shape):
        super().__init__()
        self.method, self.shape = method, tuple(shape)
        if method == "fno":
            if len(shape) == 1:
                self.net = FourierNeuralOperator(1, 1, width=cfg["width"], modes=cfg["modes"], layers=4, use_grid=True)
            else:
                self.net = FNO2d(1, 1, width=cfg["width"], modes1=cfg["modes"], modes2=cfg["modes"], layers=4,
                                 use_grid=True)
        else:
            n = int(np.prod(shape))
            self.net = DeepONet(n, len(shape), 1, hidden=cfg["hidden"], modes=cfg["p"], depth=cfg["depth"])
            self.register_buffer("xy", coords(shape))

    def forward(self, x):
        if self.method == "fno":
            return self.net(x.unsqueeze(1)).y.squeeze(1)
        y = self.net(x.reshape(len(x), -1), self.xy).y            # (B, N, 1)
        return y.reshape(len(x), *self.shape)


def train_operator(problem: str, method: str, cfg: Dict, seed: int, budget: ComputeBudget, *,
                   eval_splits=("val",)) -> Dict:
    D = Data(problem)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = OperatorModel(problem, method, cfg, D.shape)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=OPERATOR_WD)
    Xtr, Ytr = D.x("train"), D.y("train")
    ym, ys = torch.from_numpy(D.ym), torch.from_numpy(D.ys)
    n = len(Xtr)
    state = {"perm": rng.permutation(n), "pos": 0}

    def step(progress):
        set_lr(opt, cosine_lr(progress, cfg["lr"]))
        if state["pos"] + BATCH > n:
            state["perm"], state["pos"] = rng.permutation(n), 0
        idx = torch.from_numpy(state["perm"][state["pos"]:state["pos"] + BATCH])
        state["pos"] += BATCH
        pred = model(Xtr[idx]) * ys + ym
        tgt = Ytr[idx]
        loss = (torch.linalg.vector_norm((pred - tgt).reshape(BATCH, -1), dim=1)
                / torch.linalg.vector_norm(tgt.reshape(BATCH, -1), dim=1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    run = train_within_budget(step, budget)
    model.eval()
    out = {"train": run.to_dict(), "n_params": sum(p.numel() for p in model.parameters())}
    with torch.no_grad():
        for split in eval_splits:
            pred = torch.cat([D.decode(model(xb)) for xb in torch.split(D.x(split), 50)]).numpy()
            out[split] = scores(problem, pred, D.raw[f"{split}_y"])
        x1 = D.x("test")[:1]
        out["infer_batch1"] = time_per_query(lambda: model(x1), repeats=30, warmup=5)
        xb = D.x("test")[:50]
        out["infer_batch50"] = time_per_query(lambda: model(xb), repeats=5, warmup=1, n_per_call=50)
    return out


# ------------------------------------------------------------------------------------------------- PINNs
class Bilinear:
    """Bilinear interpolant of a cell-centred n x n field on the unit square and its gradient (constant per cell
    in the other direction); values outside the outermost centres are held constant. No autograd needed: the
    coefficient does not depend on the network."""

    def __init__(self, a: np.ndarray):
        self.a = torch.from_numpy(a.astype(np.float32))
        self.n = a.shape[0]

    def _locate(self, x, y):
        n = self.n
        gx = (x.detach().reshape(-1) * n - 0.5).clamp(0, n - 1 - 1e-6)
        gy = (y.detach().reshape(-1) * n - 0.5).clamp(0, n - 1 - 1e-6)
        i, j = gx.floor().long().clamp(max=n - 2), gy.floor().long().clamp(max=n - 2)
        fx, fy = gx - i, gy - j
        inside_x = ((x.detach().reshape(-1) * n - 0.5) > 0) & ((x.detach().reshape(-1) * n - 0.5) < n - 1)
        inside_y = ((y.detach().reshape(-1) * n - 0.5) > 0) & ((y.detach().reshape(-1) * n - 0.5) < n - 1)
        A = self.a
        a00, a10, a01, a11 = A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1]
        return a00, a10, a01, a11, fx, fy, inside_x, inside_y

    def value(self, x, y):
        a00, a10, a01, a11, fx, fy, _, _ = self._locate(x, y)
        return ((1 - fx) * (1 - fy) * a00 + fx * (1 - fy) * a10 + (1 - fx) * fy * a01 + fx * fy * a11).reshape(-1, 1)

    def dx(self, x, y):
        a00, a10, a01, a11, fx, fy, ix, _ = self._locate(x, y)
        g = ((1 - fy) * (a10 - a00) + fy * (a11 - a01)) * self.n
        return (g * ix).reshape(-1, 1)

    def dy(self, x, y):
        a00, a10, a01, a11, fx, fy, _, iy = self._locate(x, y)
        g = ((1 - fx) * (a01 - a00) + fx * (a11 - a10)) * self.n
        return (g * iy).reshape(-1, 1)


class HardNet(nn.Module):
    """Wraps a library NeuralNetwork with input features and hard initial/boundary conditions.

    burgers: (x, t) -> u = u0(x) + t * N(sin 2pi kx, cos 2pi kx, 2t - 1),   k = 1, 2   (periodic, IC exact)
    darcy:   (x, y) -> u = 16 x(1-x) y(1-y) * 0.1 * N(2x-1, 2y-1)            (u = 0 on the boundary exactly)
    navier_stokes: (x, y, t) -> w = w0(x, y) + (t/T) N_w(features), psi = N_psi(features) / (4 pi^2)
    """

    def __init__(self, problem: str, cfg: Dict, init=None):
        super().__init__()
        self.problem = problem
        n_in = {"burgers": 5, "darcy": 2, "navier_stokes": 9}[problem]
        n_out = 2 if problem == "navier_stokes" else 1
        self.mlp = NeuralNetwork(num_inputs=n_in, num_outputs=n_out, num_layers=cfg["depth"],
                                 num_neurons=cfg["width"], activation=nn.Tanh())
        self.init = init

    @staticmethod
    def _per(z):
        w = 2 * math.pi
        return [torch.sin(w * z), torch.cos(w * z), torch.sin(2 * w * z), torch.cos(2 * w * z)]

    def forward(self, X):
        if self.problem == "burgers":
            x, t = X[:, :1], X[:, 1:2]
            n = self.mlp(torch.cat(self._per(x) + [2 * t - 1], 1))
            return self.init(x) + t * n
        if self.problem == "darcy":
            x, y = X[:, :1], X[:, 1:2]
            return 16 * x * (1 - x) * y * (1 - y) * 0.1 * self.mlp(torch.cat([2 * x - 1, 2 * y - 1], 1))
        x, y, t = X[:, :1], X[:, 1:2], X[:, 2:3]
        n = self.mlp(torch.cat(self._per(x) + self._per(y) + [2 * t / T_NS - 1], 1))
        w = self.init(x, y) + (t / T_NS) * n[:, :1]
        psi = n[:, 1:2] / (4 * math.pi ** 2)
        return torch.cat([w, psi], 1)


def _pinn_problem(problem: str, x_in: np.ndarray):
    """(spec, init, exogenous, sampler(n) -> collocation tuple, evaluator(model) -> field on the data grid)."""
    if problem == "burgers":
        init = FourierInterpolant(x_in)
        spec = PINNProblemSpec(
            pde_residuals=["Derivative(u(x, t), t) + u(x, t)*Derivative(u(x, t), x) - nu*Derivative(u(x, t), (x, 2))"],
            independent_vars=["x", "t"], dependent_vars=["u"], constants={"nu": NU_BURGERS})
        n = x_in.shape[0]
        xg = (torch.arange(n, dtype=torch.float32) / n).reshape(-1, 1)

        def sample(m, g):
            return (torch.rand(m, 1, generator=g).requires_grad_(True), torch.rand(m, 1, generator=g).requires_grad_(True))

        def evaluate(model):
            with torch.no_grad():
                return model(xg, torch.ones_like(xg)).numpy().reshape(n)
        return spec, init, sample, evaluate
    if problem == "darcy":
        a = Bilinear(x_in)
        spec = PINNProblemSpec(
            pde_residuals=["a(x, y)*(Derivative(u(x, y), (x, 2)) + Derivative(u(x, y), (y, 2)))"
                           " + ax(x, y)*Derivative(u(x, y), x) + ay(x, y)*Derivative(u(x, y), y) + 1"],
            independent_vars=["x", "y"], dependent_vars=["u"],
            exogenous={"a": a.value, "ax": a.dx, "ay": a.dy})
        n = x_in.shape[0]
        c = (torch.arange(n, dtype=torch.float32) + 0.5) / n
        X, Y = torch.meshgrid(c, c, indexing="ij")

        def sample(m, g):
            return (torch.rand(m, 1, generator=g).requires_grad_(True), torch.rand(m, 1, generator=g).requires_grad_(True))

        def evaluate(model):
            with torch.no_grad():
                return model(X.reshape(-1, 1), Y.reshape(-1, 1)).numpy().reshape(n, n)
        return spec, None, sample, evaluate
    init = FourierInterpolant(x_in)
    n = x_in.shape[0]
    f = FourierInterpolant(li_forcing(n))
    spec = PINNProblemSpec(
        pde_residuals=[
            "Derivative(w(x, y, t), t) + Derivative(psi(x, y, t), y)*Derivative(w(x, y, t), x)"
            " - Derivative(psi(x, y, t), x)*Derivative(w(x, y, t), y)"
            " - nu*(Derivative(w(x, y, t), (x, 2)) + Derivative(w(x, y, t), (y, 2))) - f(x, y)",
            "Derivative(psi(x, y, t), (x, 2)) + Derivative(psi(x, y, t), (y, 2)) + w(x, y, t)"],
        independent_vars=["x", "y", "t"], dependent_vars=["w", "psi"], constants={"nu": NU_NS},
        exogenous={"f": lambda x, y, t: f(x, y).detach()})
    g1 = torch.arange(n, dtype=torch.float32) / n
    X, Y = torch.meshgrid(g1, g1, indexing="ij")

    def sample(m, g):
        return (torch.rand(m, 1, generator=g).requires_grad_(True), torch.rand(m, 1, generator=g).requires_grad_(True),
                (T_NS * torch.rand(m, 1, generator=g)).requires_grad_(True))

    def evaluate(model):
        with torch.no_grad():
            xx, yy = X.reshape(-1, 1), Y.reshape(-1, 1)
            return model(xx, yy, torch.full_like(xx, T_NS))[:, 0].numpy().reshape(n, n)
    return spec, init, sample, evaluate


def train_pinn(problem: str, cfg: Dict, seed: int, budget: ComputeBudget, *, split: str, instance: int) -> Dict:
    """One PINN on one instance (the input field of ``split`` number ``instance``); no simulation data is used."""
    z = np.load(DATA / f"{problem}.npz")
    x_in, y_true = z[f"{split}_x"][instance].astype(np.float64), z[f"{split}_y"][instance]
    spec, init, sample, evaluate = _pinn_problem(problem, x_in)
    torch.manual_seed(seed)
    model = PINN(HardNet(problem, cfg, init))
    loss_fn = PINNFactory(spec).generate_loss_function()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    g = torch.Generator().manual_seed(1000 + seed)

    def step(progress):
        set_lr(opt, cosine_lr(progress, cfg["lr"]))
        loss, _ = loss_fn(model, {"collocation": sample(cfg["n_colloc"], g)})
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    run = train_within_budget(step, budget)
    pred = evaluate(model)
    out = {"train": run.to_dict(), "n_params": sum(p.numel() for p in model.parameters()),
           split: scores(problem, pred[None], y_true[None]), "split": split, "instance": instance}
    out["infer_eval_grid"] = time_per_query(lambda: evaluate(model), repeats=10, warmup=2)
    return out


def pinn_val_instance():
    return "val", PINN_VAL_ID
