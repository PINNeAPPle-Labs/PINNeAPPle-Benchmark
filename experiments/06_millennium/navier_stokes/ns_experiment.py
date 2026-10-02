"""Navier-Stokes existence & smoothness -- computational probes with PINNeAPPle.

Part A  PINN discovery of self-similar blow-up (method validation on a model with a known answer).
        Inviscid Burgers u_t + u u_x = 0 admits self-similar blow-up
            u = (T-t)^lambda U(xi),  xi = x/(T-t)^(1+lambda),
            -lambda U + (1+lambda) xi U' + U U' = 0,  U(0)=0, U'(0)=-1,
        with smooth (analytic) profiles only for lambda = 1/(2i+2); for lambda=1/2 the exact profile is
        the root of xi = -U - U^3. A PINN (PINNeAPPle ModifiedMLP) is trained (a) with lambda fixed on a
        grid of values -- the smoothness-penalised residual singles out the admissible rates -- and
        (b) with lambda trainable.
Part B  3-D Taylor-Green vortex at Re=1600, pseudo-spectral DNS at 32^3, 64^3, 128^3 (PINNeAPPle
        SpectralNS3D): energy, dissipation, max vorticity and the Beale-Kato-Majda integral.
Nothing here can prove or disprove regularity; the paper states exactly what is and is not shown.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "PINNeAPPle"))
from pinneapple_neural.architectures.modified_mlp import ModifiedMLP  # noqa: E402
from pinneapple_simulation.numerical_solvers.ns3d_spectral import SpectralNS3D, taylor_green_ic  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
OUT = {}
RESUME = os.environ.get("RESUME") == "1"   # reuse finished parts of results.json (machine reboot, 2026-09-25)
L = 8.0  # xi domain [-L, L]


def save():
    (HERE / "results.json").write_text(json.dumps(OUT, indent=1, default=float))


def exact_profile(xi):
    """Root of U^3 + U + xi = 0 (lambda = 1/2)."""
    U = np.empty_like(xi)
    for i, x in enumerate(xi):
        r = np.roots([1, 0, 1, x])
        U[i] = r[np.argmin(np.abs(r.imag))].real
    return U


class Profile(torch.nn.Module):
    """U(xi) odd with U(0)=0 and U'(0)=-1 by construction: U = -xi + xi^3 g(xi^2) ... scaled."""

    def __init__(self):
        super().__init__()
        self.g = ModifiedMLP(in_dim=1, out_dim=1, hidden_dim=64, n_layers=4, n_fourier=16, sigma=1.0)

    def forward(self, xi):
        s = xi / L
        return -xi + xi * s ** 2 * self.g((s ** 2)).y  # odd; U'(0) = -1 exactly


def residual(model, xi, lam):
    xi = xi.requires_grad_(True)
    U = model(xi)
    Up = torch.autograd.grad(U.sum(), xi, create_graph=True)[0]
    r = -lam * U + (1 + lam) * xi * Up + U * Up
    # smoothness: the admissible profiles are analytic; penalise the 3rd derivative's growth
    Upp = torch.autograd.grad(Up.sum(), xi, create_graph=True)[0]
    Uppp = torch.autograd.grad(Upp.sum(), xi, create_graph=True)[0]
    return r, Uppp


def train_profile(lam_init, trainable, iters, seed=0):
    torch.manual_seed(seed)
    model = Profile().double()
    lam = torch.tensor(float(lam_init), dtype=torch.float64, requires_grad=trainable)
    params = list(model.parameters()) + ([lam] if trainable else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    hist = []
    for it in range(iters):
        xi = (torch.rand(512, 1, dtype=torch.float64) * 2 - 1) * L
        r, u3 = residual(model, xi, lam)
        loss = (r ** 2).mean() + 1e-6 * (u3 ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 200 == 0:
            hist.append({"it": it, "loss": float(loss), "lambda": float(lam)})
    # LBFGS polish
    opt2 = torch.optim.LBFGS(params, lr=0.5, max_iter=300, line_search_fn="strong_wolfe")
    xi_f = torch.linspace(-L, L, 2001, dtype=torch.float64)[:, None]

    def closure():
        opt2.zero_grad()
        r, u3 = residual(model, xi_f.clone(), lam)
        l = (r ** 2).mean() + 1e-6 * (u3 ** 2).mean()
        l.backward()
        return l

    opt2.step(closure)
    r, _ = residual(model, torch.linspace(-L, L, 4001, dtype=torch.float64)[:, None], lam)  # fresh points
    return model, float(lam), float((r ** 2).mean().sqrt()), hist


def part_a():
    if RESUME and "A_trainable_lambda_second_branch" in OUT:
        return
    iters = 300 if SMOKE else 6000
    lams = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7] if not SMOKE else [0.25, 0.5]
    scan = []
    for lam in lams:
        _, _, res, _ = train_profile(lam, False, iters)
        scan.append({"lambda": lam, "residual_rms_fresh_points": res})
        print("scan", lam, res, flush=True)
    OUT["A_lambda_scan"] = scan
    m, lam_found, res, hist = train_profile(0.45, True, iters)
    xi = np.linspace(-L, L, 801)
    with torch.no_grad():
        U = m(torch.tensor(xi[:, None])).numpy()[:, 0]
    Ue = exact_profile(xi)
    OUT["A_trainable_lambda"] = {"init": 0.45, "found": lam_found, "exact_nearest": 0.5, "residual_rms": res,
                                 "profile_max_abs_err": float(np.abs(U - Ue).max()),
                                 "profile_rel_l2": float(np.linalg.norm(U - Ue) / np.linalg.norm(Ue)), "history": hist}
    m2, lam2, res2, _ = train_profile(0.22, True, iters, seed=1)
    OUT["A_trainable_lambda_second_branch"] = {"init": 0.22, "found": lam2, "exact_nearest": 0.25, "residual_rms": res2}
    np.savez_compressed(HERE / "profile.npz", xi=xi, U=U, U_exact=Ue)
    print(OUT["A_trainable_lambda"]["found"], OUT["A_trainable_lambda_second_branch"]["found"], flush=True)
    save()


def part_b():
    Re = 1600.0
    runs = dict(OUT.get("B_taylor_green", {})) if RESUME else {}
    grid = [(32, 0.02), (64, 0.01)] if SMOKE else [(48, 0.0133), (64, 0.01), (96, 0.00667)]
    t_end = 1.0 if SMOKE else 10.0
    for N, dt in grid:
        if str(N) in runs:
            continue
        t0 = time.time()
        s = SpectralNS3D(N=N, nu=1.0 / Re, dt=dt)
        h = s.run(taylor_green_ic(N), t_end, diag_every=int(round(0.05 / dt)))
        h["seconds"] = time.time() - t0
        # energy-balance check: -dE/dt vs eps (central differences)
        t, E, eps = map(np.array, (h["t"], h["E"], h["eps"]))
        dEdt = -np.gradient(E, t)
        h["energy_balance_rel_err"] = float(np.abs(dEdt[2:-2] - eps[2:-2]).max() / eps.max())
        h["peak_eps"] = float(eps.max())
        h["t_peak_eps"] = float(t[np.argmax(eps)])
        runs[str(N)] = h
        print(N, h["seconds"], h["peak_eps"], h["t_peak_eps"], h["energy_balance_rel_err"], max(h["wmax"]), flush=True)
        OUT["B_taylor_green"] = runs
        save()


if __name__ == "__main__":
    if RESUME and (HERE / "results.json").exists():
        OUT.update(json.loads((HERE / "results.json").read_text()))
    part_a()
    part_b()
