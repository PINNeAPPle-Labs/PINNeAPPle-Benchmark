"""Poincare conjecture (proved by Perelman, 2002-03) -- computational illustration of the Ricci-flow mechanism.

(A) Normalised Ricci flow on axisymmetric metrics g = e^{2u(theta)} g_S2 on the 2-sphere:
        u_t = 1 - K,  K = e^{-2u} (1 - Lap_0 u),  Lap_0 u = (sin th u_th)_th / sin th
    (area 4 pi preserved). Solved by finite differences and by a PINNeAPPle PINN; the curvature must
    converge to K = 1 (Hamilton 1988, Chow 1991).
(B) Ricci flow of left-invariant metrics g = A s1^2 + B s2^2 + C s3^2 on S^3 = SU(2) (Milnor frame):
    dg_ii/dt = -2 Ric_ii, volume-normalised; squashed (Berger-type) metrics must converge to the round
    one, and Perelman's lambda-functional (= scalar curvature R for a homogeneous metric) must be
    non-decreasing along the unnormalised flow.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "PINNeAPPle"))
from pinneapple_neural.architectures.modified_mlp import ModifiedMLP  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
OUT = {}


# ---------------------------------------------------------------------------- (A) 2-sphere
def u0(th, eps=0.8):
    """Peanut-shaped initial conformal factor, renormalised to area 4 pi."""
    u = eps * 0.5 * (3 * np.cos(th) ** 2 - 1)
    area = np.trapezoid(np.exp(2 * u) * np.sin(th), th) * 2 * np.pi
    return u - 0.5 * np.log(area / (4 * np.pi))


def curvature(u, th):
    dth = th[1] - th[0]
    s = np.sin(th)
    up = np.gradient(u, dth)
    lap = np.gradient(s * up, dth) / np.maximum(s, 1e-12)
    lap[0] = 2 * (u[1] - u[0]) / dth ** 2 * 2      # regular pole: Lap = 2 u_thth
    lap[-1] = 2 * (u[-2] - u[-1]) / dth ** 2 * 2
    return np.exp(-2 * u) * (1 - lap)


def fd_flow(n=201, T=3.0, dynamic=True):
    """dynamic=True: exact normalisation r(t) = 8 pi / A(t) (area-preserving); False: r = 2 (the PINN's form)."""
    th = np.linspace(0, np.pi, n)
    u = u0(th)
    dth = th[1] - th[0]
    dt = 0.2 * dth ** 2 * np.exp(2 * u.min())
    hist = {"t": [], "maxdevK": [], "area": []}
    t = 0.0
    snaps = {}
    while t < T:
        K = curvature(u, th)
        if len(hist["t"]) == 0 or t - hist["t"][-1] >= 0.05:
            hist["t"].append(t); hist["maxdevK"].append(float(np.abs(K - 1).max()))
            hist["area"].append(float(np.trapezoid(np.exp(2 * u) * np.sin(th), th) * 2 * np.pi))
        for tt in (0.0, 0.5, 1.0, 2.0):
            if abs(t - tt) < dt / 2 or (tt == 0 and t == 0):
                snaps[str(tt)] = u.copy()
        A = np.trapezoid(np.exp(2 * u) * np.sin(th), th) * 2 * np.pi
        u = u + dt * ((4 * np.pi / A if dynamic else 1.0) - K)
        dt = 0.2 * dth ** 2 * np.exp(2 * u.min())
        t += dt
    return th, u, hist, snaps


def pinn_flow(T=1.0, iters=4000):
    """PINN for u(theta, t) on [0, pi] x [0, T]; u(theta, 0) = u0 imposed by construction."""
    torch.manual_seed(0)
    net = ModifiedMLP(in_dim=2, out_dim=1, hidden_dim=64, n_layers=4, n_fourier=16, sigma=1.0).double()
    thg = np.linspace(0, np.pi, 2001)
    shift = float(u0(thg)[0] - 0.8 * 0.5 * (3 * np.cos(thg[0]) ** 2 - 1))   # area-normalisation constant

    def u0_t(th):  # closed form, differentiable in theta
        return 0.8 * 0.5 * (3 * torch.cos(th) ** 2 - 1) + shift

    def U(th, t):
        # cos(theta) as input keeps the pole regularity (even extension) built in
        return u0_t(th) + t * net(torch.cat([torch.cos(th), t / T], 1)).y

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for it in range(iters):
        th = (torch.rand(1024, 1, dtype=torch.float64) * (np.pi - 0.1) + 0.05).requires_grad_(True)
        t = (torch.rand(1024, 1, dtype=torch.float64) * T).requires_grad_(True)
        u = U(th, t)
        ut = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        uth = torch.autograd.grad(u.sum(), th, create_graph=True)[0]
        s = torch.sin(th)
        lap = torch.autograd.grad((s * uth).sum(), th, create_graph=True)[0] / s
        K = torch.exp(-2 * u) * (1 - lap)
        loss = ((ut - (1 - K)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda th, t: U(torch.tensor(th[:, None]), torch.full((len(th), 1), float(t), dtype=torch.float64)).detach().numpy()[:, 0], float(loss)


def part_a():
    th, uT, hist, snaps = fd_flow(T=1.0 if SMOKE else 3.0)
    OUT["A_fd"] = {"history": hist, "final_maxdevK": hist["maxdevK"][-1], "area_drift": max(hist["area"]) - min(hist["area"])}
    # exponential decay rate of max|K-1|
    t, d = np.array(hist["t"]), np.array(hist["maxdevK"])
    sel = (t > 0.3) & (d > 1e-8)
    OUT["A_fd"]["decay_rate"] = float(-np.polyfit(t[sel], np.log(d[sel]), 1)[0]) if sel.sum() > 3 else None
    pinn, loss = pinn_flow(T=1.0, iters=300 if SMOKE else 6000)
    # compare with FD at t = 0.5 and 1.0 on fresh points
    th2, _, _, snaps2 = fd_flow(T=1.01, dynamic=False)   # same equation the PINN solves
    cmp = {}
    for tt in ("0.5", "1.0"):
        if tt in snaps2:
            up = pinn(th2, float(tt))
            cmp[tt] = {"max_abs_diff_u": float(np.abs(up - snaps2[tt]).max()),
                       "rel_to_initial_amplitude": float(np.linalg.norm(up - snaps2[tt]) / np.linalg.norm(snaps2["0.0"] - snaps2["0.0"].mean()))}
    OUT["A_pinn"] = {"final_train_residual": loss, "vs_fd": cmp}
    np.savez_compressed(HERE / "sphere.npz", th=th, **{f"u_{k}": v for k, v in snaps.items()})
    print(OUT["A_fd"]["final_maxdevK"], OUT["A_fd"]["decay_rate"], OUT["A_pinn"], flush=True)


# ---------------------------------------------------------------------------- (B) SU(2)
def ricci_diag(g):
    """Principal Ricci curvatures r_i (Ric(f_i,f_i)) of g = diag(A,B,C) in a Milnor frame of SU(2)
    ([e2,e3]=2e1 cyclic). Orthonormal f_i = e_i/sqrt(g_i): lambda_1 = 2 sqrt(A)/sqrt(BC) etc.;
    mu_i = (l1+l2+l3)/2 - l_i;  r_1 = 2 mu2 mu3 (Milnor 1976)."""
    A, B, C = g
    l = np.array([2 * np.sqrt(A / (B * C)), 2 * np.sqrt(B / (A * C)), 2 * np.sqrt(C / (A * B))])
    mu = l.sum() / 2 - l
    return np.array([2 * mu[1] * mu[2], 2 * mu[0] * mu[2], 2 * mu[0] * mu[1]])


def su2_flow(g0, T=3.0, dt=1e-4, normalise=True):
    g = np.array(g0, float)
    hist = {"t": [], "g": [], "R": [], "aniso": []}
    t = 0.0
    while t <= T:
        r = ricci_diag(g)
        R = float(r.sum())
        if int(round(t / dt)) % 200 == 0:
            hist["t"].append(t); hist["g"].append(g.tolist()); hist["R"].append(R)
            hist["aniso"].append(float(g.max() / g.min() - 1))
        dg = -2 * g * r
        if normalise:  # keep volume sqrt(ABC) fixed: add (2/3) mean(R) g
            dg += (2.0 / 3.0) * R * g
        g = g + dt * dg
        t += dt
        if not np.isfinite(g).all() or g.min() <= 0:
            hist["singular_at"] = t
            break
    return hist


def part_b():
    rng = np.random.default_rng(3)
    cases = [(1.0, 1.0, 0.2), (1.0, 1.0, 4.0), (0.3, 1.0, 3.0)] + [tuple(np.exp(rng.normal(0, 0.8, 3))) for _ in range(5 if SMOKE else 20)]
    res = []
    for g0 in cases:
        g0 = np.array(g0) / np.prod(g0) ** (1 / 3)          # unit volume
        h = su2_flow(g0, T=1.0 if SMOKE else 4.0)
        hu = su2_flow(g0, T=0.05, normalise=False)         # unnormalised: Perelman lambda (= R) monotone?
        dR = np.diff(hu["R"])
        res.append({"g0": g0.tolist(), "final_anisotropy": h["aniso"][-1], "initial_anisotropy": h["aniso"][0],
                    "R_min_initial": h["R"][0], "lambda_monotone_unnormalised": bool((dR >= -1e-10).all()),
                    "singular_at": h.get("singular_at")})
    OUT["B_su2"] = {"cases": res,
                    "all_converged_to_round": bool(all(r["final_anisotropy"] < 1e-3 for r in res)),
                    "all_lambda_monotone": bool(all(r["lambda_monotone_unnormalised"] for r in res))}
    # round-sphere sanity: g = (1,1,1) -> Ric = (1/2) * g? (unit S3 of radius 2 has Ric = 2/4 g)
    OUT["B_sanity_round_ricci"] = ricci_diag(np.ones(3)).tolist()
    print(OUT["B_su2"]["all_converged_to_round"], OUT["B_su2"]["all_lambda_monotone"], OUT["B_sanity_round_ricci"], flush=True)


if __name__ == "__main__":
    import sys as _s
    if "A" in _s.argv[1:]:
        OUT.update(json.loads((HERE / "results.json").read_text()))
        OUT["A_first_attempt_fixed_r"] = {k: OUT.get(k) for k in ("A_fd", "A_pinn")}
    else:
        part_b()
        (HERE / "results.json").write_text(json.dumps(OUT, indent=1, default=float))
    part_a()
    (HERE / "results.json").write_text(json.dumps(OUT, indent=1, default=float))
