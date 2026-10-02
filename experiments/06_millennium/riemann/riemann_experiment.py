"""Riemann hypothesis -- numerical verification in a finite window + zero statistics.

1. Sample Hardy's Z(t) (real on the critical line, |Z| = |zeta(1/2+it)|) on a fine grid up to T and
   refine every sign change -> zeros ON the critical line.
2. Compare with N(T), the number of zeros in the whole critical strip 0 < Im s < T (mpmath.nzeros,
   Turing/Backlund counting). Equality of the two counts means every zero with 0 < Im s < T lies on
   the line -- the classical way RH is verified numerically (this is ~10^3 zeros, records exceed 10^13).
3. Unfold the zeros and compare nearest-neighbour spacings / pair correlation with GUE predictions
   (Montgomery-Odlyzko law).
4. Predictability probe: a PINNeAPPle recurrent model (GRU) vs linear AR and the mean, forecasting the
   next unfolded spacing from the previous 32 (held-out block, no shuffling across the split).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "PINNeAPPle"))
SMOKE = os.environ.get("SMOKE") == "1"
T_MAX = 150.0 if SMOKE else 3000.0
mp.mp.dps = 15
OUT = {}


def save():
    (HERE / "results.json").write_text(json.dumps(OUT, indent=1, default=float))


def find_zeros():
    t0 = time.time()
    zeros = []
    t = 10.0
    z_prev = float(mp.siegelz(t))
    while t < T_MAX:
        spacing = 2 * np.pi / np.log(max(t, 20) / (2 * np.pi))
        h = spacing / 6.0
        t_new = min(t + h, T_MAX)
        z_new = float(mp.siegelz(t_new))
        if z_prev == 0 or np.sign(z_new) != np.sign(z_prev):
            zeros.append(float(mp.findroot(mp.siegelz, (t, t_new), solver="illinois")))
        t, z_prev = t_new, z_new
    return np.array(zeros), time.time() - t0


def gue_wigner(s):
    return 32 / np.pi ** 2 * s ** 2 * np.exp(-4 * s ** 2 / np.pi)


def main():
    zeros, sec = find_zeros()
    zeros = np.unique(np.round(zeros, 9))
    n_line = len(zeros)
    n_strip = int(mp.nzeros(T_MAX))
    OUT["verification"] = {"T": T_MAX, "zeros_on_line_found": n_line, "N_T_all_zeros_in_strip": n_strip,
                           "all_zeros_on_line_up_to_T": n_line == n_strip, "seconds": sec,
                           "first_zeros": zeros[:5].tolist(),
                           "check_first_zero_vs_mpmath_zetazero1": float(mp.zetazero(1).imag)}
    print(OUT["verification"], flush=True)
    # unfolding with the smooth part of N(t): theta(t)/pi + 1
    unf = np.array([float(mp.siegeltheta(g) / mp.pi + 1) for g in zeros])
    s = np.diff(unf)
    OUT["spacings"] = {"mean": float(s.mean()), "var": float(s.var()), "var_GUE_wigner": float((3 * np.pi / 8) - 1),
                       "lag1_autocorr": float(np.corrcoef(s[:-1], s[1:])[0, 1])}
    # KS distance to Wigner-GUE vs Poisson
    xs = np.sort(s)
    ecdf = np.arange(1, len(xs) + 1) / len(xs)
    grid = np.linspace(0, 4, 4001)
    cdf_gue = np.cumsum(gue_wigner(grid)) * (grid[1] - grid[0])
    OUT["spacings"]["KS_vs_GUE_wigner"] = float(np.max(np.abs(ecdf - np.interp(xs, grid, cdf_gue))))
    OUT["spacings"]["KS_vs_Poisson"] = float(np.max(np.abs(ecdf - (1 - np.exp(-xs)))))
    # pair correlation (all pairs within window 3)
    d = []
    for i in range(len(unf)):
        j = i + 1
        while j < len(unf) and unf[j] - unf[i] < 3:
            d.append(unf[j] - unf[i])
            j += 1
    hist, edges = np.histogram(d, bins=60, range=(0, 3))
    centers = 0.5 * (edges[1:] + edges[:-1])
    R2 = hist / (len(unf) * (edges[1] - edges[0]))
    mont = 1 - (np.sin(np.pi * centers) / (np.pi * centers)) ** 2
    OUT["pair_correlation"] = {"u": centers.tolist(), "empirical": R2.tolist(), "montgomery": mont.tolist(),
                               "rmse": float(np.sqrt(((R2 - mont) ** 2).mean()))}
    np.savez_compressed(HERE / "zeros.npz", zeros=zeros, unfolded=unf)
    # predictability probe
    OUT["predictability"] = predictability(s)
    save()
    print(json.dumps({k: v for k, v in OUT.items() if k != "pair_correlation"}, indent=1))


def predictability(s, lags=32):
    from pinneapple_neural.architectures.recurrent.gru import GRUModel  # noqa -- PINNeAPPle recurrent family
    X = np.stack([s[i:i + lags] for i in range(len(s) - lags)]).astype(np.float32)
    y = s[lags:].astype(np.float32)
    n = len(X)
    tr, va = int(0.7 * n), int(0.85 * n)
    mu = y[:tr].mean()
    res = {"n": n, "mse_mean_predictor": float(((y[va:] - mu) ** 2).mean())}
    A = np.c_[X[:tr], np.ones(tr)]
    w = np.linalg.lstsq(A, y[:tr], rcond=None)[0]
    res["mse_linear_AR32"] = float(((np.c_[X[va:], np.ones(n - va)] @ w - y[va:]) ** 2).mean())
    torch.manual_seed(0)
    m = GRUModel(in_dim=1, out_dim=1, horizon=1, hidden_dim=64, num_layers=2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xt = torch.tensor(X[..., None] - mu)
    yt = torch.tensor(y - mu)
    best, state = 1e9, None
    for ep in range(3 if SMOKE else 60):
        m.train()
        for idx in torch.randperm(tr).split(64):
            out = m(Xt[idx])
            pred = (out.y if hasattr(out, "y") else out)[:, -1, 0] if (out.y if hasattr(out, "y") else out).dim() == 3 else (out.y if hasattr(out, "y") else out)[:, 0]
            loss = ((pred - yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            out = m(Xt[tr:va]); o = out.y if hasattr(out, "y") else out
            pv = o[:, -1, 0] if o.dim() == 3 else o[:, 0]
            v = float(((pv - yt[tr:va]) ** 2).mean())
        if v < best:
            best, state = v, {k: x.clone() for k, x in m.state_dict().items()}
    m.load_state_dict(state)
    with torch.no_grad():
        out = m(Xt[va:]); o = out.y if hasattr(out, "y") else out
        pt = o[:, -1, 0] if o.dim() == 3 else o[:, 0]
    res["mse_gru_pinneapple"] = float(((pt - yt[va:]) ** 2).mean())
    for k in ("mse_linear_AR32", "mse_gru_pinneapple"):
        res[k.replace("mse", "R2")] = 1 - res[k] / res["mse_mean_predictor"]
    return res


if __name__ == "__main__":
    main()
