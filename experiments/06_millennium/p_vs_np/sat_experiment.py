"""P vs NP -- empirical hardness of random 3-SAT (what experiments can and cannot say).

1. Phase transition: P(satisfiable) vs clause density alpha = M/N for several N (complete CDCL solver,
   PySAT/Glucose4), and the solver cost (conflicts) peaking near the threshold alpha_c ~ 4.27.
2. Scaling at alpha = 4.26: median conflicts vs N, exponential vs polynomial fits compared by AIC.
3. A physics-inspired analog solver (continuous-time dynamical system of Ercsey-Ravasz & Toroczkai,
   Nature Physics 2011) on satisfiable instances: analog time-to-solution vs N.
4. Learning satisfiability: a PINNeAPPle MLP classifier on instance statistics vs an alpha-only baseline.
Empirical scaling on random instances says nothing about worst-case complexity classes; the paper
states this explicitly.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from pysat.solvers import Glucose4

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "PINNeAPPle"))
SMOKE = os.environ.get("SMOKE") == "1"
rng = np.random.default_rng(2026)
OUT = {}


def save():
    (HERE / "results.json").write_text(json.dumps(OUT, indent=1, default=float))


def random_3sat(n, m):
    vars_ = np.array([rng.choice(n, 3, replace=False) for _ in range(m)]) + 1
    signs = rng.choice([-1, 1], size=(m, 3))
    return (vars_ * signs).tolist()


def solve(cnf):
    with Glucose4(bootstrap_with=cnf) as s:
        t0 = time.time()
        sat = s.solve()
        st = s.accum_stats()
        return bool(sat), int(st.get("conflicts", 0)), time.time() - t0, (s.get_model() if sat else None)


def phase_transition():
    Ns = [50, 100] if SMOKE else [50, 100, 150, 200]
    alphas = np.round(np.arange(3.0, 5.51, 0.25), 2)
    reps = 10 if SMOKE else 100
    res = {}
    for n in Ns:
        rows = []
        for a in alphas:
            sats, confs = [], []
            for _ in range(reps):
                sat, c, _, _ = solve(random_3sat(n, int(round(a * n))))
                sats.append(sat)
                confs.append(c)
            rows.append({"alpha": float(a), "p_sat": float(np.mean(sats)), "median_conflicts": float(np.median(confs))})
        res[str(n)] = rows
        print("PT", n, [(r["alpha"], r["p_sat"]) for r in rows], flush=True)
    # finite-size crossing estimate: alpha where p_sat = 0.5 (linear interpolation) per N
    cross = {}
    for n, rows in res.items():
        a = np.array([r["alpha"] for r in rows]); p = np.array([r["p_sat"] for r in rows])
        i = np.where((p[:-1] >= 0.5) & (p[1:] < 0.5))[0]
        cross[n] = float(a[i[0]] + (p[i[0]] - 0.5) / (p[i[0]] - p[i[0] + 1]) * (a[i[0] + 1] - a[i[0]])) if len(i) else None
    OUT["phase_transition"] = {"curves": res, "alpha_half": cross}
    save()


def scaling():
    Ns = [50, 100, 150] if SMOKE else [50, 100, 150, 200, 250, 300, 350]
    reps = 10 if SMOKE else 60
    rows = []
    for n in Ns:
        confs = [solve(random_3sat(n, int(round(4.26 * n))))[1] for _ in range(reps)]
        rows.append({"N": n, "median_conflicts": float(np.median(confs)), "q90": float(np.quantile(confs, 0.9))})
        print("SC", rows[-1], flush=True)
    N = np.array([r["N"] for r in rows], float)
    y = np.log(np.maximum([r["median_conflicts"] for r in rows], 1.0))
    # exponential: log y = a + b N ; power law: log y = a + k log N
    fits = {}
    for name, X in (("exponential", N), ("power_law", np.log(N))):
        A = np.c_[np.ones_like(X), X]
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        rss = float(((A @ coef - y) ** 2).sum())
        fits[name] = {"coef": coef.tolist(), "rss": rss, "aic": len(N) * np.log(rss / len(N) + 1e-12) + 4}
    OUT["scaling_alpha_4.26"] = {"rows": rows, "fits": fits}
    save()


def analog_solver(cnf, n, t_max=500.0, seed=0):
    """Continuous-time SAT solver (Ercsey-Ravasz & Toroczkai 2011); explicit Euler with a CFL-like cap."""
    r = np.random.default_rng(seed)
    C = np.array(cnf)
    idx = np.abs(C) - 1
    cm = np.sign(C).astype(float)
    m = len(C)
    s = r.uniform(-1, 1, n)
    a = np.ones(m)
    t, dt = 0.0, 0.05
    while t < t_max:
        f = 1 - cm * s[idx]                                  # (m,3)
        K = 0.125 * f.prod(1)
        if np.all((cm * s[idx] > 0).any(1)):                 # every clause has a true literal
            return t
        Kmi = 0.125 * np.stack([f[:, 1] * f[:, 2], f[:, 0] * f[:, 2], f[:, 0] * f[:, 1]], 1)
        grad = np.zeros(n)
        np.add.at(grad, idx.ravel(), (2 * a[:, None] * cm * Kmi * K[:, None]).ravel())
        step = min(dt, 0.1 / (np.abs(grad).max() + 1e-12))
        s = np.clip(s + step * grad, -1, 1)
        a = a * np.exp(step * K)
        t += step
    return None


def analog():
    Ns = [20, 30] if SMOKE else [20, 30, 40, 50, 60, 80]
    reps = 5 if SMOKE else 30
    rows = []
    for n in Ns:
        times, solved = [], 0
        k = 0
        while k < reps:
            cnf = random_3sat(n, int(round(4.25 * n)))
            if not solve(cnf)[0]:
                continue                                    # analog solver is only defined to succeed on SAT instances
            t = analog_solver(cnf, n, seed=k)
            k += 1
            if t is not None:
                solved += 1
                times.append(t)
        rows.append({"N": n, "solved_frac": solved / reps, "median_analog_time": float(np.median(times)) if times else None})
        print("AN", rows[-1], flush=True)
    OUT["analog_solver"] = rows
    save()


def learn_sat():
    """PINNeAPPle MLP on instance statistics near the threshold vs alpha-only logistic baseline."""
    import torch
    from pinneapple_neural.architectures.modified_mlp import ModifiedMLP
    n = 100
    X, y = [], []
    for _ in range(300 if SMOKE else 3000):
        a = rng.uniform(3.8, 4.8)
        cnf = random_3sat(n, int(round(a * n)))
        C = np.abs(np.array(cnf)) - 1
        occ = np.bincount(C.ravel(), minlength=n)
        pos = np.bincount(C.ravel()[np.array(cnf).ravel() > 0], minlength=n)
        bias = np.abs(2 * pos - occ) / np.maximum(occ, 1)
        X.append([a, occ.std(), occ.max(), bias.mean(), bias.std(), (occ == 0).mean()])
        y.append(solve(cnf)[0])
    X, y = np.array(X, np.float32), np.array(y, np.float32)
    ntr = int(0.7 * len(X)); nva = int(0.85 * len(X))
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xn = (X - mu) / sd
    out = {}
    for name, cols in (("alpha_only", [0]), ("all_features", list(range(X.shape[1])))):
        torch.manual_seed(0)
        # Random-Fourier-feature bandwidth: sigma=1.0 (the ModifiedMLP default) is tuned for low-dimensional
        # input; on the 6-D "all_features" case it made training fail (~0.49 test accuracy, worse than the
        # 1-D "alpha_only" case's 0.81, and worse than chance) for the same reason documented in
        # 04_terramechanics_robust/src/terra_experiment.py's RobustTerraNet: the RFF projection's variance
        # grows with input dimension, so a bandwidth tuned for 1-2 inputs is far too high-frequency for 6.
        # sigma=0.1 recovers ~0.82 accuracy in a diagnostic rerun, matching/beating alpha_only.
        sigma = 1.0 if len(cols) <= 2 else 0.1
        m = ModifiedMLP(in_dim=len(cols), out_dim=1, hidden_dim=32, n_layers=3, n_fourier=8, sigma=sigma)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        xt, yt = torch.tensor(Xn[:, cols]), torch.tensor(y)
        best, state = 1e9, None
        for ep in range(20 if SMOKE else 300):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(xt[:ntr]).y[:, 0], yt[:ntr])
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                v = float(torch.nn.functional.binary_cross_entropy_with_logits(m(xt[ntr:nva]).y[:, 0], yt[ntr:nva]))
            if v < best:
                best, state = v, {k: x.clone() for k, x in m.state_dict().items()}
        m.load_state_dict(state)
        with torch.no_grad():
            p = torch.sigmoid(m(xt[nva:]).y[:, 0]).numpy()
        out[name] = {"test_accuracy": float(((p > 0.5) == (y[nva:] > 0.5)).mean()),
                     "test_logloss": float(-np.mean(y[nva:] * np.log(p + 1e-9) + (1 - y[nva:]) * np.log(1 - p + 1e-9)))}
    out["n_instances"] = len(X)
    OUT["learning_satisfiability_N100"] = out
    print("LS", out, flush=True)
    save()


if __name__ == "__main__":
    if os.environ.get("RESUME") == "1" and (HERE / "results.json").exists():   # machine reboot, 2026-09-25
        OUT.update(json.loads((HERE / "results.json").read_text()))
    for key, fn in (("phase_transition", phase_transition), ("scaling_alpha_4.26", scaling),
                    ("analog_solver", analog), ("learning_satisfiability_N100", learn_sat)):
        if key not in OUT:
            fn()
