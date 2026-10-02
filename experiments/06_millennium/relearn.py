"""Re-run the learning probes with validation-selected PINNeAPPle classifiers.

The first runs used a single ModifiedMLP configuration (Fourier bandwidth fixed a priori) and it
under-performed trivial baselines on BSD and Hodge. Both numbers are kept in the results as
"learning_first_attempt"; this script adds "learning" with the model chosen on validation only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "PINNeAPPle"))
sys.path.insert(0, str(HERE))
from common_classifier import fit_select  # noqa: E402


def split(n, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(0.7 * n), int(0.85 * n)
    return perm[:a], perm[a:b], perm[b:]


def bsd():
    res = json.loads((HERE / "bsd" / "results.json").read_text())
    recs = [r for r in json.loads((HERE / "bsd" / "curves.json").read_text()) if "error" not in r]
    R = [r for r in recs if r["rank_proven"] and r["rank_lo"] <= 2]
    primes = res["learning"]["primes"]
    X = np.array([[ap / math.sqrt(p) for ap, p in zip(r["ap"], primes)] for r in R], np.float32)
    y = np.array([r["rank_lo"] for r in R])
    tr, va, te = split(len(R))
    w = 1.0 / np.maximum(np.bincount(y[tr], minlength=3), 1); w = w / w.sum() * 3
    out, _ = fit_select(X, y, tr, va, te, 3, weight=w)
    mn = np.array([sum(ap * math.log(p) / p for ap, p in zip(r["ap"], primes)) for r in R])
    cm = [mn[tr][y[tr] == k].mean() for k in range(3)]
    out["mestre_nagao_accuracy_same_split"] = float((np.argmin([np.abs(mn[te] - c) for c in cm], 0) == y[te]).mean())
    out["majority_class_accuracy"] = float((y[te] == np.bincount(y[tr]).argmax()).mean())
    res.setdefault("learning_first_attempt", {k: v for k, v in res["learning"].items() if "murm" not in k and k != "primes"})
    res["learning"].update({"validation_selected": out})
    (HERE / "bsd" / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print("BSD", out["selected_on_validation"], out["test_accuracy"], out["mestre_nagao_accuracy_same_split"], flush=True)


def hodge():
    sys.path.insert(0, str(HERE / "hodge"))
    from hodge_experiment import parse_cicy
    res = json.loads((HERE / "hodge" / "results.json").read_text())
    data = parse_cicy(HERE / "hodge" / "cicylist.txt")
    R, C = 12, 15
    X = np.zeros((len(data), R * C), np.float32)
    for i, d in enumerate(data):
        M = np.zeros((R, C)); M[:d["Q"].shape[0], :d["Q"].shape[1]] = d["Q"]
        X[i] = M.ravel()
    y = np.array([d["h11"] for d in data])
    nps = np.array([d["Q"].shape[0] for d in data])
    # an extra, physically meaningful input: number of projective factors (the "favourable" guess)
    X = np.concatenate([X, nps[:, None].astype(np.float32)], 1)
    tr, va, te = split(len(data))
    out, p = fit_select(X, y, tr, va, te, int(y.max()) + 1, epochs=300)
    out["baseline_h11_equals_num_P_factors_accuracy"] = float((nps[te] == y[te]).mean())
    nf = nps[te] != y[te]
    out["accuracy_on_non_favourable_test"] = float((p[nf] == y[te][nf]).mean())
    res["learning_h11_first_attempt"] = res.get("learning_h11")
    res["learning_h11"] = out
    (HERE / "hodge" / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print("HODGE", out["selected_on_validation"], out["test_accuracy"], out["baseline_h11_equals_num_P_factors_accuracy"], flush=True)


def sat():
    sys.path.insert(0, str(HERE / "p_vs_np"))
    import sat_experiment as S
    S.rng = np.random.default_rng(7)
    n = 100
    X, y = [], []
    for _ in range(3000):
        a = S.rng.uniform(3.8, 4.8)
        cnf = S.random_3sat(n, int(round(a * n)))
        Cm = np.abs(np.array(cnf)) - 1
        occ = np.bincount(Cm.ravel(), minlength=n)
        pos = np.bincount(Cm.ravel()[np.array(cnf).ravel() > 0], minlength=n)
        bias = np.abs(2 * pos - occ) / np.maximum(occ, 1)
        X.append([a, occ.std(), occ.max(), bias.mean(), bias.std(), (occ == 0).mean()])
        y.append(int(S.solve(cnf)[0]))
    X, y = np.array(X, np.float32), np.array(y)
    tr, va, te = split(len(X))
    res = json.loads((HERE / "p_vs_np" / "results.json").read_text())
    out = {}
    for name, cols in (("alpha_only", [0]), ("all_features", list(range(6)))):
        o, _ = fit_select(X[:, cols], y, tr, va, te, 2)
        out[name] = o
    res["learning_satisfiability_N100_first_attempt"] = res.get("learning_satisfiability_N100")
    res["learning_satisfiability_N100"] = out
    (HERE / "p_vs_np" / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print("SAT", {k: (v["selected_on_validation"], v["test_accuracy"]) for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    for job in sys.argv[1:] or ["bsd", "hodge", "sat"]:
        {"bsd": bsd, "hodge": hodge, "sat": sat}[job]()
