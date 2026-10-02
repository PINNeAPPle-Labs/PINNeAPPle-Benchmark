"""Birch and Swinnerton-Dyer -- numerical checks on a family of elliptic curves (PARI/GP) + learning.

For every curve y^2 = x^3 + a x + b with small |a|, |b| (non-singular, distinct j/discriminant pairs):
  1. analytic rank r_an (order of vanishing of L(E,s) at s=1, numerically) vs. algebraic rank bounds
     from 2-descent (ellrank): BSD predicts r_an = rank.
  2. for proven-rank 0 and 1 curves, the conjectural order of Sha from the BSD formula
         Sha_an = L^(r)(E,1)/r! / (Reg * Omega * prod c_p / |E_tors|^2)
     must be a perfect square integer; we measure its distance to the nearest square.
  3. a PINNeAPPle MLP predicts the rank from normalised Frobenius traces a_p/sqrt(p), p < 500,
     compared with a Mestre-Nagao sum baseline (the "murmuration" signal).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import cypari2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "PINNeAPPle"))
SMOKE = os.environ.get("SMOKE") == "1"
pari = cypari2.Pari()
pari.allocatemem(2 * 10 ** 9)
pari.default("parisizemax", 4 * 10 ** 9)
PRIMES = [int(p) for p in pari.primes(95)]  # primes < 500


def curves(bound):
    seen = set()
    for a in range(-bound, bound + 1):
        for b in range(-bound, bound + 1):
            if 4 * a ** 3 + 27 * b ** 2 == 0:
                continue
            E = pari.ellinit([a, b])
            Em = pari.ellminimalmodel(E)
            key = (str(Em[12]), str(Em[11]))  # (j, discriminant) of the minimal model
            if key in seen:
                continue
            seen.add(key)
            yield a, b, Em


def analyse(a, b, E):
    rec = {"a": a, "b": b}
    t0 = time.time()
    N = int(pari.ellglobalred(E)[0])
    rec["conductor"] = N
    ar = pari.ellanalyticrank(E)
    r_an, Lr = int(ar[0]), float(ar[1])
    rec["r_an"], rec["L_r"] = r_an, Lr
    rk = pari.ellrank(E)
    lo, hi = int(rk[0]), int(rk[1])
    rec["rank_lo"], rec["rank_hi"] = lo, hi
    rec["rank_proven"] = lo == hi
    if lo == hi and lo <= 1:
        gens = rk[3]
        if lo == 1:
            gens = pari.ellsaturation(E, gens, 1000)
            reg = float(pari.matdet(pari.ellheightmatrix(E, gens)))
        else:
            reg = 1.0
        c = float(pari.ellbsd(E))                      # Omega * prod c_p / |tors|^2 (PARI's BSD constant)
        sha = Lr / math.factorial(lo) / (reg * c) if r_an == lo else float("nan")
        rec["sha_an"] = sha
        if np.isfinite(sha):
            k = round(math.sqrt(max(sha, 0)))
            rec["sha_nearest_square"] = k * k
            rec["sha_dist_to_square"] = abs(sha - k * k)
    rec["ap"] = [int(pari.ellap(E, p)) for p in PRIMES]
    rec["seconds"] = time.time() - t0
    return rec


def learn(recs):
    import torch
    from pinneapple_neural.architectures.modified_mlp import ModifiedMLP
    R = [r for r in recs if r["rank_proven"] and r["rank_lo"] <= 2]
    X = np.array([[ap / math.sqrt(p) for ap, p in zip(r["ap"], PRIMES)] for r in R], np.float32)
    y = np.array([min(r["rank_lo"], 2) for r in R])
    # Mestre-Nagao sum S(B) = sum_{p<B} a_p log p / p (larger negative -> higher rank heuristically)
    mn = np.array([sum(ap * math.log(p) / p for ap, p in zip(r["ap"], PRIMES)) for r in R])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(R))
    ntr, nva = int(0.7 * len(R)), int(0.85 * len(R))
    tr, va, te = perm[:ntr], perm[ntr:nva], perm[nva:]
    out = {"n": len(R), "class_counts": np.bincount(y, minlength=3).tolist()}
    # baseline: thresholds on the Mestre-Nagao sum fitted on train (1-D nearest class-mean)
    cm = [mn[tr][y[tr] == k].mean() if (y[tr] == k).any() else np.inf for k in range(3)]
    pred_mn = np.argmin([np.abs(mn[te] - c) for c in cm], 0)
    out["mestre_nagao_accuracy"] = float((pred_mn == y[te]).mean())
    out["majority_class_accuracy"] = float((y[te] == np.bincount(y[tr]).argmax()).mean())
    torch.manual_seed(0)
    m = ModifiedMLP(in_dim=X.shape[1], out_dim=3, hidden_dim=128, n_layers=3, n_fourier=32)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(X), torch.tensor(y)
    w = torch.tensor(1.0 / np.maximum(np.bincount(y[tr], minlength=3), 1), dtype=torch.float32)
    best, state = 1e9, None
    for ep in range(30 if SMOKE else 400):
        loss = torch.nn.functional.cross_entropy(m(xt[tr]).y, yt[tr], weight=w / w.sum() * 3)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = float(torch.nn.functional.cross_entropy(m(xt[va]).y, yt[va], weight=w / w.sum() * 3))
        if v < best:
            best, state = v, {k: x.clone() for k, x in m.state_dict().items()}
    m.load_state_dict(state)
    with torch.no_grad():
        p = m(xt[te]).y.argmax(1).numpy()
    out["mlp_accuracy"] = float((p == y[te]).mean())
    out["mlp_confusion"] = [[int(((y[te] == i) & (p == j)).sum()) for j in range(3)] for i in range(3)]
    # murmuration curve: mean a_p/sqrt(p) by rank vs p
    out["murmuration_mean_ap_norm"] = {str(k): X[y == k].mean(0).tolist() for k in range(3) if (y == k).any()}
    out["primes"] = PRIMES
    return out


def main():
    bound = 4 if SMOKE else 25
    recs, t0 = [], time.time()
    for a, b, E in curves(bound):
        try:
            recs.append(analyse(a, b, E))
        except Exception as e:  # recorded, not hidden
            recs.append({"a": a, "b": b, "error": str(e)[:200]})
        if len(recs) % 100 == 0:
            print(len(recs), time.time() - t0, flush=True)
    ok = [r for r in recs if "error" not in r]
    proven = [r for r in ok if r["rank_proven"]]
    summ = {"n_curves": len(recs), "n_errors": len(recs) - len(ok), "n_rank_proven": len(proven),
            "rank_distribution_proven": np.bincount([r["rank_lo"] for r in proven]).tolist(),
            "r_an_equals_rank_when_proven": int(sum(r["r_an"] == r["rank_lo"] for r in proven)),
            "r_an_disagrees_when_proven": [(r["a"], r["b"], r["r_an"], r["rank_lo"]) for r in proven if r["r_an"] != r["rank_lo"]],
            "undecided_rank_cases": int(len(ok) - len(proven)),
            "r_an_within_bounds_when_undecided": int(sum(r["rank_lo"] <= r["r_an"] <= r["rank_hi"] for r in ok if not r["rank_proven"]))}
    sh = [r for r in proven if "sha_dist_to_square" in r]
    summ["sha_checked"] = len(sh)
    summ["sha_max_dist_to_square"] = max((r["sha_dist_to_square"] for r in sh), default=None)
    summ["sha_values_counts"] = {str(v): int(sum(r["sha_nearest_square"] == v for r in sh))
                                 for v in sorted({r["sha_nearest_square"] for r in sh})}
    summ["conductor_range"] = [min(r["conductor"] for r in ok), max(r["conductor"] for r in ok)]
    summ["seconds"] = time.time() - t0
    out = {"summary": summ, "learning": learn(ok)}
    print(json.dumps(out["summary"], indent=1), json.dumps({k: v for k, v in out["learning"].items() if "murm" not in k and k != "primes"}), flush=True)
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=float))
    (HERE / "curves.json").write_text(json.dumps(recs))


if __name__ == "__main__":
    main()
