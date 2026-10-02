"""Hodge conjecture -- what can be computed, and what that does (not) test.

(a) CICY threefolds (Candelas et al. list, 7890 configurations): Euler characteristic computed
    independently from each configuration matrix (Chern classes + intersection theory on products of
    projective spaces) vs. 2 (h11 - h21) from the published Hodge numbers. For threefolds the Hodge
    conjecture is already a theorem (Lefschetz (1,1) + hard Lefschetz), so this is a data/topology
    consistency check, not a test of the conjecture.
(b) Learning h11 from the configuration matrix with a PINNeAPPle ModifiedMLP vs. the "favourable"
    baseline h11 = number of projective factors.
(c) Fermat fourfolds X_m: x0^m + ... + x5^m = 0. Shioda's character description: the primitive middle
    cohomology splits into characters alpha = (a0..a5), a_i in (Z/m)\\{0}, sum a_i = 0; alpha is a Hodge
    class iff |t alpha| = 3 for every t in (Z/m)^x, where |alpha| = sum <a_i>/m. We count Hodge characters
    and how many are pair-decomposable (a_i + a_j = 0 pairings, i.e. spanned by linear-subspace cycles).
    Sanity checks built in: Fermat quartic surface -> 19 primitive Hodge classes (rho = 20), Fermat cubic
    surface -> 6 (rho = 7).
"""
from __future__ import annotations

import itertools
import json
import math
import os
import re
import sys
from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "PINNeAPPle"))
SMOKE = os.environ.get("SMOKE") == "1"
OUT = {}


# ---------------------------------------------------------------------------- (a) CICY
def parse_cicy(path):
    txt = path.read_text()
    out = []
    for block in txt.split("Num    :")[1:]:
        h11 = int(re.search(r"H11\s*:\s*(\d+)", block).group(1))
        h21 = int(re.search(r"H21\s*:\s*(\d+)", block).group(1))
        rows = [list(map(int, r.split(","))) for r in re.findall(r"^\{([\d, ]+)\}\s*$", block, flags=re.M)]
        num_ps = int(re.search(r"NumPs\s*:\s*(\d+)", block).group(1))
        Q = np.array(rows[-num_ps:])
        out.append({"h11": h11, "h21": h21, "Q": Q})
    return out


def euler_characteristic(Q):
    """chi = int_X c3(TX); c(TX) = prod_i (1+J_i)^(n_i+1) / prod_j (1 + sum_i q_ij J_i)."""
    n = Q.sum(1) - 1                                  # Calabi-Yau: sum_j q_ij = n_i + 1
    k = Q.shape[0]
    # degree <= 3 part of the total Chern class as a dict {exponent tuple: coefficient}
    def mul(p, q):
        r = {}
        for a, ca in p.items():
            for b, cb in q.items():
                e = tuple(x + y for x, y in zip(a, b))
                if sum(e) <= 3:
                    r[e] = r.get(e, 0) + ca * cb
        return r
    unit = tuple([0] * k)
    c = {unit: 1}
    for i in range(k):
        base = {unit: 1}
        ei = tuple(1 if t == i else 0 for t in range(k))
        for p in range(1, 4):
            base[tuple(p * x for x in ei)] = math.comb(int(n[i]) + 1, p)
        c = mul(c, base)
    for j in range(Q.shape[1]):
        L = {tuple(1 if t == i else 0 for t in range(k)): int(Q[i, j]) for i in range(k) if Q[i, j]}
        inv = {unit: 1}                               # 1/(1+L) = 1 - L + L^2 - L^3
        pw = {unit: 1}
        for p in range(1, 4):
            pw = mul(pw, L)
            for e, v in pw.items():
                inv[e] = inv.get(e, 0) + (-1) ** p * v
        c = mul(c, inv)
    c3 = {e: v for e, v in c.items() if sum(e) == 3 and v}
    cols = [tuple(Q[:, j]) for j in range(Q.shape[1])]

    @lru_cache(maxsize=None)
    def coef(j, rem):                                  # coefficient of x^rem in prod_{j'>=j} L_j'
        if j == len(cols):
            return 1 if not any(rem) else 0
        tot = 0
        for i, q in enumerate(cols[j]):
            if q and rem[i] > 0:
                r = list(rem); r[i] -= 1
                tot += q * coef(j + 1, tuple(r))
        return tot
    chi = 0
    for e, v in c3.items():
        rem = tuple(int(a - b) for a, b in zip(n, e))
        if min(rem) >= 0:
            chi += v * coef(0, rem)
    return int(chi)


def part_a_b():
    data = parse_cicy(HERE / "cicylist.txt")
    if SMOKE:
        data = data[:300]
    ok, bad = 0, []
    for k, d in enumerate(data):
        chi = euler_characteristic(d["Q"])
        if chi == 2 * (d["h11"] - d["h21"]):
            ok += 1
        else:
            bad.append((k + 1, chi, d["h11"], d["h21"]))
    OUT["cicy_euler_check"] = {"n": len(data), "agree": ok, "disagree_examples": bad[:20], "n_disagree": len(bad)}
    print(OUT["cicy_euler_check"], flush=True)
    # (b) learning h11
    import torch
    from pinneapple_neural.architectures.modified_mlp import ModifiedMLP
    R, C = 12, 15
    X = np.zeros((len(data), R * C), np.float32)
    for i, d in enumerate(data):
        M = np.zeros((R, C)); M[:d["Q"].shape[0], :d["Q"].shape[1]] = d["Q"]
        X[i] = M.ravel()
    y = np.array([d["h11"] for d in data])
    nps = np.array([d["Q"].shape[0] for d in data])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(data)); ntr, nva = int(0.7 * len(perm)), int(0.85 * len(perm))
    tr, va, te = perm[:ntr], perm[ntr:nva], perm[nva:]
    out = {"baseline_h11_equals_num_P_factors_accuracy": float((nps[te] == y[te]).mean())}
    torch.manual_seed(0)
    ncls = int(y.max()) + 1
    m = ModifiedMLP(in_dim=R * C, out_dim=ncls, hidden_dim=256, n_layers=4, n_fourier=64, sigma=0.5)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(X), torch.tensor(y)
    best, state = 1e9, None
    for ep in range(10 if SMOKE else 300):
        for idx in torch.tensor(tr)[torch.randperm(len(tr))].split(256):
            loss = torch.nn.functional.cross_entropy(m(xt[idx]).y, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = float(torch.nn.functional.cross_entropy(m(xt[va]).y, yt[va]))
        if v < best:
            best, state = v, {k: x.clone() for k, x in m.state_dict().items()}
    m.load_state_dict(state)
    with torch.no_grad():
        p = m(xt[te]).y.argmax(1).numpy()
    out["mlp_accuracy"] = float((p == y[te]).mean())
    out["mlp_accuracy_non_favourable_subset"] = float((p[nps[te] != y[te]] == y[te][nps[te] != y[te]]).mean()) if (nps[te] != y[te]).any() else None
    out["n_test"] = int(len(te))
    OUT["learning_h11"] = out
    print(out, flush=True)


# ---------------------------------------------------------------------------- (c) Fermat
def hodge_characters(m, n):
    """Characters of the primitive middle cohomology of the Fermat n-fold of degree m that are Hodge."""
    units = [t for t in range(1, m) if gcd(t, m) == 1]
    d = n // 2
    head = np.stack(np.meshgrid(*[np.arange(1, m)] * (n + 1), indexing="ij"), -1).reshape(-1, n + 1)
    last = (-head.sum(1)) % m
    A = np.concatenate([head, last[:, None]], 1)[last != 0]
    keep = np.ones(len(A), bool)
    for t in units:
        keep &= ((t * A) % m).sum(1) == (d + 1) * m
    return [tuple(int(v) for v in a) for a in A[keep]]


def pair_decomposable(a, m):
    a = list(a)
    if not a:
        return True
    x = a[0]
    for j in range(1, len(a)):
        if (x + a[j]) % m == 0:
            return pair_decomposable(a[1:j] + a[j + 1:], m)
    return False


def part_c():
    checks = {"fermat_quartic_surface_primitive_hodge": len(hodge_characters(4, 2)),
              "expected_quartic": 19,
              "fermat_cubic_surface_primitive_hodge": len(hodge_characters(3, 2)),
              "expected_cubic": 6}
    assert checks["fermat_quartic_surface_primitive_hodge"] == 19 and checks["fermat_cubic_surface_primitive_hodge"] == 6, checks
    rows = []
    for m in range(3, (9 if SMOKE else 21)):
        B = hodge_characters(m, 4)
        dec = sum(pair_decomposable(a, m) for a in B)
        rows.append({"m": m, "hodge_characters": len(B), "pair_decomposable": dec, "not_pair_decomposable": len(B) - dec})
        print(rows[-1], flush=True)
    OUT["fermat_fourfolds"] = {"sanity": checks, "rows": rows}


if __name__ == "__main__":
    part_c()
    (HERE / "results.json").write_text(json.dumps(OUT, indent=1, default=float))
    part_a_b()
    (HERE / "results.json").write_text(json.dumps(OUT, indent=1, default=float))
