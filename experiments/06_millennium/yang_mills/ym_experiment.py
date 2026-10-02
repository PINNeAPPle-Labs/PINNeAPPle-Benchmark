"""Yang-Mills existence and mass gap -- lattice Monte Carlo evidence at finite spacing.

SU(2) pure gauge theory, Wilson action S = beta * sum_p (1 - 1/2 Re Tr U_p), on an L^3 x T periodic
lattice. Updates: Kennedy-Pendleton heat bath + over-relaxation, checkerboard-vectorised with links
stored as unit quaternions. Observables:
  * average plaquette;
  * Creutz ratios chi(R) from R x R Wilson loops -> string tension a^2 sigma (confinement);
  * 0++ glueball correlator from APE-smeared spatial plaquettes -> effective mass a*m(t).
Statistical errors from a jackknife over measurement blocks. A finite a*m > 0 at a few couplings is
lattice evidence consistent with a mass gap; it is not a construction of the continuum theory.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SMOKE = os.environ.get("SMOKE") == "1"
DEV = torch.device("cpu")  # measured faster than MPS for this stencil workload
torch.set_num_threads(3)
torch.manual_seed(12345)


# ---------------------------------------------------------------------------- quaternion SU(2)
def qmul(a, b):
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return torch.stack([a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
                     a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
                     a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
                     a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0])


def qdag(a):
    return torch.stack([a[0], -a[1], -a[2], -a[3]])


def shift(x, mu, s):
    """Field value at site n + s*mu_hat (periodic). x has shape (4, T, L, L, L)."""
    return torch.roll(x, -s, dims=1 + mu)


class Lattice:
    def __init__(self, L, T, beta):
        self.L, self.T, self.beta = L, T, beta
        self.shape = (T, L, L, L)
        self.U = torch.zeros((4, 4) + self.shape, device=DEV)   # U[mu] quaternion field
        self.U[:, 0] = 1.0                                         # cold start
        n = torch.tensor(np.indices(self.shape).sum(0), device=DEV)
        self.parity = [(n % 2 == p) for p in (0, 1)]

    def staple(self, mu):
        A = torch.zeros((4,) + self.shape, device=DEV)
        U = self.U
        for nu in range(4):
            if nu == mu:
                continue
            # upper: U_nu(x+mu) U_mu(x+nu)^dag U_nu(x)^dag
            up = qmul(qmul(shift(U[nu], mu, 1), qdag(shift(U[mu], nu, 1))), qdag(U[nu]))
            # lower: U_nu(x+mu-nu)^dag U_mu(x-nu)^dag U_nu(x-nu)
            Unu_m = shift(U[nu], nu, -1)
            lo = qmul(qmul(qdag(shift(Unu_m, mu, 1)), qdag(shift(U[mu], nu, -1))), Unu_m)
            A += up + lo
        return A

    def update(self, n_or=1):
        for mu in range(4):
            for par in self.parity:
                A = self.staple(mu)
                k = torch.sqrt((A ** 2).sum(0))
                V = A / k                                          # SU(2) direction of the staple sum
                m = par
                # heat bath (Kennedy-Pendleton) for x0 with weight sqrt(1-x0^2) exp(beta k x0)
                bk = self.beta * k[m]
                x0 = self._kp(bk)
                r = torch.sqrt(torch.clamp(1 - x0 ** 2, min=0))
                v = torch.randn(3, len(x0), device=DEV)
                v = v / v.norm(dim=0)
                X = torch.cat([x0[None], r * v])
                Unew = qmul(X, qdag(V[:, m]))                     # U = X V^dag
                self.U[mu][:, m] = Unew
            for _ in range(n_or):
                for par in self.parity:
                    A = self.staple(mu)
                    V = A / torch.sqrt((A ** 2).sum(0))
                    Vd = qdag(V[:, par])
                    self.U[mu][:, par] = qmul(qmul(Vd, qdag(self.U[mu][:, par])), Vd)   # U' = V^dag U^dag V^dag
        # re-unitarise against round-off
        self.U /= torch.sqrt((self.U ** 2).sum(1, keepdim=True))

    @staticmethod
    def _kp(bk):
        """Kennedy-Pendleton sampler for SU(2) heat bath, vectorised with rejection."""
        out = torch.empty(len(bk), device=DEV)
        todo = torch.arange(len(bk), device=DEV)
        while len(todo):
            b = bk[todo]
            r1, r2, r3, r4 = torch.rand(4, len(todo), device=DEV).clamp(min=1e-12)
            lam2 = -(torch.log(r1) + torch.cos(2 * np.pi * r2) ** 2 * torch.log(r3)) / (2 * b)
            ok = r4 ** 2 <= 1 - lam2
            out[todo[ok]] = 1 - 2 * lam2[ok]
            todo = todo[~ok]
        return out

    # ------------------------------------------------------------------------ observables
    def plaquette_field(self, mu, nu, U=None):
        U = self.U if U is None else U
        P = qmul(qmul(U[mu], shift(U[nu], mu, 1)), qmul(qdag(shift(U[mu], nu, 1)), qdag(U[nu])))
        return P[0]                                        # 1/2 Re Tr = q0

    def plaquette(self):
        return float(np.mean([float(self.plaquette_field(m, n).mean()) for m in range(4) for n in range(m + 1, 4)]))

    def wilson_loop(self, R, Tt, mu, nu, U):
        """Mean R x Tt loop in the (mu, nu) plane (1/2 Re Tr)."""
        def line(field, d, n):
            out = field.clone()
            cur = field
            for _ in range(n - 1):
                cur = shift(cur, d, 1)
                out = qmul(out, cur)
            return out
        a = line(U[mu], mu, R)
        b = line(U[nu], nu, Tt)
        W = qmul(qmul(a, shift(b, mu, R)), qmul(qdag(shift(a, nu, Tt)), qdag(b)))
        return float(W[0].mean().cpu())

    def ape_spatial(self, alpha=0.5, n=4):
        """APE-smeared spatial links (mu=1,2,3), temporal links untouched."""
        U = self.U.clone()
        for _ in range(n):
            new = U.clone()
            for i in (1, 2, 3):
                S = torch.zeros_like(U[i])
                for j in (1, 2, 3):
                    if j == i:
                        continue
                    S += qmul(qmul(U[j], shift(U[i], j, 1)), qdag(shift(U[j], i, 1)))
                    Uj_m = shift(U[j], j, -1)
                    S += qmul(qmul(qdag(Uj_m), shift(U[i], j, -1)), shift(Uj_m, i, 1))
                M = (1 - alpha) * U[i] + alpha / 4 * S
                new[i] = M / torch.sqrt((M ** 2).sum(0))
            U = new
        return U

    def glueball_op(self):
        U = self.ape_spatial()
        O = sum(self.plaquette_field(i, j, U) for i in (1, 2, 3) for j in (1, 2, 3) if i < j)
        return O.reshape(self.T, -1).sum(1).cpu().double().numpy()   # zero-momentum projection per time slice


def jackknife(samples, fn, n_blocks=20):
    samples = np.asarray(samples)
    blocks = np.array_split(np.arange(len(samples)), n_blocks)
    full = fn(samples)
    jk = np.array([fn(np.delete(samples, b, axis=0)) for b in blocks])
    err = np.sqrt((n_blocks - 1) * ((jk - jk.mean(0)) ** 2).mean(0))
    return full, err


def run(beta, L, T, n_therm, n_meas, every):
    lat = Lattice(L, T, beta)
    t0 = time.time()
    for _ in range(n_therm):
        lat.update()
    plaq, ops, loops = [], [], []
    for s in range(n_meas):
        for _ in range(every):
            lat.update()
        plaq.append(lat.plaquette())
        ops.append(lat.glueball_op())
        loops.append([np.mean([lat.wilson_loop(R, R, mu, nu, lat.U) for mu, nu in ((1, 2), (1, 3), (2, 3), (0, 1))])
                      for R in (1, 2, 3)])
    ops, loops = np.array(ops), np.array(loops)

    def corr(o):
        c = np.array([np.mean([(o[:, t] * o[:, (t + dt) % T]).mean() for t in range(T)]) for dt in range(T // 2 + 1)])
        return c - o.mean() ** 2

    def meff(o):
        c = corr(o)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log(c[:-1] / c[1:])

    m_eff, m_err = jackknife(ops, meff)
    c, c_err = jackknife(ops, corr)
    p, p_err = jackknife(np.array(plaq)[:, None], lambda x: x.mean(0))
    w, w_err = jackknife(loops, lambda x: x.mean(0))
    # string tension estimate from square loops: a^2 sigma ~ -ln(W(3,3) W(1,1) / W(2,2)^2) / ... use chi from squares
    # ln W(R,R) = -sigma R^2 - 4 mu R + c  ->  -ln[W(3)W(1)/W(2)^2] = sigma (9 + 1 - 8) = 2 sigma (perimeter, c cancel)
    sq = lambda x: -np.log(x.mean(0)[2] * x.mean(0)[0] / x.mean(0)[1] ** 2) / 2.0
    sig, sig_err = jackknife(loops, sq)
    return {"beta": beta, "L": L, "T": T, "n_therm": n_therm, "n_meas": n_meas, "sweeps_between": every,
            "seconds": time.time() - t0, "plaquette": [float(p[0]), float(p_err[0])],
            "wilson_square_loops_R123": [w.tolist(), w_err.tolist()],
            "string_tension_a2sigma_from_squares": [float(sig), float(sig_err)],
            "glueball_corr": [c.tolist(), c_err.tolist()], "m_eff": [m_eff.tolist(), m_err.tolist()]}


def main():
    cfg = [(2.3, 4, 8, 20, 30, 2)] if SMOKE else [(2.3, 8, 16, 200, 800, 2), (2.4, 8, 16, 200, 800, 2), (2.5, 8, 16, 200, 800, 2)]
    out = {"runs": []}
    for beta, L, T, nt, nm, ev in cfg:
        r = run(beta, L, T, nt, nm, ev)
        out["runs"].append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("glueball_corr",)}), flush=True)
        (HERE / "results.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
