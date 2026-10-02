"""Close the count: where the coarse sign-change search found fewer zeros than N(t), resample finely.

Two zeros closer together than the sampling step produce no sign change of Z(t) (a "Lehmer pair"-type
near-miss). The window-by-window comparison with N(t) (Turing/Backlund count of ALL zeros in the critical
strip) localises the deficit; the window is then resampled 8x finer. Only when every window closes is the
statement "all zeros with 0 < Im s < T lie on the critical line" made.
"""
from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
mp.mp.dps = 15
z = np.load(HERE / "zeros.npz")
zeros = list(z["zeros"])
R = json.loads((HERE / "results.json").read_text())
T = R["verification"]["T"]
edges = np.arange(10.0, T + 1e-9, 100.0)
if edges[-1] < T:
    edges = np.append(edges, T)
report = []
for a, b in zip(edges[:-1], edges[1:]):
    need = int(mp.nzeros(b)) - int(mp.nzeros(a))
    have = sum(a < g <= b for g in zeros)
    if need == have:
        continue
    for factor in (8, 32, 128):
        spacing = 2 * np.pi / np.log(max(a, 20) / (2 * np.pi))
        grid = np.arange(a, b, spacing / (6 * factor))
        vals = np.array([float(mp.siegelz(t)) for t in grid])
        found = [float(mp.findroot(mp.siegelz, (grid[i], grid[i + 1]), solver="illinois"))
                 for i in np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]]
        if len(found) == need:
            break
    new = [g for g in found if all(abs(g - h) > 1e-6 for h in zeros)]
    report.append({"window": [float(a), float(b)], "N_diff": need, "coarse_found": have, "fine_factor": factor,
                   "fine_found": len(found), "added_zeros": new,
                   "min_gap_in_window": float(np.min(np.diff(sorted(found)))) if len(found) > 1 else None})
    zeros = sorted([g for g in zeros if not (a < g <= b)] + found)
    print(report[-1], flush=True)
zeros = np.array(sorted(zeros))
R["verification_refined"] = {"windows_refined": report, "zeros_on_line_found": int(len(zeros)),
                             "N_T_all_zeros_in_strip": R["verification"]["N_T_all_zeros_in_strip"],
                             "all_zeros_on_line_up_to_T": int(len(zeros)) == R["verification"]["N_T_all_zeros_in_strip"]}
np.savez_compressed(HERE / "zeros_refined.npz", zeros=zeros,
                    unfolded=np.array([float(mp.siegeltheta(g) / mp.pi + 1) for g in zeros]))
(HERE / "results.json").write_text(json.dumps(R, indent=1, default=float))
print(R["verification_refined"])
