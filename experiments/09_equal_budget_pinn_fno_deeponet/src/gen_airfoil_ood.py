"""Out-of-distribution airfoil runs, with the campaign's exact OpenFOAM setup (pinneapple_design.aero.case.run_case,
Re 4e6, the same O-grid), written like generate_dataset.py's runs into data/airfoil_ood/runs/.

    python gen_airfoil_ood.py

Twelve cases, fixed before any model saw them:
* shapes outside the training box of the CST weights (geometry.BOUNDS): NACA 0021, 2421 (thickness), 4418 (thickness,
  aft camber), 6409, 6412 (camber), each at alpha = 2 and 6 deg (inside the training range of angles);
* incidence below the training range (-2 to 14 deg): NACA 2412 and NACA 0012 at alpha = -4 deg.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import DATA, LIB, log

sys.path.insert(0, str(LIB))
from pinneapple_design.aero.case import read_internal, run_case  # noqa: E402
from pinneapple_design.aero.geometry import BOUNDS, PARAMS, naca4  # noqa: E402

OUT = DATA / "airfoil_ood"
CASES = [(c, a) for c in ("0021", "2421", "4418", "6409", "6412") for a in (2.0, 6.0)] + [("2412", -4.0), ("0012", -4.0)]
RE = 4e6


def one(case):
    code, alpha = case
    shape = naca4(code)
    rid = f"NACA{code}_a{alpha:+03.0f}"
    work = str(OUT / "cases" / rid)
    r = run_case(work, shape, alpha, reynolds=RE)
    outside = [PARAMS[i] for i in range(6) if not BOUNDS[i, 0] <= shape[i] <= BOUNDS[i, 1]]
    rec = {"id": rid, "set": "ood", "shape_id": f"NACA {code}", "shape": shape.tolist(), "alpha": alpha, "reynolds": RE,
           "outside_box": outside, "alpha_outside_range": not -2.0 <= alpha <= 14.0,
           **{k: v for k, v in r.items() if k != "time_dir"}}
    d = r.get("time_dir")
    if d:
        p, U, nut = (read_internal(os.path.join(d, f)) for f in ("p", "U", "nut"))
        np.savez_compressed(OUT / "runs" / f"{rid}.npz", p=p.astype(np.float32), Ux=U[:, 0].astype(np.float32),
                            Uy=U[:, 1].astype(np.float32), nut=nut.astype(np.float32))
    (OUT / "runs" / f"{rid}.json").write_text(json.dumps(rec))
    shutil.rmtree(work, ignore_errors=True)
    return rid, r.get("status"), r.get("seconds")


if __name__ == "__main__":
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    todo = [c for c in CASES if not (OUT / "runs" / f"NACA{c[0]}_a{c[1]:+03.0f}.json").exists()]
    with ProcessPoolExecutor(4) as ex:
        for res in ex.map(one, todo):
            log(*res)
