"""Control experiment for the spacing-predictability probe: run the identical pipeline on
(a) unfolded eigenvalues of a GUE random matrix (same universality class conjectured for zeta zeros),
(b) a Poisson process (independent spacings). Results are stored next to the zeta result."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import riemann_experiment as R

rng = np.random.default_rng(1)
N = 3000
A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
Hm = (A + A.conj().T) / 2
ev = np.linalg.eigvalsh(Hm) / np.sqrt(N)            # semicircle on [-2, 2]
x = np.clip(ev / 2, -1, 1)
cdf = 0.5 + (x * np.sqrt(1 - x ** 2) + np.arcsin(x)) / np.pi   # semicircle CDF
unf = N * cdf
mid = unf[int(0.1 * N): int(0.9 * N)]
s_gue = np.diff(mid)
s_poi = rng.exponential(1.0, len(s_gue))
out = {"gue_eigenvalues": R.predictability(s_gue), "poisson": R.predictability(s_poi), "n_spacings": int(len(s_gue))}
open("gue_control.json", "w").write(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1))
