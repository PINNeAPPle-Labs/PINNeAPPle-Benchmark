# CURRENT_STATE — PINNeAPPle-Benchmark

**Update 2026-10-17**: experiment 09 follow-ups on branch `exp/09-followup`: `reproduce.sh --smoke` verified from
scratch; the airfoil PINN budget study now reaches 4 h for both PINNs (potential-flow PINN: velocity error 0.59 at
100 s → 0.99 at 4 h on the held-out shape; it gets worse with training). The 4 h runs were resumed from checkpoints
after three container restarts (logs `results/bench09_run_4h*.log`). The budget-study figure needs PINNeAPPle PR #803
(it plotted Cd instead of each method's own metric).

**Update 2026-10-14**: experiment `09_equal_budget_pinn_fno_deeponet` (PINNeAPPle issue #299) is finished on branch
`exp/09-equal-budget`: every run is in `results/runs/`, the report in `results/table.md` and the README. It is the
first experiment built on the library's benchmark harness (`pp.benchmark`, PINNeAPPle PR #662): `src/bench09.py` is
its configuration and the only code here; the benchmark protocol layer the note below says was missing now exists in
the library. See `.agent/FAILED_APPROACHES.md` items 1-4 for what went wrong on the way.

**Update 2026-10-02**: all eight experiments are now here and finished. `main` is at `7a65884` ("Move the
remaining finished experiments (02, 03 code+papers, 04, 06) here (#2)"), on top of `2fd39f4` ("readmes") and
`ee7e5f7` (PR #1, the first move). No open PRs, no running jobs anywhere in this repo or in
`PINNeAPPle-Climate`. See `experiments/README.md`'s table for the complete list of experiments and papers.

Two real bugs were found and fixed while finishing `04` and `06/p_vs_np` (same root cause, found
independently in each): `PINNeAPPle`'s `ModifiedMLP` defaults to a Random-Fourier-Feature bandwidth
(`sigma=1.0`) tuned for low-dimensional input, which silently broke training (not a subtle accuracy loss —
worse than a trivial baseline) on higher-dimensional inputs. Fixed to `sigma=0.1` above 2 input dimensions
in both places. The broken first attempt is kept in each paper's record, not hidden — see
`experiments/README.md`'s closing section and `.agent/DECISIONS.md`.

**Still not built**: the actual benchmark protocol layer (fixed splits/metrics/compute-budget per case,
one reproducible command) the strategy doc (`helm/docs/estrategia-infraestrutura-aberta.md`) envisions —
what's here is eight finished research experiments with papers, not yet a benchmark suite with a shared
harness. See `TODO.md`'s "Real strategic work" section, unchanged by this update.

---

*(Original snapshot below, from the first move on 2026-09-28 — superseded by the above.)*

Snapshot: 2026-09-28.

- **Git**: `main` at commit `ee7e5f7`... (see above for the current commit).
- **Contents at that time**: only `05`, `07`, `08`.
- **Nothing running in this repo at that time** — the live jobs were all in `PINNeAPPle-Climate`.
