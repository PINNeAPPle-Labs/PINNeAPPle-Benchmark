# CURRENT_STATE — PINNeAPPle-Benchmark

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
