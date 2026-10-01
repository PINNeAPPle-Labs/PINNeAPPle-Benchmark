# CURRENT_STATE — PINNeAPPle-Benchmark

Snapshot: 2026-09-28.

- **Git**: `main` at commit `ee7e5f7` ("Move finished experiments (05, 07, 08) here from
  PINNeAPPle-Climate (#1)"), on top of `a812911` (initial README+LICENSE). No open PRs, no open issues
  (checked at write time). No CI configured.
- **Contents**: `experiments/05_thermal_channel_twin`, `experiments/07_pdr_thesis_reproduction`,
  `experiments/08_supersonic_shock_train`, each with a final, real `paper/*.pdf` built from real results.
  `experiments/paperkit/paperkit.py` vendored.
- **Nothing is running in this repo right now** — all the live background jobs (`02`, `03`, `04`,
  `06_millennium/p_vs_np`) are in the sibling `PINNeAPPle-Climate` repo. See that repo's
  `.agent/CURRENT_STATE.md` for their live status; this repo just waits for them.
- **No benchmark protocol/harness exists yet** — see `ARCHITECTURE.md` and `DECISIONS.md` for what the
  strategic plan wants this to become vs. what's actually here.

## What to check when picking this up

1. `git log origin/main` / `gh pr list` in this repo — has anything landed since this was written?
2. `PINNeAPPle-Climate/.agent/CURRENT_STATE.md` — are `02`/`03`/`04`/`06_millennium/p_vs_np` finished yet?
   If so, they (and their papers) are the next thing to move here — see `TODO.md`.
