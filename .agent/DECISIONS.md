# DECISIONS — PINNeAPPle-Benchmark

## Why this repo exists at all, and why public (2026-09-27/28)

The org-wide strategy document (`helm/docs/estrategia-infraestrutura-aberta.md`) recommended exactly one
new repository out of the whole "Open Physics AI Infrastructure" strategy: a public benchmark repo,
because (a) `PINNeAPPle-arena` (the existing benchmark-runner app) is private and is an application, not a
citable artifact; (b) benchmark protocols/reference data/results version at a different cadence than the
core library and shouldn't bloat it; (c) a separate public repo lets results be cited by version
("PINNeAPPle-Benchmark v1, case 02") independent of the library's own release cycle. This repo was created
on GitHub (org `PINNeAPPle-Labs`, public) on 2026-09-27, with a first README+LICENSE commit, following that
recommendation and explicit user confirmation to create it.

## Why the move happened the way it did (2026-09-28)

- **Only finished experiments were moved.** Four experiments (`02`, `03`, `04`, and the `p_vs_np` part of
  `06_millennium`) were still running as background processes in `PINNeAPPle-Climate` at move time. Their
  scripts hold open, path-based file handles (`log()`/`save()` helpers reopen files by an absolute `Path`
  computed once at script start — see `PINNeAPPle-Climate/.agent/ARCHITECTURE.md`). Moving a directory
  those processes are actively re-opening files in, mid-run, would have broken them the next time they
  tried to write. Verified before and after each move: the four PIDs (`9256`, `9258`, `9271`, `40672`)
  were confirmed still alive immediately after the `mv`.
- **`06_millennium` was moved as a whole-directory decision, deferred entirely**, rather than moving its
  six finished sub-projects and leaving `p_vs_np` behind: they share two root-level files
  (`common_classifier.py`, `relearn.py`) that `p_vs_np`'s own script imports, and splitting the directory
  would have broken those imports for the still-running job.
- **Large/raw data was deliberately excluded from git**, not just left as an afterthought: before
  committing, staged file sizes were checked (`git diff --cached --name-only | xargs du -ch`) to confirm
  nothing near GitHub's 50 MB/100 MB thresholds landed in the repo, and a `.gitignore` was written *before*
  the first `git add` for `data/`, `results_smoke/`, `*.pt` — informed by measuring `07_pdr_thesis_
  reproduction/data` at 5.2 GB (the public RIDI dataset — re-downloadable, should never be in git) and
  per-experiment `.pt` checkpoints up to ~81 MB.
- **`paperkit.py` was copied, not turned into a shared package or submodule.** Reasoning: this repo is
  public; `PINNeAPPle-Climate` (which still needs the same file for `01_sst_satellite_forecasting`, which
  did not move) is presumably private/internal. A cross-repo import or submodule would make this public
  repo's build depend on a private sibling repo. Trade-off accepted: the two copies can drift; whoever
  improves `paperkit.py` in one place should manually copy the change to the other (there is no sync
  mechanism — noted as a real, accepted downside, not solved).
- **Conflict resolution on merge**: none needed for this repo specifically (it was brand new, so the PR
  merged cleanly) — contrast with `PINNeAPPle-CFD`, where the equivalent move-related PR did hit a real
  conflict; see that repo's own `DECISIONS.md` if relevant.

## Not yet decided

- Whether/how to build the actual protocol layer (fixed splits, standard metrics, a `pinneapple-benchmark
  run <case>` command) that the strategy doc envisions — this repo today is just a landing spot for
  finished experiments, not yet a benchmark suite. See `TODO.md`.
- Whether `02`/`03`/`04`/`06_millennium` (once their jobs finish) move here as-is or get restructured to
  fit a future protocol layer.
