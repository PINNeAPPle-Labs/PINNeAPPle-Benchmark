# SESSION_HANDOFF — PINNeAPPle-Benchmark

Written 2026-09-28 for a fresh agent picking this up with no access to the prior conversation.

## One-paragraph situation

This is a brand-new (2026-09-27) public repo, created after the user reviewed a strategic recommendation
(`helm/docs/estrategia-infraestrutura-aberta.md`) to have exactly one new public repo for a Physics AI
benchmark suite. As of writing it holds three finished, real research experiments (with real papers) moved
in from a sibling private repo, `PINNeAPPle-Climate` — not because they were designed as benchmark cases,
but because they were finished work that didn't belong in a climate-specific repo. It is **not yet** the
protocol-driven benchmark suite the strategy doc describes; that's future work (`TODO.md`).

## Immediate first steps

1. Check `PINNeAPPle-Climate/.agent/CURRENT_STATE.md` (sibling repo) — are the four experiments that were
   still running at move time (`02`, `03`, `04`, `06_millennium/p_vs_np`) finished now? If so, moving them
   here (same pattern as the 05/07/08 move — see `DECISIONS.md`) is the most concrete next action.
2. Otherwise, the real open-ended work is building the actual benchmark protocol layer — read
   `helm/docs/estrategia-infraestrutura-aberta.md` (sibling `helm` repo) for the target design before
   starting; do not invent a protocol independent of that document without checking it first, since the
   user has already reasoned through this once.

## Conventions to preserve

- Same anti-fabrication/honesty conventions as everything moved from `PINNeAPPle-Climate` — see that
  repo's `SESSION_HANDOFF.md` for the full list (every claim traces to `results.json`, failed first
  attempts kept in the record, "not run" never faked, explicit "what this shows / does not show" boxes).
- **Never commit multi-hundred-MB raw data or checkpoints to this repo's git history** — it's public, and
  the `.gitignore` convention (`data/`, `results_smoke/`, `*.pt`) exists specifically to prevent that. Check
  staged file sizes before any commit that adds new experiment data (`git diff --cached --name-only | xargs
  du -ch`).

## Where the rest of the documentation lives

`PINNeAPPle-Climate/.agent/` has the deep architecture/scientific-context documentation for the experiment
code itself (unchanged by the move). `helm/.agent/` (if written) has the org-wide strategy context that
justified creating this repo. `PINNeAPPle/.agent/` has the core library's own handoff.
