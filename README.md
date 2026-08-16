# physics-verifiers

**RL environments whose reward is computed by physics.** Closed-form
eigenvalues, conservation laws, observed convergence order, and statistics with
known sampling error — not an LLM judge and not a hackable test suite.

> You cannot persuade the conservation of energy, you cannot prompt-inject a
> Lyapunov equation, and the observed order of convergence either is 2 or it
> isn't.

Apache 2.0 · [spec](PHYSICS-VERIFIERS-SPEC.md) · [interface notes](docs/INTERFACE-NOTES.md)

## Status — 2026-08-16

Day 1 of 30. What exists is real; what does not is listed.

| | state |
|---|---|
| Step 0 — read the installed `verifiers` | **done**, and it found a live API split |
| `core/contracts.py` | done |
| `core/truth/` — uniform, exponential, OU/Lyapunov | done, two routes each, 9 checks |
| Families 1–5 | not started |
| Adapters | not started |
| `judge_vs_physics` experiment | not started |

## The step-0 finding, because it changes what to build

`pip install verifiers` gives **0.1.5**, whose API is `Environment` / `Rubric` /
`SingleTurnEnv`. The repository's *own* environments are written against
`verifiers.v1` — `Task`, `Runtime`, `@reward`, `run_uv_script` — which **no
released version contains**. Two live APIs, one installable.

So `core/` imports neither, and there will be two adapters. Details, with both
contracts quoted from source, in [`docs/INTERFACE-NOTES.md`](docs/INTERFACE-NOTES.md).

## Prior art, already measured

`ai-os/projects/coclea-sr/environments/` (paused) is a working environment of
four graded physics tasks in this shape:

```
reference   1.0  1.0  1.0  1.0
cheater     0.0  0.0  0.0  0.0
```

The cheating solution tries three real routes — import the truth module,
re-derive with mpmath, read an expected-answer file out of the working directory
— and earns nothing on all four. That repo is the ancestor of this one and it
already paid for three lessons, recorded in [`CLAUDE.md`](CLAUDE.md).

## Running the tests

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
```
