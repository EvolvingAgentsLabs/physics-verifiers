# physics-verifiers

**RL environments whose reward is computed by physics.** Closed-form
eigenvalues, conservation laws, observed convergence order, and statistics with
known sampling error — not an LLM judge and not a hackable test suite.

> You cannot persuade the conservation of energy, you cannot prompt-inject a
> Lyapunov equation, and the observed order of convergence either is 2 or it
> isn't.

Apache 2.0 · [spec](PHYSICS-VERIFIERS-SPEC.md) · [interface notes](docs/INTERFACE-NOTES.md)

## Status

All five families run. `python examples/run_local_eval.py --all` over the bundled
references: **17 tasks, mean reward 0.9383, none refused at the gate.** 135 tests.

| family | what the verifier is | levels |
|---|---|---|
| `eigenmodes` | closed forms, and roots of a transcendental equation | 1–4 |
| `conservation` | energy, reversibility, the PDE residual, the initial data | 1–4 |
| `convergence` | the observed order of convergence | 1–4 |
| `stochastic` | estimators against their own sampling distribution | 1–4 |
| `mms` | a manufactured solution — a **generator**, not a list | 1–4 |

Not yet built: `experiments/judge_vs_physics/`, and the adapter for the
unreleased `verifiers.v1`.

## See it in one command

```bash
python examples/demo_antihack.py
```

Three agents, one generated task:

```
A · honest solver          reward 0.9977   error 2.3e-09
B · fluent and wrong       reward 0.0000   error 4.4e-01
C · malformed              reward 0.0000   refused: no `omegas` list
```

B writes a correct-sounding Sturm-Liouville derivation and returns the textbook
quarter-wave formula, which is wrong on an exponential profile — its first
eigenvalue is 1.95 against the true 3.48.

**We predicted an LLM judge would be fooled by this class of answer. We ran it,
and it was not** — see [`experiments/judge_vs_physics/RESULTS.md`](experiments/judge_vs_physics/RESULTS.md).
The judge caught 12 of 12 scripted fabrications and said why. What survives the
result is narrower and still true: a physics verifier costs nothing per
evaluation, is deterministic, is specific about *which* invariant failed, and
carries a formal guarantee rather than an opinion.

```bash
python examples/demo_refinement.py
```

The reward as a signal rather than a grade: best-of-N selection, then the verdict
fed back. Measured: 0.0000 → 0.0019 → 0.8812.

## Why it cannot be talked into

The one-page answer to a sceptical research engineer is
[`docs/ANTIHACK.md`](docs/ANTIHACK.md), where every claim is a named test —
including the four cheats each leg of the conservation trio kills, and the
fabricated stochastic answer that scores `z = 0.00` on the variance and dies on
inter-replicate spread.

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
