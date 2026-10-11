# physics-verifiers

> **Paused** since 2026-08-16 — kept readable, not developed. The lab's active projects are lora-kernel, agentvcs and gene-evidence: [evolvingagentslabs.github.io](https://evolvingagentslabs.github.io).

**A generator of physics tasks with exact truth.** Choose a solution, choose
coefficients, derive the forcing that makes them consistent — and you have a
problem nobody has seen whose answer you know to machine precision. Verification
comes with it, because the answer was known before the question was asked.

> A model can grade a task. It can also solve one. **It cannot create one with a
> known answer.**

Apache 2.0 · [spec](PHYSICS-VERIFIERS-SPEC.md) · [interface notes](docs/INTERFACE-NOTES.md)

## Why the generator and not the verifier

This repository was built to argue that physics verifiers resist reward hacking
where LLM judges do not. **We measured that twice and it is not true**, at least
on this material:

| | scripted fabrications | subtle numerical defects |
|---|---|---|
| LLM judge | 0/12 got past | 0/9 got past |
| physics | 0/12 got past | 0/9 got past |

The judge did not merely mark the defects down — it **named** them: *"the boundary
treatment at the free end fails to halve the control volume"*, *"misses the factor
of 2 in the noise variance"*. Those are two of the four defects this repository
made in its **own reference solutions**.

Full write-up, including the pre-registration written before the first API call:
[`experiments/judge_vs_physics/`](experiments/judge_vs_physics/RESULTS.md).

What survives never depended on the judge being bad — and the largest part of it
is not verification at all. A judge can tell you an answer is wrong. It cannot
hand you ten thousand fresh problems whose answers are already known.

## The generator, measured

```bash
python scripts/generate_corpus.py --n 300
```

```
generated 180 tasks in 7.4s  ->  24.3 tasks/s
distinct: 180/180  collisions: 0
rejected by the self-check: 0

operator                          d    n    median error of ONE fixed solver
reaction_diffusion                1   25                           8.215e-06
reaction_diffusion                2   18                           4.977e-05
reaction_diffusion                3   17                           6.872e-05
advection_reaction_diffusion      1   18                           1.179e-05
advection_reaction_diffusion      2   14                           4.224e-05
advection_reaction_diffusion      3   28                           6.339e-05
```

Three claims, each a test rather than a sentence:

* **24 tasks/second**, self-check included. It was 0.28 s/task, then 14.9 s once
  the operator library grew — `simplify` on the forcing. Expanding instead, and
  checking numerically, took it to 0.04 s.
* **No collisions.** The axes — operator, solution, coefficients, length,
  difficulty — are sampled independently, so the space is their product.
* **Difficulty is monotone** under one *fixed* second-order solver, in both
  steady families. If it were not, the knob would be a label.

Every instance is self-checked **through the code it ships**: the rendered `mu`,
`K` and `F` are run against the held solution at random points. A symbolic
residual proves only that sympy agrees with itself; a dropped term in code
generation is invisible to it and fatal to the task.

## Also here: five verifier families

They are what makes a generated task scorable, and they stand on their own as a
regression suite. `python examples/run_local_eval.py --all`: **17 tasks, mean
reward 0.9383, none refused at the gate.**

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
