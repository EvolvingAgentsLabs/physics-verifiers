# Why these rewards cannot be talked into

The question this document answers is the one a sceptical research engineer asks
first: **"how do I know a model cannot get a high reward without solving the
problem?"**

Every answer below ends in a theorem or a measurement, never in "because the
judge is good". Each is a test in this repository, named, so it is a result and
not a claim.

---

## The general shape

Two processes that never coexist with each other's inputs:

| | has | never has |
|---|---|---|
| the runner | numpy, scipy, the candidate | any truth: `core/` is not on its path |
| the verifier | mpmath, sympy, the truth | the candidate's source |

One JSON file crosses. The candidate cannot read the verifier because the
verifier is not in its filesystem, and the verifier cannot be influenced by the
candidate's prose because it never sees it.

`mpmath` and `sympy` are blocked from the runner specifically: they are the
*verifier's own methods*, and a candidate using them would be checked against its
own tool — agreement rather than correctness.
`tests/test_sandbox.py::test_a_candidate_cannot_reach_the_scorer_s_own_tools`
proves it against a control body that scores 1.0 without the import, so the zero
is attributable to the block.

**Honest limit:** the sandbox is a subprocess, not a security boundary. It stops
an accidental download and an honest fetch. A determined escape from in-process
Python is possible and the real boundary is a container — which the hub runtimes
already provide. Saying otherwise would cost more credibility than it buys.

---

## Family 1 — eigenmodes

> *The reward requires the roots of a transcendental equation with a freshly
> sampled parameter. Either you compute them or you do not have them.*

`alpha` is drawn continuously, so the eigenvalues of the exponential profile are
in no table, no textbook and no training set.

The instance also draws a **uniform** profile half the time, where the textbook
quarter-wave formula *is* the answer. Measured
(`test_the_textbook_formula_only_works_where_it_is_the_answer`): a solution that
knows only that formula scores ≥ 0.99 on uniform draws and < 0.05 on exponential
ones. That is the signal a training run should get from a memorised formula —
not a uniform pass, not a uniform zero.

`examples/demo_antihack.py` runs it: an agent that writes a fluent, correct
Sturm-Liouville derivation and returns the wrong formula scores **0.0000 with 44%
error**, its first eigenvalue 1.95 against the true 3.48.

> **Measured, and against our prediction:** an LLM judge shown this class of
> answer was **not** fooled — 0 of 12 scripted fabrications got past it, and it
> named the defect each time
> ([`experiments/judge_vs_physics/RESULTS.md`](../experiments/judge_vs_physics/RESULTS.md)).
> The case for a physics verifier here is cost, determinism, specificity and a
> formal guarantee — not that judges are easy to fool. On the blatant attacks
> tested, they are not.

**Level 3 adds a second gate.** Half the reward is naming the injected defect,
and the injected cause must be the **unique argmax** over keyword hits — a text
reciting all four causes earns nothing
(`test_a_diagnosis_that_names_everything_does_not_earn_the_half`). The first
version asked only "is the cause mentioned", and reciting the bug bank was the
cheapest winning strategy.

---

## Family 2 — conservation

> *Which of these four would your fake output satisfy?*

|   | check | the cheat it kills | what it alone lets through |
|---|---|---|---|
| i | energy is constant | a dissipative integrator | `u = 0` conserves energy perfectly |
| ii | time-reversibility | any numerical damping, however small | a field that ignores the initial data |
| iii | the PDE residual at unseen control points | a smooth invented function | a correct field of a *different* problem |
| iv | the initial data | the above | — |

Measured, each dying by its intended leg:

```
all zeros                  energy perfectly constant  -> initial data
frozen initial state       starts right, energy fine  -> the PDE
correct wave, wrong data   PDE and energy both exact  -> initial data
the leaky integrator       PDE residual 8.4e-3 < 2e-2 -> energy 0.55, reverse 0.55
```

The control points come from a seed **not in the prompt**
(`test_the_control_points_are_not_in_the_prompt`). A candidate that satisfied the
PDE only where it was told would be interpolating, which is a different and much
easier problem.

---

## Family 3 — convergence

> *You cannot fake a slope you did not produce.*

The verifier calls the candidate's solver at grid sizes **it** chooses and fits
how the error falls.

A candidate handed the *exact* eigenvalues in its own source — the strongest
possible cheat — scores **zero**, because zero error at every grid is not order
infinity; it is a solver that never discretised anything
(`test_a_hardcoded_answer_earns_nothing`). A wrong answer dressed with a fake
`1/N²` term also scores zero: the error against the truth flattens at the offset
instead of falling.

---

## Family 4 — stochastic

> *Being too good is a failure mode.*

This is the family that answers the open problem the market patches with LLM
judges. The reward does not compare trajectories — it compares estimators against
their own sampling distribution:

    z = (theta_hat - theta_exact) / SE(M, T)

An honest simulation lands at |z| ≈ 1 and is **not penalised for it**: one
standard error is free, because that is what a standard error means.

A candidate that skips simulating and returns the exact variance has `z = 0.00` —
better than any honest run — and dies on **inter-replicate spread**: replicates
that agree perfectly did not come from a simulation. Measured,
`z_spread = -4.69`, `fabrication_suspected = True`
(`test_the_exact_answer_with_no_scatter_is_caught_as_fabrication`).

A candidate that adds fabricated noise to produce the right spread gets the
variance and the scatter right and the **correlation time** wrong: white noise
does not decay. Also measured, also zero.

---

## Family 5 — MMS

> *Recovering the solution from the forcing is the task.*

`F` is derived from a manufactured `u_m` and reveals its structure. Inverting
`F -> u_m` means solving `-(K u')' + mu u = F`, which is the problem — and a model
that does it symbolically rather than numerically has done the work, which is the
ability worth rewarding. There is no third route.

`u_m` never leaves the verifier: it lives in a process-local table keyed by the
hash that ships. The adapter **refuses** to serve this family through a dataset
row for exactly that reason, rather than serving it with a verifier that would
return zero for everyone.

---

## The check that covers a family nobody has written yet

`tests/test_no_truth_travels.py` recomputes each task's truth independently and
asserts it appears in no prompt and — the part a review is likely to miss — in no
`params`. `TaskInstance` is what an adapter serialises to an agent, and `params`
is where an answer would slip through unnoticed, because nothing in it looks like
an answer.

It runs over every family at once, so a sixth inherits it.

---

## What we have not defended against

Stated because a list of defences with no gaps in it is not credible:

- **A container escape.** See the honest limit above.
- **A model that solves the problem and then lies about how.** It gets the
  reward, and it should: the reward is for the answer.
- **Statistical luck.** A fabricated answer can land inside the |z| gate by
  chance; the spread and autocorrelation checks make it unlikely, not impossible.
  The rate is not measured here.
- **A judge at any subtlety beyond the blatant.** `experiments/judge_vs_physics/`
  ran at N=12 and the judge caught every scripted fabrication. Those attacks are
  the easy case and they are ours — the payloads each family was built to kill.
  Whether a judge sees a plausible scheme with a factor of two, or a correct
  scheme with the wrong boundary condition — the kind this repository has caught
  in its **own reference solutions** four times — is untested.
- **An attacker that will actually attack.** 12 of 12 model-attacker replies
  solved the problem instead of cheating, under increasingly explicit
  instructions not to. A defence cannot be tested by an adversary that declines.
