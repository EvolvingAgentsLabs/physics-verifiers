# judge_vs_physics — result

**Run 2026-08-16.** Model and judge: `google/gemini-3.5-flash` via OpenRouter.
12 tasks × 3 arms = 36 rows, all transcripts in `results.json`.
The prediction was written and committed in `PREREGISTRATION.md` before the first
API call.

## The headline: the prediction was wrong

|  | honest mean | model-attacker hack rate | scripted-attacker hack rate |
|---|---|---|---|
| **LLM judge** | 0.833 | 92% (11/12) | **0% (0/12)** |
| **physics** | 0.731 | 83% (10/12) | **0% (0/12)** |

`hack rate = P(reward > 0.7)`.

Predicted: judge 30–60%, physics ≈ 0%. **Physics came out as predicted. The judge
did not — it caught every scripted fabrication.**

Not by luck, either. It read the code and said why:

> *"The provided code is a complete fabrication that returns trivially zero
> fields, failing…"* — on the zeros attack
>
> *"The reply generates independent Gaussian noise at each time step, which
> completely fails…"* — on the white-noise attack

`PREREGISTRATION.md` said what to do here, and this is it: **the claim "LLM judges
are soft verifiers" is not supported by this experiment at this scale.**

## The second finding: the model would not cheat

**12 of 12** model-attacker replies used a real numerical method anyway.

Told to maximise its score without solving, it wrote an exact modal expansion and
a matrix-exponential sampler. Told again, with explicit constraints — no scipy,
no linear algebra, no iterative scheme, no analytical solution — it wrote a
Velocity Verlet integrator and scored 0.955.

That is why the scripted arm exists. Without it the physics "hack rate" of 83%
would have been reported as a verifier failure when it is the opposite: the model
solved the problem, and physics correctly paid it.

This is a finding about **the attacker**, not about either scorer, and it limits
what the whole experiment can conclude. An adversary that will not adversary
cannot test a defence.

## What this does and does not support

**Supported, and unchanged by the result:**

* A physics verifier costs **nothing per evaluation** — no second model call, no
  judge tokens. Half the API spend of this run was the judge.
* It is **deterministic**. The same output scores the same forever; a judge's
  score moves with its model, its temperature and its prompt.
* It carries a **formal guarantee**. `0.000` on the zeros attack is not an opinion
  that the field is trivial; it is the initial-data check failing.
* It is **specific**. `breakdown` names which of energy, reversibility, the PDE
  residual or the initial data failed, and by how much. The judge returns a
  number and a sentence. `examples/demo_refinement.py` shows the reward rising
  0.0000 → 0.0019 → 0.8812 because the breakdown said something actionable.

**Not supported by this run:**

* That an LLM judge is easy to fool on this material. At N=12, with these
  attacks, this judge, it was not fooled once.

## Why the judge did well, stated against our own interest

The scripted attacks are **the easy case, and they are ours**. They are the
payloads each family's own tests were built to kill: a zero field, a frozen
state, replicates with no spread, white noise. They are blatant in code, and the
judge was asked to look at code and be sceptical.

A competent adversary would write something that *runs*, that produces
plausible-looking numbers, and whose defect is a factor of two or a boundary
condition — the kind this repository has caught in **its own reference solutions**
four times. Whether a judge sees those is the interesting question and this run
does not answer it.

## Other limits, listed rather than buried

* **N = 12** per arm, not the spec's 30. Stated in the pre-registration in
  advance; adequate for 0/12 versus 11/12, not for a 20% effect.
* **One model as both solver and judge.** A judge scoring output from its own
  family may be unusually good at recognising its patterns — or unusually bad.
  Untested.
* **Two of 36 replies did not finish cleanly**, both honest, both scoring 0.
  That drags the honest physics mean down and is not a verifier result. An
  earlier version had 14 of 36 truncated because this model reasons into a
  separate field; disabling that fixed most of it.
* **The judge cannot execute code.** Modelled deliberately, and a reader may
  fairly object. The answer is that a judge which executes code is a verifier
  with an extra step — and it still needs the physics to know what the output
  should have been.

## What to run next, if this is worth pursuing

The question this run raises is sharper than the one it was built for:
**at what subtlety does a judge stop noticing?** A ladder of attacks — blatant
fabrication, a plausible scheme with a factor of two, a correct scheme with the
wrong boundary condition, a correct scheme at the wrong resolution — scored by
both. Physics is flat across that ladder by construction. If the judge's curve
falls, the original claim is recovered with a number attached; if it does not,
that is worth knowing too.

That is a real experiment and it is not this one.

## Reproducing

```bash
export OPENROUTER_API_KEY=...
python experiments/judge_vs_physics/run.py --n 12
```

Every prompt, reply, verdict and manifest is in `results.json`, including the
attacks. The scripted payloads are in `scripted_attacks.py`.
