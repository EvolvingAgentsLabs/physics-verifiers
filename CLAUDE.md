# Working in physics-verifiers (for coding agents)

`PHYSICS-VERIFIERS-SPEC.md` is the single source of truth. **Read it before
writing code.** These are its §1 rules plus what building it has already taught.

## The rules

1. **The truth never travels with the task.** The agent sees `prompt`; the
   verifier holds the truth. `core/truth/` and the prompts live in separate
   modules, and no prompt states a value from which the answer can be
   interpolated.
2. **`core/` imports no hub interface.** Not tidiness — `docs/INTERFACE-NOTES.md`
   records that two `verifiers` APIs are live and only one installs. Adapters
   translate; the physics does not move when that resolves.
3. **`reward = hard_gate * soft_score`.** A continuous score refines a binary
   check, never replaces it. `Verdict` refuses to be constructed with a failed
   gate and a non-zero reward.
4. **Nothing is memorisable.** Every instance samples its parameters and computes
   its truth on the fly. A static dataset is a dataset to memorise.
5. **The verifier never imports the agent's code.** Two processes with disjoint
   dependency sets: the runner has numpy and scipy and no truth; the scorer has
   mpmath and sympy and no candidate.
6. **Two independent routes per truth**, and route B must be able to disagree
   with route A. A route that restates the formula confirms a typo in it.
7. **Tests first**, one commit per component, and no commit without them.
8. **English in code, prompts and docs.**

## What building it has already taught

**State every convention in the prompt.** The ancestor of this repo lost a
correct solution to the factor of two between `<xi xi> = 2D delta` and
`= D delta` — relative error exactly 1.0. An environment that rewards guessing
the author's convention measures telepathy, not physics.

**Pick the threshold's shape before its value.** An absolute residual bound
failed on the eighth mode and passed on the first, because both terms scale with
the root. A bound that tightens with a parameter is a bound on that parameter in
disguise.

**A wrong reason is a defect even when the conclusion is right.** Three
docstrings said `tan` has a pole inside every bracket. It does not — they sit at
the endpoints. The conclusion (use the entire form) was right; the reason a
reader would have used was wrong, and only writing the assertion out found it.

**A check that cannot fail is not a check.** `test_the_crosscheck_can_actually_fail`
exists because two routes that secretly share an implementation agree perfectly
and prove nothing.

**Mark claims `[read]` or `[ran]`.** Step 0's finding is `[ran]`: `pip install`,
then read the installed source. It contradicts every tutorial.

## Running it

```bash
.venv/bin/python -m pytest tests/ -q
```

The three tests that define the product (spec §2.2): `test_reference_agent` (a
correct solution scores ~1), `test_antigaming` (trivial outputs score ~0),
`test_truth_crosscheck` (the two routes agree).
