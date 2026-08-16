"""Family 5 — MMS. The generator that turns a catalogue into a curriculum.

Every other family is bounded by the handful of problems with closed forms. This
one is not: choose any smooth solution, choose any smooth coefficients, and
*derive* the forcing that makes them consistent. **Every draw is a new problem
with exact truth**, which is what a training run needs and what no hand-built
catalogue can supply.

`core/truth/mms_engine.py` does the manufacturing and verifies each instance
against its own PDE symbolically before it ships.

## The anti-hack argument

`F` reveals the structure of `u_m` but recovering `u_m` from `F` means solving
`-(K u')' + mu u = F` — which is the task. If a model does that symbolically
rather than numerically, it has done the work; there is no third route where it
gets the answer without solving something.

## What the candidate is not told

The manufactured solution never appears in the prompt, in `params`, or in the
manifest — only its hash. `tests/test_no_truth_travels.py` asserts that for every
family, because `TaskInstance` is the object an adapter hands to an agent and
anything inside it is public by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core import attest, sandbox
from core.contracts import AgentOutput, TaskInstance, Verdict, failed
from core.truth import mms_engine as mms

NAME = "mms"

#: Set from a measurement, not from an intuition.
#:
#: A correct second-order finite-volume solve on the 401-point output grid lands
#: at 1.5e-5 to 3.5e-5 — that is the *discretisation* error of any such scheme at
#: that resolution, not a defect. Level 1 first asked for 1e-8 on the reasoning
#: that a correct solve is "essentially exact", and scored the reference 0.0.
#:
#: A tolerance below what a correct solution can reach at the resolution the task
#: states is a tolerance nobody can meet, and it fails the candidate for the
#: task's own arithmetic.
TOL = {1: 1e-3, 2: 1e-4, 3: 1e-3, 4: 0.15}
N_POINTS = 401

BROKEN_MMS_SOLVER = """
import numpy as np

def solve(length, n_points, mu, K, F):
    x = np.linspace(0.0, length, n_points)
    dx = x[1] - x[0]
    n = n_points - 1                      # u(0) = 0 is eliminated
    xi = x[1:]

    lower = K(xi - dx / 2.0)
    upper = K(xi[:-1] + dx / 2.0)

    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = (upper[i] if i < n - 1 else 0.0) + lower[i]
    A /= dx ** 2
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = -upper[i] / dx ** 2

    A += np.diag(mu(xi))                  # <-- the reaction term, unweighted
    u = np.linalg.solve(A, F(xi))
    return {"u": np.concatenate(([0.0], u)).tolist()}
"""


@dataclass(frozen=True)
class MMS:
    name: str = NAME

    def generate(self, level: int, rng: np.random.Generator) -> TaskInstance:
        if level not in (1, 2, 3, 4):
            raise ValueError(f"level must be 1..4, got {level}")
        L = mms.sample_length(rng)
        length = float(L)
        instance = mms.manufacture(rng, L)

        # The factory verifies itself, per instance. A bad draw is caught here
        # rather than by scoring a correct candidate as wrong.
        why = mms.self_check(instance)
        if why:
            raise ArithmeticError(f"the manufactured instance is unsound: {why}")

        params = {
            "length": length,
            "n_points": N_POINTS,
            "code": instance.code,
            "tolerance": TOL[level],
        }
        # `truth` is hashed and dropped. It must not enter `params`.
        truth = {"u": [float(v) for v in instance.u(np.linspace(0.0, length, N_POINTS))]}
        task = TaskInstance(
            family=NAME,
            level=level,
            prompt=_prompt(level, params, instance),
            params=params,
            seed=0,
            manifest=attest.manifest(NAME, level, params, 0, truth),
        )
        _SOLUTIONS[task.manifest["truth_hash"]] = instance
        return task

    def verify(self, task: TaskInstance, output: AgentOutput) -> Verdict:
        return _verify(task, output)


#: Manufactured solutions, held by the verifier and keyed by the hash that ships.
#:
#: A process-local side table rather than a field on the task: `TaskInstance` is
#: the object an adapter serialises to an agent, so anything on it is public.
#: This is the mechanism that makes "the truth does not travel" structural rather
#: than a rule somebody remembers.
_SOLUTIONS: dict[str, mms.Manufactured] = {}


_SCHEMA = (
    "Reply with a single Python module in one ```python block and nothing else. "
    "numpy and scipy are available; `mpmath` and `sympy` are not."
)


def _prompt(level: int, p: dict[str, Any], instance: mms.Manufactured) -> str:
    extra = {
        1: "A straightforward second-order scheme on the output grid is enough here.",
        2: "The tolerance is ten times tighter than level 1. A direct solve on the output grid lands just short of it; solving on a finer grid and interpolating, or extrapolating, closes the gap.",
        3: (
            "The solver below has a defect. Its error against the exact solution is "
            "large and does not fall as the grid refines. Find it and fix it.\n\n"
            f"```python{BROKEN_MMS_SOLVER}```\n"
        ),
        4: (
            "Return the solution **and** the observed order of convergence of your own "
            "scheme, measured against grids of your choosing. A second-order scheme "
            "should report about 2."
        ),
    }[level]
    extra_return = ', "order": float' if level == 4 else ""
    return f"""Solve the steady reaction-diffusion equation on a fixed-free interval:

    -( K(x) u'(x) )' + mu(x) u(x) = F(x),     u(0) = 0,   u'(L) = 0

with L = {p['length']:.6f} and the coefficients and forcing below. They are given
as Python; `F` is not a physical source, it is whatever makes the equation
consistent — solve the equation as stated.

```python
{instance.code}
```

    def solve(length, n_points, mu, K, F):
        ...
        return {{"u": [...]{extra_return}}}

`u` is the solution sampled on `np.linspace(0, length, n_points)`, endpoints
included, so `u[0] = 0`. You will be called with length={p['length']:.6f},
n_points={p['n_points']}, and `mu`, `K`, `F` as the functions above.

{extra}

Your solution is compared against an exact one in the max norm, relative to its
own scale. Relative error below {p['tolerance']:.0e} scores full marks.

{_SCHEMA}"""


def _verify(task: TaskInstance, output: AgentOutput) -> Verdict:
    instance = _SOLUTIONS.get(task.manifest["truth_hash"])
    if instance is None:
        # Refused rather than skipped: a verifier that cannot find its truth must
        # not report a pass.
        return failed("the manufactured solution for this task is not available")

    values = output.data.get("u")
    if not isinstance(values, list) or len(values) != task.params["n_points"]:
        return failed(f"expected `u` with {task.params['n_points']} samples", schema_error=True)
    try:
        got = np.asarray([float(v) for v in values], dtype=float)
    except (TypeError, ValueError):
        return failed("`u` is not numeric", schema_error=True)
    if not np.all(np.isfinite(got)):
        return failed("the solution is not finite")

    x = np.linspace(0.0, task.params["length"], task.params["n_points"])
    want = instance.u(x)
    scale = float(np.max(np.abs(want))) or 1.0
    if abs(float(got[0])) > 1e-9 * scale:
        return failed("the solution does not satisfy u(0) = 0", u0=float(got[0]))

    err = float(np.max(np.abs(got - want)) / scale)
    soft = math.exp(-err / task.params["tolerance"])
    breakdown: dict[str, Any] = {"max_relative_error": err, "tolerance": task.params["tolerance"]}

    if task.level == 4:
        order = output.data.get("order")
        if not isinstance(order, (int, float)) or not math.isfinite(float(order)):
            return failed("level 4 also asks for the observed order", schema_error=True, **breakdown)
        breakdown["order_reported"] = float(order)
        breakdown["order_score"] = math.exp(-abs(float(order) - 2.0) / 0.3)
        soft = math.sqrt(soft * breakdown["order_score"])

    return Verdict(
        reward=float(min(1.0, soft)), hard_gate=True, breakdown=breakdown, evidence={"n": int(x.size)}
    )


def run_in_sandbox(task: TaskInstance, source: str, *, timeout_s: float = 180.0) -> AgentOutput:
    """Prepend the coefficient code, so `mu`, `K` and `F` are in scope.

    They are passed as *names the module defines* rather than as callables
    through JSON, because JSON cannot carry a function. The candidate's `solve`
    receives them as arguments, which is what its prompt promised.
    """
    combined = f"{task.params['code']}\n\n{source}\n"
    wrapper = combined + """

def _entry(length, n_points):
    return solve(length, n_points, mu, K, F)
"""
    result = sandbox.run_candidate(
        wrapper,
        "_entry",
        {"length": task.params["length"], "n_points": task.params["n_points"]},
        timeout_s=timeout_s,
    )
    if not result.ok:
        return AgentOutput(data={"_error": result.error}, raw=source)
    return AgentOutput(data=result.data, raw=source)
