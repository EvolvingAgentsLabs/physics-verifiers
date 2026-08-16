"""Family 3 — CONVERGENCE. The observed order is not negotiable. Spec §3.

The cheapest environment to run and the densest in "does this model actually
understand numerics". A solver is submitted as a function of the grid size; the
verifier runs it at `N`, `2N` and `4N` and computes

    p = log2( ||e_N|| / ||e_2N|| )

against a truth it holds. There is no partial credit for a plausible-looking
scheme: the slope is what it is.

## The anti-hack argument

**You cannot fake a slope you did not produce.** The verifier calls the
candidate's own solver three times, at grid sizes it chooses, and measures how
the error falls. A candidate that hard-codes the answer has zero error at every
N, which is not order 2 — it is `p = inf`, and the gate refuses a non-finite
slope. A candidate that adds a fake `1/N^2` term to a wrong answer still has the
wrong answer at every N, so the errors do not fall at all.

The one strategy that works is implementing a second-order scheme, which is the
thing being rewarded.

## Why the errors must also *decrease*

The hard gate requires monotone decrease before it computes a slope. Two errors
that happen to sit in a 4:1 ratio give `p = 2` whether or not either is small —
including a solver that is diverging in a way that happens to halve twice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core import attest, sandbox
from core.contracts import AgentOutput, TaskInstance, Verdict, failed
from core.truth import exponential_string as expo
from core.truth import uniform_string as uni

NAME = "convergence"

#: Target order per level. Level 4 asks for fourth order, which forces a compact
#: scheme or a correctly done extrapolation.
TARGET_ORDER = {1: 2.0, 2: 2.0, 3: 2.0, 4: 4.0}

#: Spec §3: `soft = exp(-|p_observed - p_target| / 0.15)`.
ORDER_TOL = 0.15

#: The grids, **per target order**. Four sizes give three ratios, so the slope is
#: a fit rather than one lucky pair.
#:
#: Level 4 uses coarser grids on purpose. Measured: a correct fourth-order solver
#: on 250-2000 cells produces errors 5.1e-10, 3.0e-11, 3.7e-11, 1.6e-10 — it falls
#: and then **rises**, because Richardson reaches the floating-point floor by the
#: second grid and the rest is roundoff. The fitted slope came out at 0.48 for a
#: solver that is exactly right.
#:
#: A measurement whose own precision floor sits above the signal reports the
#: floor. The grids must be coarse enough that discretisation error still
#: dominates at the finest one.
GRIDS_BY_ORDER = {2.0: (250, 500, 1000, 2000), 4.0: (24, 48, 96, 192)}


def grids_for(target_order: float) -> tuple[int, ...]:
    return GRIDS_BY_ORDER[target_order]

#: The level-3 solver: **correct everywhere except the free end**.
#:
#: One line differs from the reference — the apex gets a full cell instead of
#: half — which is an O(dx) error in the total mass and therefore O(dx) in every
#: eigenvalue. That is what "degrades to first order" means, and it is what the
#: prompt claims.
#:
#: The first version of this fixture also sampled K at the nodes and dropped a
#: `dx`, which made it simply **wrong**: relative error 1.000 at every grid, no
#: order at all. A prompt saying "this reports order near 1" over code that
#: reports nothing is a prompt describing something the candidate cannot find.
DEGRADED_SOLVER = """
import numpy as np
from scipy.linalg import eigh_tridiagonal

def eigenfrequencies(n_modes, length, c, alpha, n_cells):
    dx = length / n_cells
    x = np.arange(1, n_cells + 1, dtype=float) * dx
    K = lambda t: (c ** 2) * np.exp(-alpha * t)
    lower, upper = K(x - dx / 2.0), K(x[:-1] + dx / 2.0)

    volume = np.full(n_cells, dx)          # <-- the apex should own HALF a cell

    d = np.empty(n_cells)
    d[:-1] = (upper + lower[:-1]) / dx
    d[-1] = lower[-1] / dx
    e = -upper / dx
    m = np.exp(-alpha * x) * volume

    inv = 1.0 / np.sqrt(m)
    lam, _ = eigh_tridiagonal(d * inv * inv, e * inv[:-1] * inv[1:],
                              select="i", select_range=(0, n_modes - 1))
    return {"omegas": np.sqrt(np.maximum(lam, 0.0)).tolist()}
"""


def _truth(params: dict[str, Any]) -> list[float]:
    if params["alpha"] == 0.0:
        return uni.omegas(params["n_modes"], params["c"], params["length"])
    return expo.omegas(params["n_modes"], params["alpha"], params["length"], params["c"])


def observed_order(errors: list[float], grids: tuple[int, ...]) -> float:
    """Least-squares slope of `log2(error)` against `log2(N)`, negated.

    A fit over all grids rather than one ratio: with three ratios available,
    reporting only the last one throws away the evidence that the trend is a
    trend. The sign is flipped so a *falling* error reads as a positive order,
    which is the convention everyone states orders in.
    """
    n = np.log2(np.asarray(grids, dtype=float))
    e = np.log2(np.asarray(errors, dtype=float))
    return float(-np.polyfit(n, e, 1)[0])


@dataclass(frozen=True)
class Convergence:
    name: str = NAME

    def generate(self, level: int, rng: np.random.Generator) -> TaskInstance:
        if level not in (1, 2, 3, 4):
            raise ValueError(f"level must be 1..4, got {level}")
        params = _sample(level, rng)
        return TaskInstance(
            family=NAME,
            level=level,
            prompt=PROMPTS[level](params),
            params=params,
            seed=0,
            manifest=attest.manifest(NAME, level, params, 0, _truth(params)),
        )

    def verify(self, task: TaskInstance, output: AgentOutput) -> Verdict:
        return _verify(task, output)


def _sample(level: int, rng: np.random.Generator) -> dict[str, Any]:
    # Level 1 is the uniform string, where a naive scheme is already second
    # order. From level 2 the coefficients vary, which is where the boundary
    # treatment starts to matter and the classic failure is falling to order 1.
    alpha = 0.0 if level == 1 else round(float(rng.uniform(0.5, 4.0)), 6)
    return {
        "length": round(float(rng.uniform(0.5, 2.0)), 6),
        "c": round(float(rng.uniform(0.5, 3.0)), 6),
        "alpha": alpha,
        "n_modes": 4,
        "target_order": TARGET_ORDER[level],
    }


_SCHEMA = (
    "Reply with a single Python module in one ```python block and nothing else. "
    "numpy and scipy are available; `mpmath` and `sympy` are not. Your solver is "
    "called three times with different `n_cells`, and the reward is the observed "
    "order of convergence — hard-coding an answer produces no slope at all."
)


def _signature(p: dict[str, Any]) -> str:
    return f"""    def eigenfrequencies(n_modes, length, c, alpha, n_cells):
        ...
        return {{"omegas": [...]}}

with mu(x) = exp(-alpha x), K(x) = c^2 exp(-alpha x), u(0) = 0, du/dx(L) = 0,
n_modes={p['n_modes']}, length={p['length']:.6f}, c={p['c']:.6f},
alpha={p['alpha']:.6f}. `alpha = 0` means both are constant."""


def _prompt_l1(p):
    return f"""Implement a second-order finite-difference eigenvalue solver for the
fixed-free string. It will be called at several grid sizes and scored on the
**observed order of convergence**, which must come out at 2.

{_signature(p)}

{_SCHEMA}"""


def _prompt_l2(p):
    return f"""Implement an eigenvalue solver for the fixed-free string with *variable*
coefficients that achieves second order. The reward is the observed order.

{_signature(p)}

The usual way to lose it: the free end. A boundary treatment that is first order
degrades the whole solution to first order however good the interior is.

{_SCHEMA}"""


def _prompt_l3(p):
    return f"""The solver below reports an observed convergence order near 1 instead of 2.
Find what degrades it and fix it, without losing stability.

```python{DEGRADED_SOLVER}```

Return the corrected function with the same signature:

{_signature(p)}

The reward is the observed order of your corrected solver.

{_SCHEMA}"""


def _prompt_l4(p):
    return f"""Achieve **fourth**-order convergence for the fixed-free string eigenvalue
problem. The reward is the observed order, targeting 4.

{_signature(p)}

A second-order scheme with Richardson extrapolation done correctly qualifies, as
does a compact fourth-order stencil. The extrapolation must be inside your
solver: it is called once per grid size and the order is measured across those
calls.

{_SCHEMA}"""


PROMPTS = {1: _prompt_l1, 2: _prompt_l2, 3: _prompt_l3, 4: _prompt_l4}


def _verify(task: TaskInstance, output: AgentOutput) -> Verdict:
    """`output.data` carries the per-grid errors, computed by `run_in_sandbox`."""
    grids = grids_for(task.params["target_order"])
    errors = output.data.get("errors")
    if not isinstance(errors, list) or len(errors) != len(grids):
        return failed("the solver did not run at every grid size", schema_error=True)
    if any((not isinstance(e, (int, float))) or not math.isfinite(e) for e in errors):
        return failed("an error is not finite")
    if any(e <= 0 for e in errors):
        # Zero error at every grid is what a hard-coded answer looks like. It is
        # not order infinity; it is a solver that never solved anything, and the
        # slope is undefined rather than excellent.
        return failed(
            "an error is zero, so no order can be measured — a solver whose error does not "
            "change with the grid did not discretise anything",
            suspected_hardcoding=True,
        )
    if any(b >= a for a, b in zip(errors, errors[1:])):
        return failed("the error does not decrease with the grid", errors=errors)

    p = observed_order(errors, grids)
    if not math.isfinite(p):
        return failed("the observed order is not finite", errors=errors)
    target = task.params["target_order"]
    soft = float(min(1.0, math.exp(-abs(p - target) / ORDER_TOL)))
    return Verdict(
        reward=soft,
        hard_gate=True,
        breakdown={
            "observed_order": p,
            "target_order": target,
            "errors": errors,
            "grids": list(grids),
            "tolerance": ORDER_TOL,
        },
        evidence={"errors": errors},
    )


def run_in_sandbox(task: TaskInstance, source: str, *, timeout_s: float = 180.0) -> AgentOutput:
    """Run the candidate at every grid and return the error against the truth.

    The candidate never sees the truth or the errors — it is called with a grid
    size and returns eigenvalues. The comparison happens here, outside its
    process, which is what keeps `observed_order` a measurement of its solver
    rather than of its arithmetic.
    """
    want = np.asarray(_truth(task.params), dtype=float)
    errors: list[float] = []
    for n_cells in grids_for(task.params["target_order"]):
        result = sandbox.run_candidate(
            source,
            "eigenfrequencies",
            {
                "n_modes": task.params["n_modes"],
                "length": task.params["length"],
                "c": task.params["c"],
                "alpha": task.params["alpha"],
                "n_cells": int(n_cells),
            },
            timeout_s=timeout_s,
        )
        if not result.ok:
            return AgentOutput(data={"_error": result.error}, raw=source)
        got = result.data.get("omegas")
        if not isinstance(got, list) or len(got) != task.params["n_modes"]:
            return AgentOutput(data={"_error": f"bad output at n_cells={n_cells}"}, raw=source)
        arr = np.asarray([float(v) for v in got], dtype=float)
        # Relative error in the max norm: the high modes are where a boundary
        # defect shows, and an L2 average over four modes would dilute it.
        errors.append(float(np.max(np.abs(arr - want) / np.abs(want))))
    return AgentOutput(data={"errors": errors}, raw=source)
