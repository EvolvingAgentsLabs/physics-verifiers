"""Family 2 — CONSERVATION. The verifier is a theorem. Spec §3.

The flagship of the pitch. The candidate integrates the undamped string from
given initial data and returns the trajectory; the verifier checks three things,
and **the point is that it takes all three**.

## The trio, and why each one alone is not enough

|   | check | the cheat it kills | the cheat it lets through |
|---|---|---|---|
| i | energy is constant | a dissipative integrator that "looks fine" | any constant field: `u = 0` conserves energy perfectly |
| ii | time-reversibility | anything with numerical damping, however small | a field that ignores the initial data and returns to it |
| iii | the PDE residual at random control points | a smooth invented function | a correct field with the wrong initial data |

Add the initial-data check and no class survives. That table is figure 1 of the
post and it is the answer to the sceptical research engineer: *"which of these
would your fake output satisfy?"*

## Why the residual is evaluated at points the candidate does not know

The control points are drawn from the task's seed and are **not in the prompt**.
A candidate that satisfied the PDE only where it was told to would be doing
interpolation, which is a different and much easier problem than integration.

## Every tolerance sits above its own instrument's floor, measured

The energy is a *continuum* functional estimated from a discrete field, so the
estimator has an error of its own. Measured on an **exact** modal solution with
no integration error at all, at 400 cells:

    excluding the Dirichlet node   1.5e-2      <- the first version's estimator
    including it                   6.5e-5      <- the floor that remains

The exclusion was a real bug — `np.gradient` at the first interior node used a
one-sided difference and missed `u(0) = 0`, exactly where `u_x` is largest — and
it made a symplectic reference look like it was leaking 1.5% of its energy.

What remains, 6.5e-5, is the estimator and not the integrator. **So the energy
tolerance is 1e-3 at every level**, and level 4's discrimination comes from
reversibility instead, which compares fields directly and has no derivative
estimator in it. Promising 1e-8 on a quantity the instrument resolves to 6e-5
would be a criterion nobody could aim at and we could not check.

The tolerances are stated in the prompt. Spec §1.1's rule is that the *truth*
does not travel, not that the *criterion* is a secret.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core import attest, sandbox
from core.contracts import AgentOutput, TaskInstance, Verdict, failed

NAME = "conservation"

#: Per level: energy drift, reversibility error, and the PDE residual gate.
#: `energy` is 1e-3 everywhere because the estimator's floor is 6.5e-5 (see the
#: module docstring). `reverse` tightens with the level because it compares
#: fields and its floor is machine precision.
TOL = {
    1: {"energy": 1e-3, "reverse": 1e-3, "residual": 5e-2},
    2: {"energy": 1e-3, "reverse": 1e-5, "residual": 2e-2},
    3: {"energy": 1e-3, "reverse": 1e-5, "residual": 2e-2},
    4: {"energy": 1e-3, "reverse": 1e-8, "residual": 2e-2},
}

#: The fine window for the PDE residual: a third call, spanning a fraction of a
#: period with many samples.
#:
#: The first version measured the residual on the *same* 24 snapshots the energy
#: uses — 5.8 of them per period. A second time difference over samples that
#: coarse is not an approximation of `u_tt` at all, and it scored a correct
#: symplectic reference at 1.95 against a 1e-2 gate. The check was measuring the
#: sampling rate.
MICRO_PERIODS = 0.02
MICRO_SAMPLES = 33

N_CONTROL = 24

LEAKY_INTEGRATOR = """
import numpy as np

def evolve(n_cells, length, c, n_periods, samples, u0, v0):
    dx = length / n_cells
    x = np.arange(1, n_cells + 1) * dx
    u = np.asarray(u0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()

    period = 4.0 * length / c
    dt = 0.4 * dx / c
    n_steps = max(1, int(n_periods * period / dt))

    def accel(u):
        lap = np.empty_like(u)
        lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
        lap[0] = (u[1] - 2 * u[0]) / dx ** 2
        lap[-1] = (u[-2] - u[-1]) / dx ** 2
        return (c ** 2) * lap

    out_u, out_v, out_t = [], [], []
    keep = max(1, n_steps // max(1, samples - 1))
    for k in range(n_steps + 1):
        if k % keep == 0 and len(out_t) < samples:
            out_u.append(u.tolist()); out_v.append(v.tolist()); out_t.append(k * dt)
        a = accel(u)
        v = v + dt * a
        u = u + dt * v          # <-- semi-implicit in name only: v was already updated
        v *= 0.99995            # <-- a residual damping term nobody asked for
    while len(out_t) < samples:
        out_t.append(n_steps * dt); out_u.append(u.tolist()); out_v.append(v.tolist())
    return {"t": out_t, "u": out_u, "v": out_v, "x": x.tolist()}
"""


def _initial(x: np.ndarray, length: float, seed: int) -> np.ndarray:
    """A smooth field satisfying `u(0)=0`, `u'(L)=0`, built from the odd modes.

    Composed from functions that already satisfy the boundaries, so the initial
    data is admissible by construction rather than by a projection the candidate
    would have to guess.
    """
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.4, 1.0, size=3) * np.array([1.0, 0.5, 0.25])
    out = np.zeros_like(x)
    for n, w in enumerate(weights, start=1):
        out += w * np.sin((2 * n - 1) * np.pi * x / (2 * length))
    return out


@dataclass(frozen=True)
class Conservation:
    name: str = NAME

    def generate(self, level: int, rng: np.random.Generator) -> TaskInstance:
        if level not in (1, 2, 3, 4):
            raise ValueError(f"level must be 1..4, got {level}")
        params = {
            "length": round(float(rng.uniform(0.8, 1.6)), 6),
            "c": round(float(rng.uniform(0.8, 1.5)), 6),
            "n_cells": 400,
            "n_periods": 4 if level < 4 else 20,
            "samples": 24,
            "init_seed": int(rng.integers(1, 2**31)),
            "control_seed": int(rng.integers(1, 2**31)),
            **TOL[level],
        }
        # The truth here is not a number but the initial data the trajectory must
        # start from. Hashing it is what makes "was this scored against the task
        # it was generated for" answerable.
        truth = {"init_seed": params["init_seed"], "control_seed": params["control_seed"]}
        return TaskInstance(
            family=NAME,
            level=level,
            prompt=_prompt(level, params),
            params=params,
            seed=params["init_seed"],
            manifest=attest.manifest(NAME, level, params, params["init_seed"], truth),
        )

    def verify(self, task: TaskInstance, output: AgentOutput) -> Verdict:
        return _verify(task, output)


_SCHEMA = (
    "Reply with a single Python module in one ```python block and nothing else. "
    "numpy and scipy are available; `mpmath` and `sympy` are not."
)


def _prompt(level: int, p: dict[str, Any]) -> str:
    extra = {
        1: "",
        2: "Your scheme must also be second order in dt.",
        3: (
            "The integrator below loses energy. Find why and fix it.\n\n"
            f"```python{LEAKY_INTEGRATOR}```\n"
        ),
        4: (
            "Energy must hold to 1e-8 over 20 periods. That rules out anything with "
            "amplitude error accumulating per step: a symplectic scheme or an exact "
            "modal propagator both qualify, and the verifier does not require either "
            "in particular."
        ),
    }[level]
    return f"""Integrate the undamped wave equation on a fixed-free string:

    mu u_tt = (K u_x)_x,     u(0,t) = 0,   u_x(L,t) = 0

with mu = 1 and K = c^2 constant, c = {p['c']:.6f}, L = {p['length']:.6f}.

The initial displacement `u0` is a sum of odd quarter-wave modes, given to you
sampled on the grid `x_i = i * L / n_cells` for i = 1..n_cells (the Dirichlet
node at x = 0 is not included). `v0` is the initial velocity on the same grid.

**You will be called twice**: once from rest, and once from the final state of
your own first run with the velocity negated. The second call is the
reversibility check, so `v0` is not decoration.

    def evolve(n_cells, length, c, n_periods, samples, u0, v0):
        ...
        return {{"t": [...], "u": [[...], ...], "v": [[...], ...], "x": [...]}}

`u` and `v` are `samples` snapshots of displacement and velocity on the `x` grid,
at the times in `t`, from t=0 to t={p['n_periods']} fundamental periods. You will
be called with n_cells={p['n_cells']}, length={p['length']:.6f}, c={p['c']:.6f},
n_periods={p['n_periods']}, samples={p['samples']}.

{extra}

Three things are checked, and all three must hold:

  1. **Energy** E = 0.5 * integral( v^2 + c^2 u_x^2 ) dx is constant to
     {p['energy']:.0e} relative drift.
  2. **Reversibility** — the verifier calls you again with the final state and
     the velocity negated; the result must return to the initial state to
     {p['reverse']:.0e}.
  3. **The PDE itself** — the residual of mu u_tt - (K u_x)_x is evaluated at
     {N_CONTROL} space-time points chosen from a seed you are not given, and must
     be below {p['residual']:.0e} relative to the field's own scale.

Check 1 alone is satisfied by returning zeros; check 3 alone by any solution of
the PDE with different initial data. Together with the initial-data check, they
have no simultaneous cheat.

{_SCHEMA}"""


# --- the three checks -------------------------------------------------------


def _with_dirichlet(u: np.ndarray, v: np.ndarray, x: np.ndarray):
    """Prepend the `x = 0` node, where `u = v = 0`.

    The returned grid starts at the first *interior* node, so the boundary the
    problem is defined by is not in it. Integrating without it drops the largest
    slope in the field and turns an exact solution into a 1.5% energy leak.
    """
    zeros = np.zeros((u.shape[0], 1))
    return np.hstack([zeros, u]), np.hstack([zeros, v]), np.concatenate([[0.0], x])


def energy(u: np.ndarray, v: np.ndarray, x: np.ndarray, c: float) -> np.ndarray:
    """`0.5 * integral(v^2 + c^2 u_x^2) dx` at each snapshot, including x = 0."""
    u, v, x = _with_dirichlet(u, v, x)
    dx = float(x[1] - x[0])
    ux = np.gradient(u, dx, axis=1)
    return 0.5 * np.trapezoid(v**2 + (c**2) * ux**2, dx=dx, axis=1)


def pde_residual(t: np.ndarray, u: np.ndarray, x: np.ndarray, c: float, rng: np.random.Generator) -> float:
    """Max `|u_tt - c^2 u_xx|` at random interior control points, scaled.

    Second differences in both variables, so it needs three consecutive snapshots
    and three consecutive nodes — the control points are drawn from the interior
    for that reason, not to be lenient.

    Scaled by the field's own second-derivative magnitude rather than by `u`: a
    residual of 1e-3 means nothing without knowing whether `u_tt` is 1 or 1e6.
    """
    if u.shape[0] < 3:
        return float("inf")
    u, _, x = _with_dirichlet(u, np.zeros_like(u), x)
    dt = float(t[1] - t[0])
    dx = float(x[1] - x[0])
    ks = rng.integers(1, u.shape[0] - 1, size=N_CONTROL)
    js = rng.integers(1, u.shape[1] - 1, size=N_CONTROL)

    utt = (u[ks + 1, js] - 2 * u[ks, js] + u[ks - 1, js]) / dt**2
    uxx = (u[ks, js + 1] - 2 * u[ks, js] + u[ks, js - 1]) / dx**2
    scale = np.maximum(np.abs(utt), (c**2) * np.abs(uxx))
    scale = np.where(scale > 0, scale, 1.0)
    return float(np.max(np.abs(utt - (c**2) * uxx) / scale))


def _verify(task: TaskInstance, output: AgentOutput) -> Verdict:
    p = task.params
    for key in ("t", "u", "v", "x"):
        if key not in output.data:
            return failed(f"no `{key}` in the output", schema_error=True)
    try:
        t = np.asarray(output.data["t"], dtype=float)
        u = np.asarray(output.data["u"], dtype=float)
        v = np.asarray(output.data["v"], dtype=float)
        x = np.asarray(output.data["x"], dtype=float)
    except (TypeError, ValueError):
        return failed("the trajectory is not numeric", schema_error=True)

    if u.ndim != 2 or v.shape != u.shape or x.size != u.shape[1] or t.size != u.shape[0]:
        return failed(f"shapes do not agree: t{t.shape} u{u.shape} v{v.shape} x{x.shape}", schema_error=True)
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v)) and np.all(np.isfinite(t))):
        return failed("the trajectory is not finite")
    if u.shape[0] < 3:
        return failed("at least three snapshots are needed to see a second time derivative")

    # --- the initial data. Without this, checks 1-3 are satisfied by *any*
    # --- admissible solution, including one of a different problem.
    u0 = _initial(x, p["length"], p["init_seed"])
    scale = float(np.max(np.abs(u0))) or 1.0
    init_err = float(np.max(np.abs(u[0] - u0)) / scale)
    if init_err > 1e-6:
        return failed("the trajectory does not start from the initial data", initial_error=init_err)

    # --- check 1: energy
    e = energy(u, v, x, p["c"])
    e0 = float(e[0])
    if e0 <= 0:
        return failed("the initial energy is not positive, so the field is trivial")
    drift = float(np.max(np.abs(e - e0)) / e0)

    # --- check 3: the PDE, on the fine window, at points the candidate was
    # --- never told. `run_in_sandbox` supplies the window; without it the check
    # --- is not attempted rather than assumed to pass.
    micro = output.data.get("_micro")
    if micro is None:
        return failed("no fine-resolution window, so the PDE could not be checked")
    rng = np.random.default_rng(p["control_seed"])
    residual = pde_residual(
        np.asarray(micro["t"], dtype=float),
        np.asarray(micro["u"], dtype=float),
        np.asarray(micro["x"], dtype=float),
        p["c"],
        rng,
    )
    if not math.isfinite(residual) or residual > p["residual"]:
        return failed(
            "the field does not satisfy the wave equation at the control points",
            pde_residual=residual,
            tolerance=p["residual"],
            energy_drift=drift,
        )

    reverse_err = output.data.get("_reverse_error")
    soft_e = math.exp(-drift / p["energy"])
    soft_r = 1.0 if reverse_err is None else math.exp(-float(reverse_err) / p["reverse"])
    return Verdict(
        reward=float(min(1.0, soft_e * soft_r)),
        hard_gate=True,
        breakdown={
            "energy_drift": drift,
            "reverse_error": reverse_err,
            "pde_residual": residual,
            "initial_error": init_err,
            "tolerances": {k: p[k] for k in ("energy", "reverse", "residual")},
        },
        evidence={"energy_first_last": [e0, float(e[-1])]},
    )


def run_in_sandbox(task: TaskInstance, source: str, *, timeout_s: float = 240.0) -> AgentOutput:
    """Forward, then backward from the final state with the velocity negated.

    The reverse leg is what kills numerical damping: a scheme that loses a
    thousandth of a percent per period looks perfect on the energy check over
    four periods and cannot find its way home.
    """
    p = task.params
    dx = p["length"] / p["n_cells"]
    x = np.arange(1, p["n_cells"] + 1, dtype=float) * dx
    u0 = _initial(x, p["length"], p["init_seed"])

    base = {
        "n_cells": p["n_cells"],
        "length": p["length"],
        "c": p["c"],
        "n_periods": p["n_periods"],
        "samples": p["samples"],
    }
    zeros = [0.0] * x.size
    fwd = sandbox.run_candidate(
        source, "evolve", {**base, "u0": u0.tolist(), "v0": zeros}, timeout_s=timeout_s
    )
    if not fwd.ok:
        return AgentOutput(data={"_error": fwd.error}, raw=source)

    data = dict(fwd.data)
    try:
        u_end = list(map(float, data["u"][-1]))
        v_end = [-float(w) for w in data["v"][-1]]
    except (KeyError, IndexError, TypeError, ValueError):
        return AgentOutput(data=data, raw=source)  # let `_verify` report the schema problem

    back = sandbox.run_candidate(
        source, "evolve", {**base, "u0": u_end, "v0": v_end}, timeout_s=timeout_s
    )
    if back.ok and isinstance(back.data.get("u"), list) and back.data["u"]:
        returned = np.asarray(back.data["u"][-1], dtype=float)
        scale = float(np.max(np.abs(u0))) or 1.0
        data["_reverse_error"] = float(np.max(np.abs(returned - u0)) / scale)
    else:
        # Reported as "not measured" rather than as a pass. A reverse leg that
        # did not run must not read as a reversible integrator.
        data["_reverse_error"] = float("inf")

    # A third call over a short window with many samples, so a second time
    # difference is actually an approximation of u_tt.
    fine = sandbox.run_candidate(
        source,
        "evolve",
        {**base, "n_periods": MICRO_PERIODS, "samples": MICRO_SAMPLES,
         "u0": u0.tolist(), "v0": zeros},
        timeout_s=timeout_s,
    )
    if fine.ok and isinstance(fine.data.get("u"), list):
        data["_micro"] = {"t": fine.data["t"], "u": fine.data["u"], "x": fine.data["x"]}
    return AgentOutput(data=data, raw=source)
