"""Family 1 — EIGENMODES. Closed-form truth, four levels. Spec §3.

The catalogue's front door: simple to explain, impossible to fake.

## The anti-hack argument, in one sentence

The exponential case's eigenvalues are the roots of a transcendental equation
with `alpha` sampled continuously. They are in no table, no textbook and no
training set, and interpolating from memory misses by orders of magnitude more
than the tolerance. **Either you compute them or you do not have them.**

The uniform case *is* memorisable — `(2n-1)pi c / 2L` is in every textbook — and
that is why level 1 samples both and never says which one it handed you. A
solution that only knows the quarter-wave formula scores on half the instances,
which is exactly the signal a training run should get.
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

NAME = "eigenmodes"

#: Tolerances per level, from spec §3. `soft_score` reaches 1.0 at or below these.
TOL = {1: 1e-6, 2: 1e-4, 3: 1e-4, 4: 1e-2}

#: The bug bank for level 3. Each is a real defect with a real signature, and the
#: diagnosis is matched against the *category*, never an exact string — a model
#: that says "the boundary node owns a full cell" and one that says "first-order
#: Neumann" have found the same thing.
BUGS: dict[str, dict[str, Any]] = {
    "neumann_order": {
        "keywords": ("neumann", "free end", "half cell", "half a cell", "ghost", "apex", "first order", "first-order"),
        "hint": "the error is concentrated in the high modes and falls only as O(dx)",
    },
    "coefficient_at_nodes": {
        "keywords": ("midpoint", "mid-point", "face", "interface", "cell wall", "averag"),
        "hint": "K is sampled where it should not be",
    },
    "grid_offset": {
        "keywords": ("off-by-one", "off by one", "spurious", "dirichlet", "extra mode", "offset", "shift"),
        "hint": "every eigenvalue is the next one up",
    },
    "asymmetric_matrix": {
        "keywords": ("symmetr", "hermitian", "self-adjoint", "generalis", "generaliz", "mass matrix"),
        "hint": "the operator being solved is not the one the problem states",
    },
}


def _soft(err: float, tol: float) -> float:
    """`exp(-err/tol)`, clipped to [0, 1]. Spec §1.2's standard form.

    Continuous on purpose: a solver at 1e-3 and one at 1e-12 are both "fail"
    under a threshold, and nothing tells the first which way to move.
    """
    if not math.isfinite(err) or err < 0:
        return 0.0
    return float(min(1.0, max(0.0, math.exp(-err / tol))))


def _profile_is_uniform(params: dict[str, Any]) -> bool:
    return params["profile"] == "uniform"


def _truth_omegas(params: dict[str, Any]) -> list[float]:
    if _profile_is_uniform(params):
        return uni.omegas(params["n_modes"], params["c"], params["length"])
    return expo.omegas(params["n_modes"], params["alpha"], params["length"], params["c"])


# --- generation -------------------------------------------------------------


@dataclass(frozen=True)
class Eigenmodes:
    name: str = NAME

    def generate(self, level: int, rng: np.random.Generator) -> TaskInstance:
        if level not in (1, 2, 3, 4):
            raise ValueError(f"level must be 1..4, got {level}")
        params = self._sample(level, rng)
        truth = _truth_omegas(params) if level != 4 else params["targets"]
        return TaskInstance(
            family=NAME,
            level=level,
            prompt=PROMPTS[level](params),
            params=params,
            seed=int(rng.bit_generator.state["state"]["state"] % (2**31)),
            manifest=attest.manifest(NAME, level, params, 0, truth),
        )

    def verify(self, task: TaskInstance, output: AgentOutput) -> Verdict:
        return VERIFIERS[task.level](task, output)


def _sample_common(rng: np.random.Generator) -> dict[str, Any]:
    """Spec §3's ranges. `profile` is a coin flip and is never stated as a hint."""
    profile = "uniform" if rng.random() < 0.5 else "exponential"
    return {
        "profile": profile,
        "length": round(float(rng.uniform(0.5, 2.0)), 6),
        "c": round(float(rng.uniform(0.5, 3.0)), 6),
        "alpha": round(float(rng.uniform(0.5, 6.0)), 6) if profile == "exponential" else 0.0,
        "n_modes": int(rng.integers(5, 16)),
    }


def _sample(level: int, rng: np.random.Generator) -> dict[str, Any]:
    p = _sample_common(rng)
    if level == 3:
        p["bug"] = list(BUGS)[int(rng.integers(0, len(BUGS)))]
    if level == 4:
        # The targets are the eigenvalues of a hidden profile **computed by the
        # same forward model the verifier uses**, so an exact solution exists.
        #
        # The first version took them from the exponential closed form, which
        # has K varying too. Level 4 fixes K constant, so those eigenvalues are
        # not reachable by any mu at all: the reference scored 0.0 with 24-52%
        # error and the task was unsolvable by construction. Targets that come
        # from a different model than the verifier's are targets nobody can hit.
        p["profile"] = "inverse"
        p["n_modes"] = 5
        p["alpha"] = round(float(rng.uniform(0.5, 3.0)), 6)
        x = np.linspace(0.0, p["length"], 801)
        hidden = np.exp(-p["alpha"] * x)
        p["targets"] = solve_profile(hidden, p["length"], p["c"] ** 2, 5)
    return p


Eigenmodes._sample = staticmethod(_sample)  # type: ignore[attr-defined]


# --- prompts ----------------------------------------------------------------

_SCHEMA = (
    "Reply with a single Python module in one ```python block and nothing else. "
    "You may use numpy, scipy and the standard library. `mpmath` and `sympy` are "
    "not available. Do not try to look up the answer: it is computed in a "
    "separate process that never sees your code."
)


def _describe(p: dict[str, Any]) -> str:
    if p["profile"] == "uniform":
        return f"mu(x) = 1 and K(x) = {p['c'] ** 2:.6f} (both constant)"
    return (
        f"mu(x) = exp(-{p['alpha']:.6f} x) and K(x) = {p['c'] ** 2:.6f} exp(-{p['alpha']:.6f} x), "
        "so the wave speed is constant and the impedance is not"
    )


def _prompt_l1(p: dict[str, Any]) -> str:
    return f"""A taut string of length L = {p['length']:.6f} is fixed at x = 0 and free at x = L:

    d/dx[ K(x) du/dx ] = -w^2 mu(x) u,     u(0) = 0,   du/dx(L) = 0

with {_describe(p)}.

Define:

    def eigenfrequencies(n_modes, length, c, alpha):
        ...
        return {{"omegas": [...]}}

returning the first `n_modes` angular eigenfrequencies in increasing order.
You will be called with n_modes={p['n_modes']}, length={p['length']:.6f},
c={p['c']:.6f}, alpha={p['alpha']:.6f}.

Relative error better than {TOL[1]:.0e} scores full marks. Note that alpha may be
zero, in which case the profile is uniform.

{_SCHEMA}"""


def _prompt_l2(p: dict[str, Any]) -> str:
    return f"""Implement a Sturm-Liouville eigenvalue solver for a fixed-free string:

    d/dx[ K(x) du/dx ] = -w^2 mu(x) u,     u(0) = 0,   du/dx(L) = 0

Your solver must work for a general smooth positive mu(x) and K(x), not only for
the case below. It will be called with {_describe(p)} and L = {p['length']:.6f}.

    def eigenfrequencies(n_modes, length, c, alpha):
        ...
        return {{"omegas": [...]}}

n_modes={p['n_modes']}. Relative error better than {TOL[2]:.0e} scores full marks,
which a second-order scheme reaches on a fine grid or with Richardson
extrapolation.

Two details decide whether you get second order at all: the free end owns **half
a cell**, and K belongs at the cell faces rather than at the nodes.

{_SCHEMA}"""


_SOLVER_TEMPLATE = """
import numpy as np
from scipy.linalg import eigh_tridiagonal

def eigenfrequencies(n_modes, length, c, alpha):
    N = 2000
    dx = length / N
    x = np.arange({start}, N + {start}) * dx
    faces_lo = _K(x - dx / 2, c, alpha)
    faces_hi = _K(x[:-1] + dx / 2, c, alpha)
{k_line}
    volume = np.full(N, dx)
{volume_line}
    d = np.empty(N)
    d[:-1] = (faces_hi + faces_lo[:-1]) / dx
    d[-1] = faces_lo[-1] / dx
    e = -faces_hi / dx
    m = _mu(x, alpha) * volume
    inv = 1.0 / np.sqrt(m)
{solve_line}
    return {{"omegas": np.sqrt(np.maximum(lam, 0.0)).tolist()}}

def _K(x, c, alpha):
    return (c ** 2) * np.exp(-alpha * x)

def _mu(x, alpha):
    return np.exp(-alpha * x)
"""

_CORRECT = {
    "start": 1,
    "k_line": "",
    "volume_line": "    volume[-1] = dx / 2.0",
    "solve_line": (
        "    lam, _ = eigh_tridiagonal(d * inv * inv, e * inv[:-1] * inv[1:],\n"
        "                              select=\"i\", select_range=(0, n_modes - 1))"
    ),
}

#: One injected defect per category. **The code shown matches the sampled bug**,
#: which the first version of this module got wrong: it always showed the same
#: two-defect solver while grading the diagnosis against a randomly chosen
#: category, so a correct explanation scored 0.5 about half the time. The bug
#: bank was decorative and the reward was measuring the sample.
_INJECT = {
    # The apex gets a full cell: O(dx) in the total mass, so O(dx) in every
    # eigenvalue, worst in the high modes.
    "neumann_order": {"volume_line": "    # volume[-1] = dx / 2.0"},
    # K read at the nodes instead of the faces.
    "coefficient_at_nodes": {
        "k_line": "    faces_lo = _K(x, c, alpha)\n    faces_hi = _K(x[:-1], c, alpha)"
    },
    # The Dirichlet node is left in as an unknown. It adds a spurious mode below
    # the fundamental, so every reported eigenvalue is the *next* one up.
    #
    # The first attempt at this category shifted the grid to start at x = 0,
    # which moves 2000 unknowns by one cell and changes the answer by 3.1e-6 —
    # **below the 1e-4 tolerance**. A broken solver that passes the numeric gate
    # is not a task: the candidate has nothing to detect and the reward measures
    # whether it wrote a diagnosis at all.
    "grid_offset": {"solve_line": (
        "    lam, _ = eigh_tridiagonal(d * inv * inv, e * inv[:-1] * inv[1:],\n"
        "                              select=\"i\", select_range=(1, n_modes))"
    )},
    # Symmetrisation dropped, so the generalised problem is solved as if M were
    # the identity.
    "asymmetric_matrix": {
        "solve_line": (
            "    lam, _ = eigh_tridiagonal(d, e,\n"
            "                              select=\"i\", select_range=(0, n_modes - 1))"
        )
    },
}


def broken_solver(bug: str) -> str:
    """The level-3 code, with exactly the sampled defect and no other."""
    if bug not in _INJECT:
        raise ValueError(f"unknown bug {bug!r}; known: {', '.join(_INJECT)}")
    return _SOLVER_TEMPLATE.format(**{**_CORRECT, **_INJECT[bug]})


def _prompt_l3(p: dict[str, Any]) -> str:
    return f"""The solver below returns eigenvalues with several percent of error, worst
in the high modes. Diagnose the cause and fix it.

```python{broken_solver(p["bug"])}```

Return **both** the corrected function and your diagnosis:

    def eigenfrequencies(n_modes, length, c, alpha):
        ...
        return {{"omegas": [...], "diagnosis": "one or two sentences naming the cause"}}

It will be called with n_modes={p['n_modes']}, length={p['length']:.6f},
c={p['c']:.6f}, alpha={p['alpha']:.6f}. Relative error better than {TOL[3]:.0e}
scores full marks, **and** the diagnosis must name the cause — a fix that works
without an explanation scores half.

{_SCHEMA}"""


def _prompt_l4(p: dict[str, Any]) -> str:
    targets = ", ".join(f"{t:.6f}" for t in p["targets"])
    return f"""Inverse problem. Design a smooth positive mass profile mu(x) on
[0, {p['length']:.6f}] such that the fixed-free string

    d/dx[ K du/dx ] = -w^2 mu(x) u,     u(0) = 0,   du/dx(L) = 0

with constant K = {p['c'] ** 2:.6f} has its first five angular eigenfrequencies
within 1% of:

    {targets}

Return mu sampled on a uniform grid of 801 points over [0, L], endpoints
included:

    def profile(length, n_points):
        ...
        return {{"mu": [...]}}          # length n_points, all strictly positive

You will be called with length={p['length']:.6f}, n_points=801. Your profile is
solved by an independent solver and the resulting eigenvalues compared against
the targets. A solution exists.

{_SCHEMA}"""


PROMPTS = {1: _prompt_l1, 2: _prompt_l2, 3: _prompt_l3, 4: _prompt_l4}


# --- verification -----------------------------------------------------------


def _omegas_from(output: AgentOutput) -> list[float] | None:
    values = output.data.get("omegas")
    if not isinstance(values, list) or not values:
        return None
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return None


def _hard_gate(got: list[float], want_n: int) -> str | None:
    """Real, positive, increasing, and the right number of them.

    Spec §3's gate. It is not a formality: a solver that returns a sorted list of
    the *same* eigenvalue, or one that leaks a negative from a non-symmetric
    matrix, is broken in a way the relative error alone would score partial
    credit for.
    """
    if len(got) != want_n:
        return f"expected {want_n} eigenvalues, got {len(got)}"
    if any(not math.isfinite(v) for v in got):
        return "an eigenvalue is not finite"
    if any(v <= 0 for v in got):
        return "an eigenvalue is not positive, so the operator was not positive definite"
    if any(b <= a for a, b in zip(got, got[1:])):
        return "the eigenvalues are not strictly increasing"
    return None


def _score_omegas(task: TaskInstance, got: list[float]) -> tuple[float, dict[str, Any]]:
    want = _truth_omegas(task.params)
    errs = [abs(g - w) / abs(w) for g, w in zip(got, want)]
    worst = max(errs)
    return _soft(worst, TOL[task.level]), {
        "max_relative_error": worst,
        "per_mode_relative_error": errs,
        "tolerance": TOL[task.level],
    }


def _verify_values(task: TaskInstance, output: AgentOutput) -> Verdict:
    got = _omegas_from(output)
    if got is None:
        return failed("no `omegas` list in the output", schema_error=True)
    why = _hard_gate(got, task.params["n_modes"])
    if why:
        return failed(why)
    soft, breakdown = _score_omegas(task, got)
    return Verdict(reward=soft, hard_gate=True, breakdown=breakdown, evidence={"omegas": got})


def diagnosis_names(bug: str, text: str) -> tuple[bool, dict[str, int]]:
    """Did this text diagnose `bug`, or recite the bug bank?

    **The injected cause must be the unique argmax** over keyword hits, not
    merely present. The first version asked `any(keyword in text)`, and a text
    listing all four causes earned the diagnosis half — the cheapest possible
    strategy, and one this repo's own anti-gaming test caught immediately.

    Argmax rather than exclusivity because the vocabularies genuinely overlap:
    a correct Neumann diagnosis says "half a cell", and "half" also belongs to
    the coefficient bug. Requiring zero overlap would fail honest answers.
    """
    hits = {name: sum(1 for k in spec["keywords"] if k in text) for name, spec in BUGS.items()}
    best = max(hits.values())
    winners = [name for name, n in hits.items() if n == best]
    return (best > 0 and winners == [bug]), hits


def _verify_diagnosis(task: TaskInstance, output: AgentOutput) -> Verdict:
    """Level 3: half the reward is the fix, half is naming the cause.

    Split rather than gated, because they are different abilities and a training
    signal that conflates them teaches neither. A fix with no explanation is a
    lucky edit; an explanation with no fix is a guess.
    """
    base = _verify_values(task, output)
    if not base.hard_gate:
        return base
    text = str(output.data.get("diagnosis", "")).lower()
    named, hits = diagnosis_names(task.params["bug"], text)
    reward = 0.5 * base.reward + 0.5 * (base.reward if named else 0.0)
    return Verdict(
        reward=reward,
        hard_gate=True,
        breakdown={
            **base.breakdown,
            "diagnosis_named_the_cause": named,
            "bug": task.params["bug"],
            "keyword_hits": hits,
        },
        evidence={"diagnosis": text[:400]},
    )


def _verify_inverse(task: TaskInstance, output: AgentOutput) -> Verdict:
    """Level 4: solve the agent's profile ourselves and compare to the targets.

    The forward solve here is *our* numerics, not a truth lookup — there is no
    closed form for an arbitrary profile. It is the same solver `reference.py`
    uses, which is fine: the claim under test is "does this profile have those
    eigenvalues", and both sides answering with the same well-tested method is
    the correct arrangement for that claim.
    """
    mu = output.data.get("mu")
    if not isinstance(mu, list) or len(mu) != 801:
        return failed("expected `mu` as a list of 801 samples", schema_error=True)
    try:
        arr = np.asarray([float(v) for v in mu], dtype=float)
    except (TypeError, ValueError):
        return failed("`mu` is not numeric", schema_error=True)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        return failed("a mass profile must be finite and strictly positive")

    got = solve_profile(arr, task.params["length"], task.params["c"] ** 2, 5)
    want = task.params["targets"]
    errs = [abs(g - w) / abs(w) for g, w in zip(got, want)]
    worst = max(errs)
    return Verdict(
        reward=_soft(worst, TOL[4]),
        hard_gate=True,
        breakdown={"max_relative_error": worst, "achieved": got, "targets": want, "tolerance": TOL[4]},
        evidence={"achieved": got},
    )


VERIFIERS = {1: _verify_values, 2: _verify_values, 3: _verify_diagnosis, 4: _verify_inverse}


def solve_profile(mu_samples: np.ndarray, length: float, K: float, n_modes: int) -> list[float]:
    """Forward solve for an arbitrary sampled mu, constant K. Finite volume.

    Used by level 4 to evaluate what the agent proposed. The apex owns half a
    cell — the detail that decides whether this is second order, and the one the
    level-3 bug bank breaks on purpose.
    """
    mu_nodes = np.asarray(mu_samples, dtype=float)
    n_grid = mu_nodes.size
    dx = length / (n_grid - 1)
    # Unknowns are the interior and apex nodes; u(0) = 0 is eliminated.
    mu_u = mu_nodes[1:]
    n = mu_u.size

    volume = np.full(n, dx)
    volume[-1] = dx / 2.0
    d = np.empty(n)
    d[:-1] = 2.0 * K / dx
    d[-1] = K / dx
    e = np.full(n - 1, -K / dx)
    m = mu_u * volume

    from scipy.linalg import eigh_tridiagonal

    inv = 1.0 / np.sqrt(m)
    lam, _ = eigh_tridiagonal(
        d * inv * inv, e * inv[:-1] * inv[1:], select="i", select_range=(0, n_modes - 1)
    )
    return [float(v) for v in np.sqrt(np.maximum(lam, 0.0))]


def run_in_sandbox(task: TaskInstance, source: str, *, timeout_s: float = 120.0) -> AgentOutput:
    """Execute a candidate for this task and return its output.

    The entrypoint and kwargs come from the level, so a candidate is called
    exactly as its prompt promised.
    """
    if task.level == 4:
        entry, kwargs = "profile", {"length": task.params["length"], "n_points": 801}
    else:
        entry = "eigenfrequencies"
        kwargs = {
            "n_modes": task.params["n_modes"],
            "length": task.params["length"],
            "c": task.params["c"],
            "alpha": task.params["alpha"],
        }
    result = sandbox.run_candidate(source, entry, kwargs, timeout_s=timeout_s)
    if not result.ok:
        return AgentOutput(data={"_error": result.error}, raw=source)
    return AgentOutput(data=result.data, raw=source)
