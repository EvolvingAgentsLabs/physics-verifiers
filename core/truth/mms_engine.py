"""The generator: unlimited physics tasks with exact truth.

Salari & Knupp, *Code verification by the method of manufactured solutions*,
Sandia SAND2000-1444. Choose a solution, choose coefficients, **derive** the
forcing that makes them consistent, and you have a problem nobody has seen whose
answer you know exactly.

This is the repository's central asset, and the reason is narrower than it first
looked. `experiments/judge_vs_physics/` measured an LLM judge twice against this
material and it detected as well as the verifier did — blatant fabrications and
subtle numerical defects alike. **A model can grade a task. It can also solve
one. It cannot create one with a known answer.** Task supply with exact truth is
the scarce thing, and it is what this module is.

## What it spans

| axis | values |
|---|---|
| operator | steady reaction-diffusion, advection-reaction-diffusion, wave |
| solution | odd quarter-wave modes × an even envelope, weights sampled |
| coefficients | two families, contrast growing with difficulty |
| length | five exact rationals |
| difficulty | 1–3, **measured** monotone in a fixed solver's error |

The axes are sampled independently, so the space is their product.
`scripts/generate_corpus.py` measures throughput and collisions rather than
asserting them.

## The boundaries hold by construction

`u(0) = 0` and `u'(L) = 0` are not imposed afterwards — every piece already
satisfies them:

* `sin((2n-1) pi x / 2L)` is zero at 0 and flat at L.
* `cos(2 pi m x / L)` is flat at **both** ends, so multiplying by it preserves
  both conditions. That is what lets the solution library grow without a
  projection step the candidate would have to guess at.

## The factory verifies itself, per instance, through the code it ships

`self_check` runs the **rendered** `mu`, `K` and `F` — the ones a candidate will
actually receive — against the `u_m` the verifier holds, numerically, at random
interior points. Plus the boundary conditions symbolically, which are cheap.

That is a stronger claim than the symbolic residual it replaced. The symbolic
version proved only that sympy is consistent with itself; a code-generation bug —
a `pycode` translation dropping a term, a `math.` that should have been `np.` —
was invisible to it and fatal to the task.

It has already earned its keep twice. With a float length, `u'(L)` came out as
`2.396*pi*cos(0.5*pi)`, which sympy will not evaluate to zero, and the instance
was rejected at generation rather than scoring a correct candidate as wrong.

## Cost is part of the contract, and it was the binding constraint

`Rational(x).limit_denominator(10**6)` on a sampled float, three of them in a
trigonometric expression, **did not finish in ten minutes**. And `simplify` on
the forcing cost **14.9 seconds** for a wave instance at difficulty 3 — a
generator whose unit cost is fifteen seconds is not a supply of anything.

Denominators stay small, lengths come from a fixed rational set, and the forcing
is `expand`ed rather than `simplify`ed. Expanded is also **less revealing**:
`simplify` factors `F` back toward the shape of `u_m`, which is the structure the
candidate is supposed to have to invert.

Measured after all three: **0.05 s per task**, published by
`scripts/generate_corpus.py` rather than asserted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import sympy as sp

X = sp.Symbol("x", real=True, positive=True)
T = sp.Symbol("t", real=True, nonnegative=True)

#: Denominators stay small on purpose — see the module docstring.
_DENOM = 24

#: Lengths, as exact rationals, so `pi x / 2L` cancels at `x = L` symbolically.
LENGTHS = (sp.Rational(1, 2), sp.Rational(3, 4), sp.Integer(1), sp.Rational(5, 4), sp.Rational(3, 2))

#: `(u, mu, K, b) -> ` the left-hand side of `L[u] = F`.
#:
#: `advection` breaks the operator's symmetry, and that is what makes it harder:
#: a scheme that is second order for a self-adjoint operator is not
#: automatically second order once a first-derivative term is present, and
#: centred differencing of that term is the classic place to lose stability.
OPERATORS: dict[str, Callable[..., sp.Expr]] = {
    "reaction_diffusion": lambda u, mu, K, b: -sp.diff(K * sp.diff(u, X), X) + mu * u,
    "advection_reaction_diffusion": (
        lambda u, mu, K, b: -sp.diff(K * sp.diff(u, X), X) + b * sp.diff(u, X) + mu * u
    ),
    "wave": lambda u, mu, K, b: mu * sp.diff(u, T, 2) - sp.diff(K * sp.diff(u, X), X),
}

TIME_DEPENDENT = frozenset({"wave"})
DIFFICULTIES = (1, 2, 3)


@dataclass(frozen=True)
class Manufactured:
    """The coefficients and forcing that go out, and the solution that stays."""

    operator: str
    u_expr: sp.Expr
    mu_expr: sp.Expr
    K_expr: sp.Expr
    b_expr: sp.Expr
    length: sp.Rational
    difficulty: int
    code: str
    f_expr: sp.Expr
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def time_dependent(self) -> bool:
        return self.operator in TIME_DEPENDENT

    def u(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        expr = self.u_expr.subs(T, sp.Rational(float(t)).limit_denominator(_DENOM)) if self.time_dependent else self.u_expr
        return np.asarray(sp.lambdify(X, expr, "numpy")(x), dtype=float) * np.ones_like(x)


def _rational(value: float) -> sp.Rational:
    return sp.Rational(value).limit_denominator(_DENOM)


def sample_length(rng: np.random.Generator) -> sp.Rational:
    return LENGTHS[int(rng.integers(0, len(LENGTHS)))]


def _mode(n: int, length: sp.Rational) -> sp.Expr:
    """`sin((2n-1) pi x / 2L)`: zero at 0, flat at L."""
    return sp.sin((2 * n - 1) * sp.pi * X / (2 * length))


def _envelope(m: int, length: sp.Rational) -> sp.Expr:
    """`cos(2 pi m x / L)`: flat at both ends, so it preserves both conditions.

    An envelope with a non-zero slope at `L` would break `u'(L) = 0`, and the
    self-check would reject the instance — correctly, and every time.
    """
    return sp.cos(2 * sp.pi * m * X / length)


def _sample_solution(rng, length: sp.Rational, difficulty: int) -> tuple[sp.Expr, dict]:
    n_modes = {1: 2, 2: 3, 3: 4}[difficulty]
    weights = [_rational(round(float(w), 3)) for w in rng.uniform(0.4, 1.2, size=n_modes)]
    expr = sum(w * _mode(n, length) for n, w in enumerate(weights, start=1))
    meta: dict[str, Any] = {"n_modes": n_modes, "envelope": 0}
    if difficulty >= 2:
        m = int(rng.integers(1, 3))
        amp = _rational(round(float(rng.uniform(0.2, 0.6)), 3))
        expr = expr * (1 + amp * _envelope(m, length))
        meta["envelope"] = m
    return sp.expand(expr), meta


def _sample_coefficients(rng, length: sp.Rational, difficulty: int):
    """Positive, smooth, with contrast growing with difficulty.

    Contrast is what makes a variable-coefficient problem hard: a `K` that barely
    varies is a constant-coefficient problem with extra steps.
    """
    hi = {1: 0.5, 2: 1.5, 3: 3.0}[difficulty]
    a = _rational(round(float(rng.uniform(0.2, hi)), 3))
    c = _rational(round(float(rng.uniform(0.2, hi)), 3))
    family = int(rng.integers(0, 2))
    K = 1 + a * X**2 if family == 0 else 1 + a * sp.sin(sp.pi * X / (2 * length)) ** 2
    mu = 1 + c * sp.cos(sp.pi * X / (2 * length)) ** 2 if family == 0 else 1 + c * X
    return K, mu, {"coefficient_family": family, "contrast": float(a)}


def manufacture(
    rng: np.random.Generator,
    length: sp.Rational | float,
    *,
    operator: str = "reaction_diffusion",
    difficulty: int = 1,
) -> Manufactured:
    """Sample a solution and coefficients; derive the forcing; keep the solution."""
    if operator not in OPERATORS:
        raise ValueError(f"unknown operator {operator!r}; known: {', '.join(OPERATORS)}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty is one of {DIFFICULTIES}")

    L = length if isinstance(length, sp.Basic) else _rational(length)
    u_expr, meta = _sample_solution(rng, L, difficulty)
    K_expr, mu_expr, cmeta = _sample_coefficients(rng, L, difficulty)
    meta.update(cmeta)

    b_expr = sp.Integer(0)
    if operator == "advection_reaction_diffusion":
        b_expr = _rational(round(float(rng.uniform(0.3, 1.5)), 3))
        meta["advection"] = float(b_expr)

    if operator in TIME_DEPENDENT:
        omega = _rational(round(float(rng.uniform(0.5, 2.0)), 3))
        u_expr = u_expr * sp.cos(omega * T)
        meta["omega"] = float(omega)

    # `expand`, not `simplify`. Two reasons, and the second is the better one.
    #
    # Cost: `simplify` on a wave instance at difficulty 3 took **14.9 seconds**,
    # and this runs per task. A generator whose unit cost is fifteen seconds is
    # not a supply of anything.
    #
    # And the tidier form is not wanted: `simplify` factors the forcing into
    # something closer to the shape of `u_m`, which is exactly the structure the
    # candidate is supposed to have to invert. Expanded is both cheaper and less
    # revealing.
    lhs = OPERATORS[operator](u_expr, mu_expr, K_expr, b_expr)
    F_expr = sp.expand(lhs)

    return Manufactured(
        operator=operator,
        u_expr=u_expr,
        mu_expr=mu_expr,
        K_expr=K_expr,
        b_expr=b_expr,
        length=L,
        difficulty=difficulty,
        code=_render(mu_expr, K_expr, b_expr, F_expr, operator),
        f_expr=F_expr,
        meta=meta,
    )


def _render(mu, K, b, F, operator: str) -> str:
    def py(expr):
        return sp.pycode(expr).replace("math.", "np.")

    lines = ["import numpy as np", "", f"def mu(x):\n    return {py(mu)}", "", f"def K(x):\n    return {py(K)}"]
    if operator == "advection_reaction_diffusion":
        lines += ["", f"def b(x):\n    return {py(b)} + 0 * x"]
    sig = "def F(x, t):" if operator in TIME_DEPENDENT else "def F(x):"
    lines += ["", f"{sig}\n    return {py(F)}"]
    return "\n".join(lines)


def self_check(m: Manufactured, rng: np.random.Generator | None = None) -> str | None:
    """`None` if the instance is sound, else why not. Runs on every one.

    The residual is checked **numerically, through the rendered code**, at random
    interior points — not symbolically.

    The symbolic version proved only that sympy is consistent with itself, and it
    cost up to 15 seconds. This one is a stronger claim for a fraction of the
    price: it validates the `mu`, `K` and `F` a candidate will actually receive
    against the `u_m` the verifier holds. A code-generation bug — a `pycode`
    translation that drops a term, a `math.` that should have been `np.` — is
    invisible to the symbolic check and fatal to the task, and this catches it.
    """
    rng = rng or np.random.default_rng(0)
    namespace: dict[str, Any] = {}
    try:
        exec(compile(m.code, "<manufactured>", "exec"), namespace)  # noqa: S102
    except SyntaxError as exc:
        return f"the rendered coefficient code does not parse: {exc}"

    length = float(m.length)
    xs = np.asarray(sorted(rng.uniform(0.08 * length, 0.92 * length, size=9)), dtype=float)
    ts = [0.0, 0.37, 0.81] if m.time_dependent else [0.0]

    for t in ts:
        u = sp.lambdify((X, T), m.u_expr, "numpy") if m.time_dependent else sp.lambdify(X, m.u_expr, "numpy")
        lhs_expr = OPERATORS[m.operator](m.u_expr, m.mu_expr, m.K_expr, m.b_expr)
        lhs_fn = (
            sp.lambdify((X, T), lhs_expr, "numpy") if m.time_dependent else sp.lambdify(X, lhs_expr, "numpy")
        )
        lhs = np.asarray(lhs_fn(xs, t) if m.time_dependent else lhs_fn(xs), dtype=float) * np.ones_like(xs)
        rhs = np.asarray(
            namespace["F"](xs, t) if m.time_dependent else namespace["F"](xs), dtype=float
        ) * np.ones_like(xs)
        scale = max(float(np.max(np.abs(lhs))), 1.0)
        err = float(np.max(np.abs(lhs - rhs)) / scale)
        if not np.isfinite(err) or err > 1e-10:
            return f"the rendered forcing disagrees with the operator at t={t}: {err:.3e}"
        _ = u

    spatial = m.u_expr.subs(T, 0) if m.time_dependent else m.u_expr
    if sp.simplify(spatial.subs(X, 0)) != 0:
        return "the manufactured solution violates u(0) = 0"
    slope = sp.simplify(sp.diff(spatial, X).subs(X, m.length))
    if sp.simplify(slope) != 0:
        return f"the manufactured solution violates u'(L) = 0: slope {slope}"
    return None


def evaluate(expr: sp.Expr, x: np.ndarray) -> np.ndarray:
    return np.asarray(sp.lambdify(X, expr, "numpy")(x), dtype=float) * np.ones_like(x)
