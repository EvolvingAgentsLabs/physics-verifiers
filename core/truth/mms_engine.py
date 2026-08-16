"""The Method of Manufactured Solutions: exact truth for arbitrary coefficients.

Salari & Knupp, *Code verification by the method of manufactured solutions*,
Sandia SAND2000-1444. The idea is a trick and it is the catalogue's multiplier:

1. **Choose** a solution `u_m(x)` you like — any smooth function.
2. **Choose** coefficients `mu(x)`, `K(x)` you like — any smooth positive ones.
3. **Derive** the forcing that makes `u_m` the exact solution:
   `F = -(K u_m')' + ... ` — symbolically, with sympy.
4. Hand the candidate `mu`, `K` and `F`, and hold `u_m`.

There is no restriction to the handful of cases with closed forms. **Every draw
is a new problem with exact truth**, which is what turns seventeen hand-built
tasks into a generator.

## The boundaries are satisfied by construction, not by projection

`u_m` is built from `sin((2n-1) pi x / 2L)`, which already satisfies `u(0) = 0`
and `u'(L) = 0`. Composing from admissible pieces means the manufactured solution
is admissible without anyone checking — a solution that satisfied the PDE and not
the boundaries would be a task with no answer.

## The factory verifies itself

`manufacture()` substitutes `u_m` back into the PDE with the derived `F` and
requires the residual to simplify to **exactly zero**. That runs on every
instance, so a bad draw is caught at generation rather than by scoring a correct
candidate as wrong. It costs a `simplify` per task and it is the difference
between a generator and a generator of nonsense.

## Why this is not hackable, stated for `docs/ANTIHACK.md`

`F` reveals the *structure* of `u_m` but recovering `u_m` from `F` means solving
`-(K u')' + ... = F` — which is exactly the task. And if a model does it
symbolically rather than numerically, that **is** the ability worth rewarding.
There is no third option where it gets the answer without the work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

X = sp.Symbol("x", real=True, positive=True)

#: Denominators stay small on purpose.
#:
#: The first version used `limit_denominator(10**6)`, which turns a sampled float
#: into a rational with a six-digit denominator, and `simplify` over three of
#: those in a trigonometric expression **did not finish in ten minutes**. The
#: factory has to run per instance, so its symbolic cost is a design constraint
#: rather than an implementation detail.
#:
#: Anti-memorisation does not need a continuous length: the solution's weights
#: and both coefficients still vary, so no two draws are the same problem.
_DENOM = 24

#: Lengths, as exact rationals. Sampled, and small enough that `pi x / 2L` at
#: `x = L` cancels to `pi/2` symbolically — which is what makes `u'(L) = 0` hold
#: by construction rather than approximately.
LENGTHS = (sp.Rational(1, 2), sp.Rational(3, 4), sp.Integer(1), sp.Rational(5, 4), sp.Rational(3, 2))


@dataclass(frozen=True)
class Manufactured:
    """One instance: the coefficients and forcing that go out, and the solution
    that stays here."""

    u_expr: sp.Expr
    mu_expr: sp.Expr
    K_expr: sp.Expr
    length: sp.Rational
    #: Python source for `mu`, `K` and `F` as numpy-evaluable functions. This is
    #: what the prompt carries.
    code: str
    #: The residual sympy found when substituting `u` back in. Must be zero.
    residual: sp.Expr

    def u(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(sp.lambdify(X, self.u_expr, "numpy")(x), dtype=float)


def _rational(value: float) -> sp.Rational:
    """A float as an exact rational, so `simplify` can cancel it.

    The length has to be exact in the symbolic expressions. With a float, the
    slope at `x = L` came out as `2.396*pi*cos(0.5*pi)` — sympy will not evaluate
    `cos(0.5*pi)` to zero because `0.5*pi` is a Float times a symbol, not
    `pi/2` — and the factory's own boundary check rejected the instance. Which is
    the check working: it caught an inadmissible task at generation rather than
    scoring a correct candidate as wrong.
    """
    return sp.Rational(value).limit_denominator(_DENOM)


def _basis(n: int, length: sp.Rational) -> sp.Expr:
    """`sin((2n-1) pi x / 2L)` — zero at 0, flat at L, by construction."""
    return sp.sin((2 * n - 1) * sp.pi * X / (2 * length))


def sample_length(rng: np.random.Generator) -> sp.Rational:
    return LENGTHS[int(rng.integers(0, len(LENGTHS)))]


def manufacture(rng: np.random.Generator, length: sp.Rational) -> Manufactured:
    """Sample a solution and coefficients; derive the forcing; verify it.

    The PDE is the steady form `-(K u')' + mu u = F`, which has a unique solution
    for positive `mu` and `K` — unlike `-(K u')' = F`, whose solution is fixed
    only up to the boundary conditions and which would make level 2 ambiguous.
    """
    L = length if isinstance(length, sp.Basic) else _rational(length)
    weights = [float(w) for w in rng.uniform(0.4, 1.2, size=3)]
    u_expr = sum(_rational(round(w, 3)) * _basis(n, L) for n, w in enumerate(weights, start=1))

    # Smooth, strictly positive coefficients. Rational so `simplify` can cancel
    # exactly: a float coefficient leaves 1e-17 residues and the self-check would
    # have to be given a tolerance, which is how a factory stops verifying itself.
    a = _rational(round(float(rng.uniform(0.2, 1.0)), 3))
    b = _rational(round(float(rng.uniform(0.2, 1.0)), 3))
    K_expr = 1 + a * X**2
    mu_expr = 1 + b * sp.cos(sp.pi * X / (2 * L)) ** 2

    F_expr = sp.simplify(-sp.diff(K_expr * sp.diff(u_expr, X), X) + mu_expr * u_expr)
    residual = sp.simplify(-sp.diff(K_expr * sp.diff(u_expr, X), X) + mu_expr * u_expr - F_expr)

    code = "\n".join(
        [
            "import numpy as np",
            "",
            f"def mu(x):\n    return {sp.pycode(mu_expr).replace('math.', 'np.')}",
            "",
            f"def K(x):\n    return {sp.pycode(K_expr).replace('math.', 'np.')}",
            "",
            f"def F(x):\n    return {sp.pycode(F_expr).replace('math.', 'np.')}",
        ]
    )
    return Manufactured(
        u_expr=u_expr, mu_expr=mu_expr, K_expr=K_expr, length=L, code=code, residual=residual
    )


def self_check(m: Manufactured) -> str | None:
    """`None` if the instance is sound, else why not.

    Three things, and the first is the one that makes this a factory rather than
    a hope: substituting `u` back into the PDE with the derived `F` must give
    **exactly** zero.
    """
    if sp.simplify(m.residual) != 0:
        return f"the manufactured solution does not satisfy its own PDE: residual {m.residual}"
    if sp.simplify(m.u_expr.subs(X, 0)) != 0:
        return "the manufactured solution violates u(0) = 0"
    slope = sp.simplify(sp.diff(m.u_expr, X).subs(X, m.length))
    if sp.simplify(slope) != 0:
        return f"the manufactured solution violates u'(L) = 0: slope {slope}"
    return None


def evaluate(expr: sp.Expr, x: np.ndarray) -> np.ndarray:
    return np.asarray(sp.lambdify(X, expr, "numpy")(x), dtype=float) * np.ones_like(x)
