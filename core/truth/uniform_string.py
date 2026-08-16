"""Fixed-free uniform string: eigenvalues and mode shapes, at 50 digits.

`u(0) = 0`, `u'(L) = 0`, constant `mu` and `K`, `c = sqrt(K/mu)`. Separation of
variables gives `phi'' + (w/c)^2 phi = 0`; `phi(0)=0` kills the cosine and
`phi'(L)=0` forces `cos(wL/c) = 0`, so

    w_n = (2n-1) pi c / (2 L),      phi_n(x) = sin((2n-1) pi x / (2L))

Only odd quarter-wave harmonics — the classic closed-open pipe.

## Two independent routes, because spec §1.5 asks for them

`omegas()` evaluates the formula. `omegas_by_root_finding()` bisects
`cos(wL/c) = 0` inside each branch without ever writing `(2n-1)pi/2`. They agree
to 1e-40 or the bug is ours, and `tests/test_truth_crosscheck.py` says so.

That is not ceremony. A formula this short is exactly the kind that gets
transcribed with an off-by-one — `2n+1`, or `pi/2` for `pi`, or the length in the
wrong place — and a single implementation checked against itself would agree with
the mistake.

Imports nothing outside mpmath.
"""

from __future__ import annotations

import mpmath as mp

mp.mp.dps = 50


def omega(n: int, c: float, length: float) -> float:
    """`w_n = (2n-1) pi c / 2L`, ``n >= 1``."""
    if n < 1:
        raise ValueError("modes are numbered from 1")
    if c <= 0 or length <= 0:
        raise ValueError("a string needs a positive speed and a positive length")
    return float((2 * n - 1) * mp.pi * mp.mpf(c) / (2 * mp.mpf(length)))


def omegas(n_modes: int, c: float, length: float) -> list[float]:
    return [omega(n, c, length) for n in range(1, n_modes + 1)]


def omegas_by_root_finding(n_modes: int, c: float, length: float) -> list[float]:
    """The same numbers, found rather than written.

    Bisects `cos(w L / c)` inside `[(n-1) pi c / L, n pi c / L]`, where it has
    exactly one sign change. Nothing here contains the closed form, which is the
    point: this route can disagree with `omegas()`, and a route that cannot
    disagree is not a cross-check.
    """
    if n_modes < 1:
        raise ValueError("n_modes must be at least 1")
    f = lambda w: mp.cos(w * mp.mpf(length) / mp.mpf(c))  # noqa: E731
    out: list[float] = []
    for n in range(1, n_modes + 1):
        lo = mp.mpf(n - 1) * mp.pi * mp.mpf(c) / mp.mpf(length)
        hi = mp.mpf(n) * mp.pi * mp.mpf(c) / mp.mpf(length)
        eps = (hi - lo) * mp.mpf("1e-30")
        out.append(float(mp.findroot(f, (lo + eps, hi - eps), solver="bisect", tol=mp.mpf("1e-45"))))
    return out


def mode_shape(n: int, x, c: float, length: float):
    """`phi_n(x) = sin(k_n x)`, unnormalised. Scale is not part of any claim."""
    k = (2 * n - 1) * mp.pi / (2 * mp.mpf(length))
    return [float(mp.sin(k * mp.mpf(xi))) for xi in x]


def interior_zeros(n: int) -> int:
    """`n - 1`, by the Sturm-Liouville oscillation theorem."""
    if n < 1:
        raise ValueError("modes are numbered from 1")
    return n - 1
