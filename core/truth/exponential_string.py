"""Fixed-free string with `K = K0 e^{-ax}`, `mu = mu0 e^{-ax}`: no closed form.

The wave speed `c = sqrt(K0/mu0)` is constant while the impedance is not, so the
eigenvalue problem reduces to `phi'' - a phi' + (w/c)^2 phi = 0` with solution
`phi = e^{ax/2} sin(b x)`, `b = sqrt((w/c)^2 - a^2/4)`. Applying `phi'(L) = 0`
gives the transcendental condition

    tan(b L) = -2 b / a        <=>        (a/2) sin(bL) + b cos(bL) = 0

and then `w_n = c sqrt(b_n^2 + a^2/4)`.

## Why the entire form, and not the tangent

This module solves the **right-hand** version, and the reason is narrower than
the one first written here — which was wrong, and is corrected rather than
quietly dropped.

The first draft claimed `tan` has a pole *inside every bracket*. It does not:
the poles of `tan(bL)` sit at `bL = (2k-1)pi/2`, which are exactly the **lower
endpoints** of the brackets `[(2n-1)pi/2L, n pi/L]` used here. Inside the open
interval `tan` is continuous.

What actually goes wrong is the *other* bracket. A scorer in the ancestor of this
repo bracketed on `[(n-1)pi/L, n pi/L]` — the natural-looking choice, one period
of the tangent — which puts a pole squarely in the interior. `tan` changes sign
across a pole exactly as it does across a root, so bisection cannot tell them
apart and converges to the pole. That scorer scored a **correct** reference
solution 0.0, and the failure was in the verifier, which is the worst place for
it: it reads as the candidate being wrong.

The entire form removes the question. It is finite everywhere, so a sign change
is a root and nothing else, and it has exactly one in `[(2n-1)pi/2L, n pi/L]`.

## Why this is the family's anti-hack argument

`a` is sampled continuously. These roots are in no table, no textbook and no
training set; interpolating from memory misses by orders of magnitude more than
the tolerance. Spec §3, Familia 1: *"or you compute them or you do not have
them."*
"""

from __future__ import annotations

import mpmath as mp

mp.mp.dps = 50


def _entire(b, alpha: float, length: float):
    """`(a/2) sin(bL) + b cos(bL)`. Zero at every mode, finite everywhere."""
    return (mp.mpf(alpha) / 2) * mp.sin(b * length) + b * mp.cos(b * length)


def beta(n: int, alpha: float, length: float) -> mp.mpf:
    """The n-th root, ``n >= 1``, by bisection inside its own bracket."""
    if n < 1:
        raise ValueError("modes are numbered from 1")
    if alpha <= 0:
        raise ValueError("the bracketing argument assumes alpha > 0")
    if length <= 0:
        raise ValueError("a string needs a positive length")
    lo = mp.mpf(2 * n - 1) * mp.pi / (2 * mp.mpf(length))
    hi = mp.mpf(n) * mp.pi / mp.mpf(length)
    # Nudge off the ends: each is an exact zero of one of the two terms, and
    # bisection wants a genuine sign change rather than a boundary touch.
    eps = (hi - lo) * mp.mpf("1e-30")
    return mp.findroot(
        lambda b: _entire(b, alpha, length), (lo + eps, hi - eps), solver="bisect", tol=mp.mpf("1e-45")
    )


def omega(n: int, alpha: float, length: float, c: float) -> float:
    """`w_n = c sqrt(b_n^2 + a^2/4)`."""
    b = beta(n, alpha, length)
    return float(mp.mpf(c) * mp.sqrt(b**2 + mp.mpf(alpha) ** 2 / 4))


def omegas(n_modes: int, alpha: float, length: float, c: float) -> list[float]:
    return [omega(n, alpha, length, c) for n in range(1, n_modes + 1)]


def residual(n: int, alpha: float, length: float) -> float:
    """The equation at the returned root, **relative to the size of its terms**.

    The route-independent check: a root is a root because the equation vanishes
    there, not because a solver said so. `tests/test_truth_crosscheck.py` uses
    this as the second opinion where a second *derivation* does not exist.

    Normalised by `b + a/2`, and that was measured rather than assumed. The bare
    absolute residual runs from 1.4e-30 to 2.5e-29 across the sampled ranges,
    because both terms scale with `b` — so an absolute threshold is a threshold
    on `alpha` and `n` wearing a disguise, tight for the first mode and loose for
    the eighth.
    """
    b = beta(n, alpha, length)
    scale = b + mp.mpf(alpha) / 2
    return float(abs(_entire(b, alpha, length)) / scale)


def mode_shape(n: int, x, alpha: float, length: float):
    """`phi_n(x) = e^{ax/2} sin(b_n x)`, unnormalised."""
    b = beta(n, alpha, length)
    return [float(mp.e ** (mp.mpf(alpha) * mp.mpf(xi) / 2) * mp.sin(b * mp.mpf(xi))) for xi in x]
