"""Stationary statistics of a damped mode driven by white noise.

    q'' + 2 G q' + w^2 q = xi(t),      <xi(t) xi(t')> = 2 D delta(t - t')

**Both factors of two are conventions and both are stated everywhere they
matter** — in this docstring, in every prompt that uses this truth, and in the
manifest. The ancestor of this repo lost a correct solution to exactly this: the
prompt said nothing, the reference used `D/(G w^2)`, the scorer wanted
`D/(2 G w^2)`, and the relative error came out at exactly 1.0. An environment
that rewards guessing the author's convention measures telepathy.

## Two routes, again

`variance()` writes the answer down. `variance_by_lyapunov()` solves
`A P + P A^T + Sigma Sigma^T = 0` with sympy for the companion matrix and never
contains the closed form. They must agree — spec §1.5.

The vanishing cross-covariance is not decoration: it makes
`sigma_qdot / sigma_q = w` exactly, which is the ratio Rice's crossing rate needs.
"""

from __future__ import annotations

import sympy as sp


def variance(omega: float, gamma: float, noise_intensity: float) -> float:
    """`Var(q) = D / (2 G w^2)`, with `G = gamma` and `D = noise_intensity`."""
    if omega <= 0 or gamma <= 0:
        raise ValueError("a mode with no frequency or no damping has no stationary state")
    return noise_intensity / (2.0 * gamma * omega**2)


def velocity_variance(omega: float, gamma: float, noise_intensity: float) -> float:
    """`Var(q') = D / (2 G)` — independent of `w`."""
    if gamma <= 0:
        raise ValueError("no damping, no stationary state")
    return noise_intensity / (2.0 * gamma)


def autocorrelation_time(gamma: float) -> float:
    """`tau = 1 / G`. The envelope's decay, not the oscillation's period."""
    if gamma <= 0:
        raise ValueError("no damping, no correlation time")
    return 1.0 / gamma


def lyapunov_solution() -> dict[str, sp.Expr]:
    """Solve `A P + P A^T + Sigma Sigma^T = 0` symbolically. The second route.

    Nothing here contains `D / (2 G w^2)`; it comes out of the linear system.
    A cross-check that restated the answer would agree with a wrong answer.
    """
    w, g, d = sp.symbols("w g d", positive=True)
    p11, p12, p22 = sp.symbols("p11 p12 p22", real=True)
    A = sp.Matrix([[0, 1], [-(w**2), -2 * g]])
    P = sp.Matrix([[p11, p12], [p12, p22]])
    Q = sp.Matrix([[0, 0], [0, 2 * d]])
    eqs = list((A * P + P * A.T + Q))
    sol = sp.solve(eqs, [p11, p12, p22], dict=True)[0]
    return {
        "var_q": sp.simplify(sol[p11]),
        "cov": sp.simplify(sol[p12]),
        "var_qdot": sp.simplify(sol[p22]),
    }


def variance_by_lyapunov(omega: float, gamma: float, noise_intensity: float) -> float:
    """`Var(q)` from the symbolic solution, evaluated. Route two, numerically."""
    w, g, d = sp.symbols("w g d", positive=True)
    expr = lyapunov_solution()["var_q"]
    return float(expr.subs({w: omega, g: gamma, d: noise_intensity}))


def standard_error_of_variance(true_variance: float, n_independent: int) -> float:
    """`SE(s^2) = sigma^2 sqrt(2 / (n - 1))` for a Gaussian process.

    Spec §3, Familia 4 — the whole trick. The reward compares an estimator to the
    truth **in units of its own sampling error**, so an honest simulation with
    finite samples scores well and a fabricated exact answer does not: the latter
    has no inter-replicate spread, which the same theory predicts and checks.

    `n_independent` is the number of *independent* samples, not raw steps: a
    trajectory of length T with correlation time tau carries about `T / (2 tau)`
    of them. Passing the step count here is the classic way to make an honest
    estimator look like a fraud.
    """
    if n_independent < 2:
        raise ValueError("a variance needs at least two independent samples")
    return true_variance * (2.0 / (n_independent - 1)) ** 0.5
