"""Rice's level-crossing rate for a stationary Gaussian process.

    r0 = (1 / 2pi) * (sigma_v / sigma_u) * exp(-theta^2 / 2 sigma_u^2)

For the damped mode of `ou_lyapunov`, `sigma_v / sigma_u = w` **exactly** —
because the stationary cross-covariance vanishes, which that module derives
rather than asserts. So the ratio in front is the mode's own frequency and the
crossing rate depends on nothing else.

That is what makes it a verifier rather than a fit: a candidate's simulated
crossing rate is compared against a formula with **no free parameter**, given the
variance the same candidate's trajectories exhibit.

Rice, "Mathematical analysis of random noise", Bell System Technical Journal
(1944-45). Imports nothing but the standard library.
"""

from __future__ import annotations

import math


def crossing_rate(omega: float, theta: float, sigma: float) -> float:
    """Upward crossings of level `theta` per unit time."""
    if sigma <= 0:
        raise ValueError("a process with no variance crosses nothing")
    return (omega / (2.0 * math.pi)) * math.exp(-(theta**2) / (2.0 * sigma**2))


def expected_crossings(omega: float, theta: float, sigma: float, duration: float) -> float:
    return crossing_rate(omega, theta, sigma) * duration


def crossing_count_standard_error(omega: float, theta: float, sigma: float, duration: float) -> float:
    """`sqrt(N)` for a rate process, which is the honest error bar on a count.

    Upward crossings of a high level are approximately Poisson — they are rare
    and nearly independent — so the count's standard deviation is the square root
    of its mean. **This is the number that makes the family work**: a candidate
    is not asked to match a count exactly, which no honest simulation can do, but
    to land within its own sampling error of it.

    The approximation is stated rather than hidden: near `theta = 0` crossings
    are neither rare nor independent and this underestimates the spread. The
    family samples `theta / sigma` above 1.5 for that reason.
    """
    n = expected_crossings(omega, theta, sigma, duration)
    return math.sqrt(n) if n > 0 else 0.0
