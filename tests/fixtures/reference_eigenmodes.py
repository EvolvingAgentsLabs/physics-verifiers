"""A correct solution to Family 1, as a model would be asked to write one.

numpy and scipy only. It discretises and solves rather than evaluating a closed
form: a candidate returning `(2n-1)pi c / 2L` passes the uniform instances and
fails every exponential one, which is what sampling the profile is for.

## The two details that decide whether this is second order

Both were learned by getting them wrong in the ancestor project, and both are
what Family 1 level 2 asks about in as many words:

* **The apex owns half a cell.** Giving the boundary node a full cell is an O(dx)
  error in the total mass and therefore in every eigenvalue. It scored 0.89 where
  1.0 was needed, and Richardson extrapolation made it *worse* — the giveaway
  that the error was not second order.
* **K belongs at the faces.** Sampling it at the nodes is second order for smooth
  K and first order at the boundary, which is the same bug wearing a different
  hat.
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal


def _assemble(mu, K, n_cells, length):
    dx = length / n_cells
    x = np.arange(1, n_cells + 1, dtype=float) * dx

    lower = K(x - dx / 2.0)
    upper = K(x[:-1] + dx / 2.0)

    volume = np.full(n_cells, dx)
    volume[-1] = dx / 2.0  # du/dx(L) = 0: the apex owns half a cell

    d = np.empty(n_cells)
    d[:-1] = (upper + lower[:-1]) / dx
    d[-1] = lower[-1] / dx  # no outer flux at all -- that IS the Neumann
    e = -upper / dx
    return d, e, mu(x) * volume


def _eigs(mu, K, n_modes, n_cells, length):
    d, e, m = _assemble(mu, K, n_cells, length)
    inv = 1.0 / np.sqrt(m)
    lam, _ = eigh_tridiagonal(
        d * inv * inv, e * inv[:-1] * inv[1:], select="i", select_range=(0, n_modes - 1)
    )
    return np.sqrt(np.maximum(lam, 0.0))


def _richardson(mu, K, n_modes, length, n_cells=4000):
    """Second order, so `(4 w_2N - w_N)/3` cancels the leading error."""
    a = _eigs(mu, K, n_modes, n_cells, length)
    b = _eigs(mu, K, n_modes, 2 * n_cells, length)
    return (4.0 * b - a) / 3.0


#: Overwritten per instance by the level-3 test. A static file cannot diagnose
#: code it was never shown, and a diagnosis broad enough to cover all four
#: categories would be a diagnosis that says nothing — which this repo's own
#: anti-gaming test would then have to catch. So the fixture is parameterised
#: rather than made vague.
DIAGNOSIS = "no diagnosis was supplied"


def eigenfrequencies(n_modes, length, c, alpha):
    mu = (lambda x: np.exp(-alpha * x)) if alpha else (lambda x: np.ones_like(x))
    K = (lambda x: c**2 * np.exp(-alpha * x)) if alpha else (lambda x: np.full_like(x, c**2))
    return {"omegas": _richardson(mu, K, n_modes, length).tolist(), "diagnosis": DIAGNOSIS}


def _forward(mu_samples, length, K, n_modes):
    """Same finite-volume forward solve the verifier uses. Half a cell at the apex."""
    mu_nodes = np.asarray(mu_samples, dtype=float)
    dx = length / (mu_nodes.size - 1)
    mu_u = mu_nodes[1:]
    n = mu_u.size
    volume = np.full(n, dx)
    volume[-1] = dx / 2.0
    d = np.empty(n)
    d[:-1] = 2.0 * K / dx
    d[-1] = K / dx
    e = np.full(n - 1, -K / dx)
    inv = 1.0 / np.sqrt(mu_u * volume)
    lam, _ = eigh_tridiagonal(
        d * inv * inv, e * inv[:-1] * inv[1:], select="i", select_range=(0, n_modes - 1)
    )
    return np.sqrt(np.maximum(lam, 0.0))


def profile(length, n_points, targets=None, K=None):
    """Level 4: recover the hidden mass profile from its eigenvalues.

    A one-parameter family `mu = exp(-a x)` contains the answer, and the
    fundamental **rises** monotonically with `a` — larger `a` means *less* mass
    toward the free end, so a stiffer string. Bisection on the first target finds
    it, and the remaining four follow because they came from the same family.
    Matching one eigenvalue and getting five is the evidence that the family was
    the right guess rather than a fit.

    The direction above was inverted in the first version, which "reasoned" that
    more mass lowers the frequency and then read `exp(-a x)` backwards. The
    bisection converged to a bound and the error came out at 150-370%.

    `targets` and `K` are not passed by the harness; they are read from the
    prompt by whoever writes the solution. The defaults here let the fixture be
    driven directly by a test.
    """
    x = np.linspace(0.0, length, n_points)
    if targets is None or K is None:
        return {"mu": np.exp(-x).tolist()}

    lo, hi = 0.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        w1 = _forward(np.exp(-mid * x), length, K, 1)[0]
        if w1 > targets[0]:
            hi = mid          # too stiff: less decay, i.e. a smaller `a`
        else:
            lo = mid
    return {"mu": np.exp(-0.5 * (lo + hi) * x).tolist()}
