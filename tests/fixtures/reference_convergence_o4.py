"""Fourth order by Richardson inside the solver. Family 3 level 4.

The extrapolation has to live in the candidate: the verifier calls it once per
grid size and measures the slope across those calls, so an extrapolation done
afterwards would not appear in the numbers at all.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal


def _eigs(n_modes, length, c, alpha, n_cells):
    dx = length / n_cells
    x = np.arange(1, n_cells + 1, dtype=float) * dx
    K = lambda t: (c**2) * np.exp(-alpha * t)          # noqa: E731
    lower, upper = K(x - dx / 2.0), K(x[:-1] + dx / 2.0)
    volume = np.full(n_cells, dx)
    volume[-1] = dx / 2.0
    d = np.empty(n_cells)
    d[:-1] = (upper + lower[:-1]) / dx
    d[-1] = lower[-1] / dx
    e = -upper / dx
    inv = 1.0 / np.sqrt(np.exp(-alpha * x) * volume)
    lam, _ = eigh_tridiagonal(
        d * inv * inv, e * inv[:-1] * inv[1:], select="i", select_range=(0, n_modes - 1)
    )
    return np.sqrt(np.maximum(lam, 0.0))


def eigenfrequencies(n_modes, length, c, alpha, n_cells):
    a = _eigs(n_modes, length, c, alpha, n_cells)
    b = _eigs(n_modes, length, c, alpha, 2 * n_cells)
    return {"omegas": ((4.0 * b - a) / 3.0).tolist()}
