"""A second-order solver, parameterised by grid size. Family 3 levels 1-3."""
import numpy as np
from scipy.linalg import eigh_tridiagonal


def eigenfrequencies(n_modes, length, c, alpha, n_cells):
    dx = length / n_cells
    x = np.arange(1, n_cells + 1, dtype=float) * dx
    K = lambda t: (c**2) * np.exp(-alpha * t)          # noqa: E731
    lower, upper = K(x - dx / 2.0), K(x[:-1] + dx / 2.0)

    volume = np.full(n_cells, dx)
    volume[-1] = dx / 2.0                              # the apex owns half a cell

    d = np.empty(n_cells)
    d[:-1] = (upper + lower[:-1]) / dx
    d[-1] = lower[-1] / dx
    e = -upper / dx
    m = np.exp(-alpha * x) * volume

    inv = 1.0 / np.sqrt(m)
    lam, _ = eigh_tridiagonal(
        d * inv * inv, e * inv[:-1] * inv[1:], select="i", select_range=(0, n_modes - 1)
    )
    return {"omegas": np.sqrt(np.maximum(lam, 0.0)).tolist()}
