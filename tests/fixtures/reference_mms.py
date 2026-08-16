"""A correct finite-volume solve of `-(K u')' + mu u = F`, fixed-free. Family 5.

The apex owns half a cell, and the reaction term and the forcing are integrated
over each cell rather than sampled at the node — which is the same statement:
this is a finite *volume* scheme, and every term has to carry the same weight.
"""
import numpy as np


def solve(length, n_points, mu, K, F):
    x = np.linspace(0.0, length, n_points)
    dx = x[1] - x[0]
    xi = x[1:]                      # u(0) = 0 is eliminated
    n = xi.size

    lower = K(xi - dx / 2.0)        # inner face of each cell
    upper = K(xi[:-1] + dx / 2.0)   # outer face, except the apex's

    volume = np.full(n, dx)
    volume[-1] = dx / 2.0           # u'(L) = 0: the apex owns half a cell

    main = np.empty(n)
    main[:-1] = (upper + lower[:-1]) / dx
    main[-1] = lower[-1] / dx
    main = main + mu(xi) * volume   # the reaction term carries the same weight

    A = np.diag(main) + np.diag(-upper / dx, 1) + np.diag(-upper / dx, -1)
    u = np.linalg.solve(A, F(xi) * volume)

    return {"u": np.concatenate(([0.0], u)).tolist(), "order": 2.0}
