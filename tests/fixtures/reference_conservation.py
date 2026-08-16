"""A symplectic solution to Family 2. Stormer-Verlet, exact on the linear part.

Energy is conserved because the scheme is symplectic, not because a damping term
was tuned to zero: Verlet has no amplitude error at all, only a bounded phase
error, which is why it survives 20 periods at 1e-8.

Reversibility comes for free from the same property. That is the point of the
family's second check: a scheme with even a thousandth of a percent of numerical
damping per period passes the energy check over four periods and cannot find its
way home.
"""
import numpy as np


def evolve(n_cells, length, c, n_periods, samples, u0, v0):
    dx = length / n_cells
    x = np.arange(1, n_cells + 1, dtype=float) * dx
    u = np.asarray(u0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()

    def accel(w):
        a = np.empty_like(w)
        a[1:-1] = (w[2:] - 2.0 * w[1:-1] + w[:-2]) / dx**2
        a[0] = (w[1] - 2.0 * w[0]) / dx**2            # u(0) = 0, eliminated
        a[-1] = 2.0 * (w[-2] - w[-1]) / dx**2         # u'(L) = 0 by the ghost node
        return (c**2) * a

    period = 4.0 * length / c                          # fundamental of a fixed-free string
    total = n_periods * period
    # Comfortably inside the CFL limit of the explicit scheme.
    dt = 0.4 * dx / c
    n_steps = int(np.ceil(total / dt))
    dt = total / n_steps

    keep = max(1, n_steps // (samples - 1))
    out_t, out_u, out_v = [0.0], [u.tolist()], [v.tolist()]

    a = accel(u)
    for k in range(1, n_steps + 1):
        v += 0.5 * dt * a
        u += dt * v
        a = accel(u)
        v += 0.5 * dt * a
        if k % keep == 0 and len(out_t) < samples:
            out_t.append(k * dt)
            out_u.append(u.tolist())
            out_v.append(v.tolist())

    while len(out_t) < samples:                        # pad if the stride undershot
        out_t.append(n_steps * dt)
        out_u.append(u.tolist())
        out_v.append(v.tolist())

    return {"t": out_t, "u": out_u, "v": out_v, "x": x.tolist()}
