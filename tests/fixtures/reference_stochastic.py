"""An exact sampler for the OU pair. Family 4.

`q'' + 2G q' + w^2 q = xi`, `<xi xi> = 2D delta`. As a first-order system,
`z = (q, q')` is a linear SDE whose transition over a step `dt` is Gaussian with
mean `expm(A dt) z` and a covariance that can be written down. Sampling that
transition has **no discretisation error at all**, which is what level 4 asks
for: the statistics do not depend on `dt`.

The stationary covariance is `P = diag(D/(2 G w^2), D/(2 G))`, and the transition
covariance is `P - expm(A dt) P expm(A dt)'` — the standard result, and cheaper
than Van Loan's block-matrix exponential for a 2x2.
"""
import numpy as np
from scipy.linalg import expm


def simulate(omega, gamma, noise_intensity, duration, n_replicates, seed):
    A = np.array([[0.0, 1.0], [-(omega**2), -2.0 * gamma]])
    P = np.diag([noise_intensity / (2.0 * gamma * omega**2), noise_intensity / (2.0 * gamma)])

    dt = 0.2 / omega
    n = int(duration / dt)
    Phi = expm(A * dt)
    Q = P - Phi @ P @ Phi.T
    Q = 0.5 * (Q + Q.T)                       # symmetrise away the roundoff
    L = np.linalg.cholesky(Q + 1e-15 * np.eye(2))

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_replicates):
        # Start from the stationary distribution: no burn-in needed, and no
        # transient contaminating the variance.
        z = np.linalg.cholesky(P) @ rng.standard_normal(2)
        qs = np.empty(n)
        noise = rng.standard_normal((n, 2))
        for k in range(n):
            z = Phi @ z + L @ noise[k]
            qs[k] = z[0]
        out.append(qs.tolist())
    return {"q": out, "dt": dt}
