"""Two independent routes per truth. Spec §1.5, and one of the three tests
that define the product.

The failure this prevents is the worst kind a verifier has: **the scorer is
wrong and the candidate is blamed.** It has already happened once in the ancestor
of this repo: a re-derivation of the exponential roots bracketed `tan` over a
full period, which puts a pole in the interior, converged to the pole, and scored
a correct reference solution 0.0.

A single implementation checked against itself agrees with its own mistakes. So
every truth is computed twice by routes that could disagree:

======================  =========================  ============================
truth                   route A                    route B
======================  =========================  ============================
uniform eigenvalues     the closed form            bisection, formula-free
exponential roots       bisection on the entire    the residual of the equation
                        form                       at the returned root
OU stationary variance  the written answer         sympy solving A P + P A' + Q
======================  =========================  ============================

Route B never contains route A's answer. A cross-check that restated the formula
would confirm a typo in it.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.truth import exponential_string as expo
from core.truth import ou_lyapunov as ou
from core.truth import uniform_string as uni


# --- uniform: closed form vs formula-free root finding ----------------------


@settings(deadline=None, max_examples=40)
@given(
    c=st.floats(min_value=0.5, max_value=3.0),
    length=st.floats(min_value=0.5, max_value=2.0),
    n_modes=st.integers(min_value=1, max_value=12),
)
def test_uniform_eigenvalues_agree_by_two_routes(c, length, n_modes):
    a = uni.omegas(n_modes, c, length)
    b = uni.omegas_by_root_finding(n_modes, c, length)
    for i, (x, y) in enumerate(zip(a, b), start=1):
        assert abs(x - y) / abs(y) < 1e-30, f"mode {i}: {x} vs {y}"


def test_the_crosscheck_can_actually_fail():
    """A guard on the guard.

    If `omegas_by_root_finding` secretly called `omegas`, the test above would
    pass over any error in the closed form. Perturbing the formula must break the
    agreement — and if it does not, the two routes are one route.
    """
    correct = uni.omegas(4, 1.0, 1.0)
    # The off-by-one a transcription of this formula actually makes: 2n+1.
    wrong = [(2 * n + 1) * math.pi * 1.0 / (2 * 1.0) for n in range(1, 5)]
    found = uni.omegas_by_root_finding(4, 1.0, 1.0)
    assert all(abs(x - y) / y < 1e-30 for x, y in zip(correct, found))
    assert any(abs(x - y) / y > 1e-3 for x, y in zip(wrong, found)), (
        "root finding agrees with a wrong formula, so it is not independent"
    )


# --- exponential: the root, and the equation at the root --------------------


@settings(deadline=None, max_examples=30)
@given(
    alpha=st.floats(min_value=0.5, max_value=6.0),
    length=st.floats(min_value=0.5, max_value=2.0),
    n=st.integers(min_value=1, max_value=8),
)
def test_exponential_roots_satisfy_their_own_equation(alpha, length, n):
    """No second derivation exists, so the second opinion is the equation.

    The residual is relative — see `exponential_string.residual`. The first
    version of this asserted an absolute `< 1e-30` and failed on the eighth mode
    at `alpha = 0.5`, where the true residual was 2.1e-29. Both terms scale with
    `b`, so an absolute bound is a bound on `alpha` and `n` in disguise.
    """
    assert expo.residual(n, alpha, length) < 1e-30


@settings(deadline=None, max_examples=20)
@given(
    alpha=st.floats(min_value=0.5, max_value=6.0),
    length=st.floats(min_value=0.5, max_value=2.0),
)
def test_exponential_roots_are_ordered_and_one_per_branch(alpha, length):
    """Each root inside its own bracket, none skipped, none doubled.

    A solver that lands twice in one branch produces a list that looks fine and
    is missing a mode — and every downstream comparison then compares mode `k`
    against mode `k+1`.
    """
    betas = [float(expo.beta(n, alpha, length)) for n in range(1, 9)]
    assert betas == sorted(betas)
    for n, b in enumerate(betas, start=1):
        lo = (2 * n - 1) * math.pi / (2 * length)
        hi = n * math.pi / length
        assert lo < b < hi, f"root {n} = {b} outside its bracket ({lo}, {hi})"


def test_the_bracket_is_what_makes_the_tangent_unsafe_not_the_tangent():
    """The bug that motivated this module — and the correction to why.

    The first version of this test asserted the tangent's pole lies *inside* the
    bracket used here. **It does not**, and the test said so by failing:
    `assert 4.71238898 < 4.71238898`. The poles of `tan(bL)` are at
    `bL = (2k-1)pi/2`, which are exactly the lower endpoints of
    `[(2n-1)pi/2L, n pi/L]`.

    The real hazard is the *other* bracket — `[(n-1)pi/L, n pi/L]`, one full
    period of the tangent, which is the natural-looking choice and puts a pole in
    the interior. `tan` changes sign across a pole exactly as across a root, so
    bisection cannot distinguish them. This asserts both facts.
    """
    length = 1.0
    for n in (1, 2, 3, 4):
        lo = (2 * n - 1) * math.pi / (2 * length)
        hi = n * math.pi / length
        pole = (2 * n - 1) * math.pi / (2 * length)
        assert pole == pytest.approx(lo), "the pole is at the endpoint, not inside"

        naive_lo, naive_hi = (n - 1) * math.pi / length, n * math.pi / length
        # ...and inside the natural-looking full-period bracket, it is.
        if n > 1:
            assert naive_lo < pole < naive_hi, (
                f"branch {n}: the pole should be interior to the full-period bracket"
            )

        root = float(expo.beta(n, 2.0, length))
        assert lo < root < hi
        assert abs(root - pole) > 1e-9, "the root is not the pole"


# --- Ornstein-Uhlenbeck: written answer vs solved Lyapunov equation ---------


@settings(deadline=None, max_examples=25)
@given(
    omega=st.floats(min_value=0.2, max_value=5.0),
    gamma=st.floats(min_value=0.01, max_value=1.0),
    d=st.floats(min_value=0.1, max_value=10.0),
)
def test_stationary_variance_agrees_with_the_solved_lyapunov_equation(omega, gamma, d):
    a = ou.variance(omega, gamma, d)
    b = ou.variance_by_lyapunov(omega, gamma, d)
    assert abs(a - b) / abs(b) < 1e-12, f"{a} vs {b}"


def test_the_cross_covariance_vanishes_which_is_what_makes_rice_work():
    sol = ou.lyapunov_solution()
    assert sol["cov"] == 0, f"cov = {sol['cov']}, so sigma_qdot/sigma_q is not w"


def test_the_convention_is_the_one_every_prompt_will_state():
    """`Var(q) = D / (2 G w^2)`. Pinned, because a factor of two once cost a day.

    Anything that changes this must change every prompt in the same commit; this
    test is what makes that impossible to forget.
    """
    assert ou.variance(omega=2.0, gamma=0.5, noise_intensity=1.0) == pytest.approx(0.25)
    assert ou.velocity_variance(omega=2.0, gamma=0.5, noise_intensity=1.0) == pytest.approx(1.0)


def test_the_standard_error_refuses_a_sample_of_one():
    """An estimator with no spread has no sampling error to compare against, and
    returning a number there is how a fabricated answer scores well."""
    with pytest.raises(ValueError):
        ou.standard_error_of_variance(1.0, n_independent=1)
    assert ou.standard_error_of_variance(1.0, 101) == pytest.approx(math.sqrt(2 / 100))
