"""Family 4: a stochastic reward with no LLM judge, and the fabrication it catches.

The differentiating claim of the whole repo, so it is tested as a claim: an
honest simulation scores well, and **a fabricated exact answer scores zero** —
not because it is wrong, but because it is impossibly right.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from core.contracts import AgentOutput
from core.families import stochastic as S
from core.truth import ou_lyapunov as ou
from core.truth import rice_rates as rice

REFERENCE = (pathlib.Path(__file__).parent / "fixtures" / "reference_stochastic.py").read_text()
FAM = S.Stochastic()


def _task(level: int = 2, seed: int = 2):
    return FAM.generate(level, np.random.default_rng(seed))


def _exact(task):
    p = task.params
    return ou.variance(p["omega"], p["gamma"], p["noise_intensity"])


@pytest.mark.parametrize("level", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", [0, 10])
def test_an_exact_sampler_scores_well(level, seed):
    """An honest simulation lands at |z| of order 1 and must not be punished for it."""
    task = FAM.generate(level, np.random.default_rng(level + seed))
    verdict = FAM.verify(task, S.run_in_sandbox(task, REFERENCE))
    assert verdict.hard_gate, verdict.breakdown
    assert verdict.reward >= 0.7, verdict.breakdown


def test_the_reward_does_not_punish_ordinary_sampling_noise():
    """`|z| <= 1` is free. The first version scored `exp(-|z|/3)` three times over
    and gave a *perfect* exact sampler 0.27 — it was penalising the sampling
    noise the family exists to tolerate."""
    assert S.Z_FREE == 1.0
    task = _task(1, 1)
    verdict = FAM.verify(task, S.run_in_sandbox(task, REFERENCE))
    if abs(verdict.breakdown["z_variance"]) <= 1 and abs(verdict.breakdown["z_spread"]) <= 1:
        assert verdict.reward >= 0.9


# --- the fabrications -------------------------------------------------------


def _fake_zero_spread(task) -> str:
    sd = math.sqrt(_exact(task))
    return f"""
import numpy as np
def simulate(omega, gamma, noise_intensity, duration, n_replicates, seed):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_replicates):
        x = rng.standard_normal(20000)
        x = (x - x.mean()) / x.std(ddof=1) * {sd!r}   # forced to the exact variance
        out.append(x.tolist())
    return {{"q": out, "dt": 0.05}}
"""


def _white_noise(task) -> str:
    sd = math.sqrt(_exact(task))
    return f"""
import numpy as np
def simulate(omega, gamma, noise_intensity, duration, n_replicates, seed):
    rng = np.random.default_rng(seed)
    return {{"q": (rng.standard_normal((n_replicates, 20000)) * {sd!r}).tolist(), "dt": 0.05}}
"""


def test_the_exact_answer_with_no_scatter_is_caught_as_fabrication():
    """Being *too good* is the failure mode, and it is the family's whole point.

    `z_variance` comes out at 0.00 — better than any honest run — and the
    inter-replicate spread gives it away.
    """
    task = _task()
    verdict = FAM.verify(task, S.run_in_sandbox(task, _fake_zero_spread(task)))
    assert verdict.reward == 0.0
    assert verdict.breakdown["fabrication_suspected"] is True
    assert abs(verdict.breakdown["z_variance"]) < 1.0, "it matched the variance perfectly"
    assert verdict.breakdown["z_spread"] < -S.Z_GATE


def test_white_noise_with_the_right_variance_is_caught():
    """The second cheat the trio needs: right value, wrong process."""
    task = _task()
    verdict = FAM.verify(task, S.run_in_sandbox(task, _white_noise(task)))
    assert verdict.reward == 0.0, verdict.breakdown


def test_the_broken_simulator_has_the_right_shape_and_the_wrong_amplitude():
    """It IS an OU process — the correlation time is right to 2% — and its noise
    term is missing a `sqrt(2)`. Only the variance check sees it."""
    task = _task(3, 3)
    verdict = FAM.verify(task, S.run_in_sandbox(task, S.BROKEN_SIMULATOR))
    assert verdict.reward == 0.0
    assert "variance" in verdict.breakdown["reason"]
    assert verdict.breakdown["tau_relative_error"] < 0.3, "its shape is fine; its amplitude is not"


# --- the instruments --------------------------------------------------------


def test_the_effective_sample_count_is_two_T_over_tau():
    """`q^2` decorrelates in `tau/2`, so a run of length T carries `2T/tau`.

    It was `T/(2 tau)` — four times short — and the exact sampler was accused of
    scattering half as much as it should (`z_spread = -2.3`). Measured against 40
    replicates, the implied `n_eff/(T/tau)` is 1.63, 1.71, 1.76 at T/tau = 100,
    200, 400: close to 2 and slightly under, as expected.
    """
    task = _task()
    out = S.run_in_sandbox(task, REFERENCE)
    verdict = FAM.verify(task, out)
    q = np.asarray(out.data["q"])
    duration = q.shape[1] * out.data["dt"]
    tau = ou.autocorrelation_time(task.params["gamma"])
    assert verdict.breakdown["n_independent_per_replicate"] == pytest.approx(2 * duration / tau)


def test_the_correlation_time_estimator_has_a_measured_floor():
    """The tolerance sits above the instrument's bias, not below it.

    On the exact sampler the estimator overestimates `tau` by 10-26%
    systematically. `TAU_FREE` is 0.30 for that reason, and this test fails if
    the estimator ever gets so much worse that the tolerance stops covering it.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ref", pathlib.Path(__file__).parent / "fixtures" / "reference_stochastic.py"
    )
    ref = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(ref)  # type: ignore[union-attr]

    worst = 0.0
    for omega, gamma in ((1.0, 0.05), (3.0, 0.25), (1.5, 0.2)):
        out = ref.simulate(omega, gamma, 1.0, 400 / gamma, 12, 5)
        q, dt = np.asarray(out["q"]), out["dt"]
        taus = [S.autocorrelation_time(r, dt) for r in q]
        good = [t for t in taus if math.isfinite(t) and t > 0]
        worst = max(worst, abs(float(np.median(good)) - 1 / gamma) * gamma)
    assert worst < S.TAU_FREE, f"the estimator's floor {worst:.3f} exceeds TAU_FREE"
    assert S.TAU_GATE > 2 * worst


def test_rice_crossing_rate_uses_the_mode_frequency_exactly():
    """`sigma_v/sigma_u = w` because the cross-covariance vanishes."""
    w, g, d = 2.0, 0.1, 1.0
    sigma = math.sqrt(ou.variance(w, g, d))
    assert rice.crossing_rate(w, 0.0, sigma) == pytest.approx(w / (2 * math.pi))
    assert rice.crossing_rate(w, 2 * sigma, sigma) < rice.crossing_rate(w, sigma, sigma)


# --- the gate ---------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"q": [], "dt": 0.1},
        {"q": [[1.0] * 100] * 12},
        {"q": [[1.0] * 100] * 12, "dt": 0.0},
        {"q": [[1.0] * 10] * 12, "dt": 0.1},
    ],
)
def test_a_malformed_output_is_a_zero_with_a_reason(data):
    verdict = FAM.verify(_task(), AgentOutput(data=data))
    assert verdict.reward == 0.0 and "reason" in verdict.breakdown


def test_the_prompt_states_both_factors_of_two():
    """The convention that cost the ancestor project a correct solution."""
    for level in (1, 2, 3, 4):
        prompt = FAM.generate(level, np.random.default_rng(level)).prompt
        assert "2 G q'" in prompt
        assert "2 D delta" in prompt


def test_the_prompt_never_states_the_variance():
    for level in (1, 2, 3, 4):
        task = FAM.generate(level, np.random.default_rng(level))
        assert f"{_exact(task):.6f}" not in task.prompt
