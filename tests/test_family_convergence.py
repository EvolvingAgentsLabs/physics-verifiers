"""Family 3: the observed order, and the two ways of faking it that do not work.

The anti-hack argument is short enough to state in a test name: **you cannot fake
a slope you did not produce.** The verifier calls the candidate's solver at grid
sizes it chooses and measures how the error falls. Hard-coding the answer gives
zero error at every N — no slope at all, and the gate says so rather than
crediting it as infinitely convergent.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from core.contracts import AgentOutput
from core.families import convergence as C

HERE = pathlib.Path(__file__).parent / "fixtures"
REF2 = (HERE / "reference_convergence.py").read_text()
REF4 = (HERE / "reference_convergence_o4.py").read_text()
FAM = C.Convergence()


def _task(level: int, seed: int = 0):
    return FAM.generate(level, np.random.default_rng(seed))


@pytest.mark.parametrize("level", [1, 2, 3])
def test_a_second_order_solver_measures_second_order(level):
    task = _task(level, level)
    verdict = FAM.verify(task, C.run_in_sandbox(task, REF2))
    assert verdict.hard_gate, verdict.breakdown
    assert abs(verdict.breakdown["observed_order"] - 2.0) < 0.05
    assert verdict.reward >= 0.99


def test_a_fourth_order_solver_measures_fourth_order():
    task = _task(4, 4)
    verdict = FAM.verify(task, C.run_in_sandbox(task, REF4))
    assert verdict.hard_gate, verdict.breakdown
    assert abs(verdict.breakdown["observed_order"] - 4.0) < 0.1
    assert verdict.reward >= 0.95


def test_the_level_four_grids_stay_above_the_roundoff_floor():
    """The measurement must not report its own precision limit.

    Measured on 250-2000 cells, a *correct* fourth-order solver produced errors
    5.1e-10, 3.0e-11, 3.7e-11, 1.6e-10 — falling then **rising**, because
    Richardson hits the floating-point floor by the second grid. The fitted slope
    came out at 0.48 for a solver that is exactly right. Coarser grids fixed it;
    this pins that the finest one still shows discretisation error.
    """
    task = _task(4, 4)
    errors = C.run_in_sandbox(task, REF4).data["errors"]
    assert all(b < a for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] > 1e-12, "the finest grid is at the roundoff floor"


def test_the_degraded_solver_really_is_first_order():
    """The level-3 prompt claims 'order near 1'. That claim is checked here.

    The first version of the fixture also sampled K at the nodes and dropped a
    `dx`, which made it simply wrong — relative error 1.000 at every grid, no
    order at all. A prompt describing a defect the code does not have is a prompt
    the candidate cannot satisfy.
    """
    task = _task(3, 3)
    verdict = FAM.verify(task, C.run_in_sandbox(task, C.DEGRADED_SOLVER))
    assert verdict.hard_gate, verdict.breakdown
    assert abs(verdict.breakdown["observed_order"] - 1.0) < 0.1, verdict.breakdown
    assert verdict.reward < 0.05, "a first-order solver must not score on an order-2 task"


def test_the_only_difference_from_the_reference_is_the_half_cell():
    """One line, and it is the line the prompt is about."""
    a = [ln.strip() for ln in C.DEGRADED_SOLVER.splitlines() if ln.strip()]
    b = [ln.strip() for ln in REF2.splitlines() if ln.strip() and not ln.strip().startswith(('"', "#"))]
    differing = [ln for ln in a if ln not in b]
    assert any("volume = np.full" in ln for ln in differing), differing


# --- anti-gaming ------------------------------------------------------------


def test_a_hardcoded_answer_earns_nothing():
    """Zero error at every grid is not order infinity — it is no discretisation.

    The candidate is handed the exact eigenvalues in its own source, which is the
    strongest possible cheat, and still scores zero because the reward is the
    slope rather than the value.
    """
    task = _task(2, 2)
    truth = C._truth(task.params)
    source = f"""
def eigenfrequencies(n_modes, length, c, alpha, n_cells):
    return {{"omegas": {truth!r}}}
"""
    verdict = FAM.verify(task, C.run_in_sandbox(task, source))
    assert verdict.reward == 0.0
    assert verdict.breakdown.get("suspected_hardcoding"), verdict.breakdown


def test_a_wrong_answer_with_a_fake_second_order_term_earns_nothing():
    """The error must fall *to zero*, not oscillate around a wrong value.

    A candidate that returns `wrong * (1 + 1/N^2)` produces a converging
    *sequence* whose limit is wrong, so the error against the truth flattens at
    the offset rather than falling.
    """
    task = _task(2, 2)
    truth = C._truth(task.params)
    wrong = [v * 1.05 for v in truth]
    source = f"""
def eigenfrequencies(n_modes, length, c, alpha, n_cells):
    base = {wrong!r}
    return {{"omegas": [v * (1.0 + 1.0 / (n_cells ** 2)) for v in base]}}
"""
    verdict = FAM.verify(task, C.run_in_sandbox(task, source))
    assert verdict.reward < 0.05, verdict.breakdown


def test_errors_that_do_not_decrease_are_refused_before_a_slope_is_fitted():
    """Two errors in a 4:1 ratio give p = 2 whether or not either is small."""
    task = _task(2, 2)
    grids = C.grids_for(task.params["target_order"])
    verdict = FAM.verify(task, AgentOutput(data={"errors": [1e-3, 1e-2, 1e-1, 1.0]}))
    assert not verdict.hard_gate and verdict.reward == 0.0
    assert len(grids) == 4


def test_a_solver_that_crashes_is_reported_rather_than_scored():
    task = _task(1, 1)
    out = C.run_in_sandbox(task, "def eigenfrequencies(**kw):\n    raise RuntimeError('x')\n")
    verdict = FAM.verify(task, out)
    assert verdict.reward == 0.0 and verdict.breakdown.get("schema_error")


def test_the_prompt_never_states_the_eigenvalues():
    for level in (1, 2, 3, 4):
        task = _task(level, level)
        for w in C._truth(task.params):
            assert f"{w:.6f}" not in task.prompt
