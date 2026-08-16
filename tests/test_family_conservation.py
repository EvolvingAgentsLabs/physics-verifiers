"""Family 2: the trio, and the cheat each leg of it kills.

This is the table that is figure 1 of the post, written as tests so it cannot
become a claim. Each cheat below is run against the real verifier and the
assertion names **which** check caught it — a cheat that died for the wrong
reason would mean the trio has a redundant leg and a hole somewhere else.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from core.contracts import AgentOutput
from core.families import conservation as C

REFERENCE = (pathlib.Path(__file__).parent / "fixtures" / "reference_conservation.py").read_text()
FAM = C.Conservation()


def _task(level: int = 2, seed: int = 2):
    return FAM.generate(level, np.random.default_rng(seed))


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_a_symplectic_integrator_passes_all_three(level):
    task = _task(level, level)
    verdict = FAM.verify(task, C.run_in_sandbox(task, REFERENCE))
    assert verdict.hard_gate, verdict.breakdown
    assert verdict.reward >= 0.9, verdict.breakdown
    assert verdict.breakdown["reverse_error"] < 1e-10, "Verlet is exactly reversible"


# --- the instrument's own floor ---------------------------------------------


def test_the_energy_estimator_includes_the_dirichlet_node():
    """Measured on an **exact** solution, so any drift is the estimator's.

    Excluding `x = 0` reports 1.5e-2 drift on a field with no integration error
    at all — `np.gradient` at the first interior node uses a one-sided
    difference and misses `u(0) = 0`, exactly where `u_x` is largest. It made a
    symplectic reference look like it was leaking 1.5% per run.
    """
    L, c, n = 1.2, 1.0, 400
    x = np.arange(1, n + 1) * (L / n)
    rng = np.random.default_rng(42)
    w = rng.uniform(0.4, 1.0, size=3) * np.array([1.0, 0.5, 0.25])
    ks = np.array([(2 * j - 1) * np.pi / (2 * L) for j in (1, 2, 3)])
    ts = np.linspace(0, 4 * (4 * L / c), 24)
    u = np.stack([sum(w[i] * np.cos(c * ks[i] * t) * np.sin(ks[i] * x) for i in range(3)) for t in ts])
    v = np.stack([sum(-w[i] * c * ks[i] * np.sin(c * ks[i] * t) * np.sin(ks[i] * x) for i in range(3)) for t in ts])

    e = C.energy(u, v, x, c)
    floor = float(np.max(np.abs(e - e[0])) / e[0])
    assert floor < 1e-4, f"the estimator's own floor is {floor:.2e}"
    # And the tolerance must sit above that floor, or it measures the estimator.
    assert C.TOL[4]["energy"] > floor * 5


def test_the_residual_window_is_fine_enough_in_time_to_mean_anything():
    """A second time difference over 5.8 snapshots per period is not `u_tt`.

    The first version measured the residual on the same coarse snapshots the
    energy uses and scored a correct symplectic reference at 1.95 against a 1e-2
    gate — it was measuring the sampling rate.
    """
    task = _task(2, 2)
    out = C.run_in_sandbox(task, REFERENCE)
    micro = out.data["_micro"]
    period = 4 * task.params["length"] / task.params["c"]
    spacing = micro["t"][1] - micro["t"][0]
    assert period / spacing > 100, f"only {period / spacing:.1f} samples per period"


# --- the trio, one cheat per leg --------------------------------------------


def _run(source: str, task=None):
    task = task or _task()
    return task, FAM.verify(task, C.run_in_sandbox(task, source))


ZEROS = """
import numpy as np
def evolve(n_cells, length, c, n_periods, samples, u0, v0):
    x = np.arange(1, n_cells + 1) * (length / n_cells)
    z = [[0.0] * n_cells for _ in range(samples)]
    return {"t": np.linspace(0, n_periods * 4 * length / c, samples).tolist(),
            "u": z, "v": z, "x": x.tolist()}
"""

FROZEN = """
import numpy as np
def evolve(n_cells, length, c, n_periods, samples, u0, v0):
    x = np.arange(1, n_cells + 1) * (length / n_cells)
    return {"t": np.linspace(0, n_periods * 4 * length / c, samples).tolist(),
            "u": [list(u0) for _ in range(samples)],
            "v": [[0.0] * n_cells for _ in range(samples)], "x": x.tolist()}
"""

WRONG_INITIAL = """
import numpy as np
def evolve(n_cells, length, c, n_periods, samples, u0, v0):
    x = np.arange(1, n_cells + 1) * (length / n_cells)
    k = np.pi / (2 * length); w = c * k
    ts = np.linspace(0, n_periods * 4 * length / c, samples)
    return {"t": ts.tolist(),
            "u": [(np.cos(w * t) * np.sin(k * x)).tolist() for t in ts],
            "v": [(-w * np.sin(w * t) * np.sin(k * x)).tolist() for t in ts],
            "x": x.tolist()}
"""


def test_a_field_of_zeros_conserves_energy_perfectly_and_scores_nothing():
    """Leg (i) alone is satisfied by `u = 0`. The initial-data check kills it."""
    _, verdict = _run(ZEROS)
    assert verdict.reward == 0.0
    assert "initial data" in verdict.breakdown["reason"]


def test_a_frozen_initial_state_is_killed_by_the_pde_and_not_by_energy():
    """It *starts* correctly and conserves energy; only the PDE notices.

    This is the cheat that legs (i) and (ii) both let through, and the reason
    leg (iii) exists.
    """
    _, verdict = _run(FROZEN)
    assert verdict.reward == 0.0
    assert "wave equation" in verdict.breakdown["reason"], verdict.breakdown


def test_a_correct_wave_with_the_wrong_initial_data_is_killed_by_the_initial_check():
    """It satisfies the PDE exactly and conserves energy exactly.

    Legs (i), (ii) and (iii) all pass. Only "did you solve *this* problem" fails,
    which is why the initial-data check is part of the trio rather than
    bookkeeping.
    """
    _, verdict = _run(WRONG_INITIAL)
    assert verdict.reward == 0.0
    assert "initial data" in verdict.breakdown["reason"]


def test_the_leaky_integrator_satisfies_the_pde_and_dies_on_energy():
    """The cheat leg (iii) alone would pass. Measured: residual 8.4e-3 < 2e-2."""
    task = _task(3, 3)
    verdict = FAM.verify(task, C.run_in_sandbox(task, C.LEAKY_INTEGRATOR))
    assert verdict.hard_gate, "it does satisfy the wave equation locally"
    assert verdict.breakdown["pde_residual"] < task.params["residual"]
    assert verdict.breakdown["energy_drift"] > 0.1
    assert verdict.breakdown["reverse_error"] > 0.1
    assert verdict.reward == 0.0


# --- the gate ---------------------------------------------------------------


def test_a_reverse_leg_that_did_not_run_is_not_a_pass():
    """`inf`, not `None`. A missing measurement must not read as a good one."""
    task = _task()
    out = C.run_in_sandbox(task, ZEROS)
    assert out.data["_reverse_error"] == float("inf") or out.data["_reverse_error"] > 0


def test_a_missing_fine_window_is_refused_rather_than_skipped():
    task = _task()
    data = dict(C.run_in_sandbox(task, REFERENCE).data)
    data.pop("_micro")
    verdict = FAM.verify(task, AgentOutput(data=data))
    assert verdict.reward == 0.0
    assert "could not be checked" in verdict.breakdown["reason"]


@pytest.mark.parametrize("missing", ["t", "u", "v", "x"])
def test_a_missing_field_is_a_schema_error(missing):
    task = _task()
    data = dict(C.run_in_sandbox(task, REFERENCE).data)
    data.pop(missing)
    verdict = FAM.verify(task, AgentOutput(data=data))
    assert verdict.reward == 0.0 and verdict.breakdown["schema_error"]


def test_the_prompt_states_every_tolerance_it_will_be_judged_by():
    """A hidden criterion is a task nobody can aim at."""
    for level in (1, 2, 3, 4):
        task = FAM.generate(level, np.random.default_rng(level))
        for key in ("energy", "reverse", "residual"):
            assert f"{task.params[key]:.0e}" in task.prompt, f"level {level} hides `{key}`"


def test_the_control_points_are_not_in_the_prompt():
    """A candidate that satisfied the PDE only where told would be interpolating."""
    task = _task()
    assert str(task.params["control_seed"]) not in task.prompt
