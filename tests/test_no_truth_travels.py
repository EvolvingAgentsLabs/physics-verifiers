"""Spec §1.1, enforced across every family at once.

`TaskInstance` is the object an adapter serialises and hands to an agent —
`prompt`, `params`, `seed` and `manifest` are all public by construction. The
rule is not "do not put the answer in the prompt"; it is **the truth does not
travel**, and `params` is the field where it would sneak through unnoticed,
because nothing about it looks like an answer.

This file exists because a review raised exactly that: a manifest or a params
dict read before execution could hand a candidate what it was supposed to
compute. It is a whole-repo check rather than one per family, so a sixth family
added later inherits it.

## What is deliberately allowed

`eigenmodes` level 4 states its targets — they **are** the question, an inverse
problem. What must not appear there is the hidden profile that realises them, and
that is checked separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.families.conservation import Conservation
from core.families.convergence import Convergence
from core.families.eigenmodes import Eigenmodes
from core.families.mms import MMS
from core.families.stochastic import Stochastic
from core.truth import exponential_string as expo
from core.truth import ou_lyapunov as ou
from core.truth import uniform_string as uni

FAMILIES = [Eigenmodes(), Conservation(), Convergence(), Stochastic(), MMS()]
LEVELS = (1, 2, 3, 4)


def _all_tasks():
    for family in FAMILIES:
        for level in LEVELS:
            yield family, family.generate(level, np.random.default_rng(level * 31 + 7))


def _numbers_in(obj) -> list[float]:
    """Every float anywhere in a nested structure."""
    out: list[float] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers_in(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    return out


def _truth_values(family_name: str, params: dict) -> list[float]:
    """What this task's answer actually is, computed independently here."""
    if family_name == "eigenmodes":
        if params.get("profile") == "inverse":
            return []  # its targets are the question; see the module docstring
        if params["alpha"] == 0.0:
            return uni.omegas(params["n_modes"], params["c"], params["length"])
        return expo.omegas(params["n_modes"], params["alpha"], params["length"], params["c"])
    if family_name == "convergence":
        if params["alpha"] == 0.0:
            return uni.omegas(params["n_modes"], params["c"], params["length"])
        return expo.omegas(params["n_modes"], params["alpha"], params["length"], params["c"])
    if family_name == "stochastic":
        return [
            ou.variance(params["omega"], params["gamma"], params["noise_intensity"]),
            ou.autocorrelation_time(params["gamma"]),
        ]
    return []


def test_the_manifest_carries_a_hash_and_never_a_value():
    """A hash is public-safe; a number is not."""
    for _, task in _all_tasks():
        assert set(task.manifest) == {
            "env_version", "family", "level", "params_hash", "seed", "truth_hash",
        }
        assert len(task.manifest["truth_hash"]) == 64
        assert len(task.manifest["params_hash"]) == 64


def test_no_task_parameter_is_the_answer():
    """The check the review asked for.

    `params` reaches the agent through any adapter that passes `info`, and
    nothing in it looks like an answer, so a truth leaking in there would be
    invisible in review.
    """
    for family, task in _all_tasks():
        truth = _truth_values(family.name, task.params)
        if not truth:
            continue
        present = _numbers_in(task.params)
        for want in truth:
            for got in present:
                assert abs(got - want) > 1e-9 * max(1.0, abs(want)), (
                    f"{family.name} L{task.level}: params contains the answer {want}"
                )


def test_no_prompt_states_the_answer():
    for family, task in _all_tasks():
        for want in _truth_values(family.name, task.params):
            assert f"{want:.6f}" not in task.prompt, f"{family.name} L{task.level} states {want}"
            assert f"{want:.4f}" not in task.prompt


def test_the_manufactured_solution_never_leaves_the_verifier():
    """MMS holds `u_m` in a process-local table keyed by the hash that ships.

    The forcing `F` is public — it must be, it is the question — and recovering
    `u_m` from it is exactly solving the problem.
    """
    fam = MMS()
    for level in LEVELS:
        task = fam.generate(level, np.random.default_rng(level))
        blob = repr(task.params) + task.prompt
        assert "u_expr" not in blob and "sin(" not in task.params.get("code", "").split("def F")[0]
        # The code block defines mu, K and F -- and nothing else.
        assert sorted(
            line.split("(")[0].removeprefix("def ").strip()
            for line in task.params["code"].splitlines()
            if line.startswith("def ")
        ) == ["F", "K", "mu"]


def test_the_inverse_problem_states_its_targets_and_hides_its_profile():
    """The one deliberate exception, pinned so it stays deliberate."""
    fam = Eigenmodes()
    task = fam.generate(4, np.random.default_rng(0))
    assert all(f"{t:.6f}" in task.prompt for t in task.params["targets"])
    assert f"{task.params['alpha']:.6f}" not in task.prompt


def test_every_family_generates_a_different_task_each_draw():
    """Spec §1.3. A static dataset is a dataset to memorise."""
    for family in FAMILIES:
        rng = np.random.default_rng(5)
        hashes = {family.generate(2, rng).manifest["params_hash"] for _ in range(8)}
        assert len(hashes) >= 7, f"{family.name} repeats itself"


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.name)
def test_every_family_refuses_a_level_it_does_not_have(family):
    with pytest.raises(ValueError):
        family.generate(5, np.random.default_rng(0))
