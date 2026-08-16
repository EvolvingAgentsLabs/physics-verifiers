"""Family 1, end to end: generation, the gate, the reference, and the cheats.

Two of the three tests that define the product (spec §2.2) live here — the
reference scores ~1 and trivial outputs score ~0. The third,
`test_truth_crosscheck`, is its own file.

The reward is `exp(-err/tol)`, which approaches 1 and never reaches it. So "the
reference scores ~1" is asserted as `>= 0.99`, and the docstring says so rather
than leaving a reader to wonder why an exact match scores 0.9988.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.contracts import AgentOutput
from core.families import eigenmodes as E
from core.truth import exponential_string as expo
from core.truth import uniform_string as uni

REFERENCE = (__import__("pathlib").Path(__file__).parent / "fixtures" / "reference_eigenmodes.py").read_text()
FAM = E.Eigenmodes()


def _task(level: int, seed: int = 0):
    return FAM.generate(level, np.random.default_rng(seed))


# --- generation -------------------------------------------------------------


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_a_task_carries_no_answer(level):
    """Spec §1.1. The prompt travels to the agent; the truth must not.

    Checked by looking for the actual eigenvalues in the prompt at the precision
    the tolerance cares about, not by trusting that nobody interpolated one in.
    """
    task = _task(level)
    if level == 4:
        # Level 4 states its targets on purpose — they *are* the question. What
        # must not appear is the hidden profile that realises them.
        assert f"{task.params['alpha']:.6f}" not in task.prompt
        return
    for w in E._truth_omegas(task.params):
        assert f"{w:.6f}" not in task.prompt, f"the prompt states the answer {w}"


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_the_manifest_hashes_the_truth_and_does_not_carry_it(level):
    task = _task(level)
    m = task.manifest
    assert set(m) == {"env_version", "family", "level", "params_hash", "seed", "truth_hash"}
    assert len(m["truth_hash"]) == 64
    assert m["family"] == "eigenmodes" and m["level"] == level


def test_instances_differ_so_nothing_is_memorisable():
    """Spec §1.3. Two draws must not be the same task."""
    rng = np.random.default_rng(11)
    seen = {FAM.generate(1, rng).manifest["params_hash"] for _ in range(12)}
    assert len(seen) == 12


def test_both_profiles_get_sampled_and_the_prompt_never_says_which():
    rng = np.random.default_rng(3)
    profiles = {FAM.generate(1, rng).params["profile"] for _ in range(20)}
    assert profiles == {"uniform", "exponential"}


# --- the reference ----------------------------------------------------------


#: What a solver would write after reading each broken solver. One per category,
#: because a static fixture cannot read the code it was shown and a text broad
#: enough to match all four would be a text that says nothing.
DIAGNOSES = {
    "neumann_order": "The free end owns a full cell of mass instead of half, which makes the Neumann boundary first-order.",
    "coefficient_at_nodes": "K is sampled at the nodes rather than at the cell faces, so the flux between cells uses the wrong coefficient.",
    "grid_offset": "The Dirichlet node is left in as an unknown, adding a spurious mode, so an off-by-one shifts every reported eigenvalue.",
    "asymmetric_matrix": "The generalised problem is not symmetrised by the mass matrix, so it solves the wrong operator.",
}


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_reference_solution_scores_about_one(level, seed):
    """Spec §2.2's first defining test. `>= 0.99` because `exp(-err/tol) < 1`."""
    task = _task(level, seed)
    verdict = FAM.verify(task, E.run_in_sandbox(task, REFERENCE))
    assert verdict.hard_gate, verdict.breakdown
    assert verdict.reward >= 0.99, verdict.breakdown


@pytest.mark.parametrize("bug", list(E._INJECT))
def test_the_reference_earns_both_halves_of_level_three(bug):
    """Fix and diagnosis are scored separately, so both are checked separately."""
    task = next(t for s in range(40) if (t := _task(3, s)).params["bug"] == bug)
    source = REFERENCE.replace('DIAGNOSIS = "no diagnosis was supplied"', f"DIAGNOSIS = {DIAGNOSES[bug]!r}")
    verdict = FAM.verify(task, E.run_in_sandbox(task, source))
    assert verdict.hard_gate, verdict.breakdown
    assert verdict.breakdown["diagnosis_named_the_cause"], bug
    assert verdict.reward >= 0.99, verdict.breakdown


def test_a_fix_with_no_diagnosis_earns_exactly_half():
    """The split is the point: a lucky edit is not the same as understanding."""
    task = _task(3, 0)
    verdict = FAM.verify(task, E.run_in_sandbox(task, REFERENCE))
    assert verdict.hard_gate
    assert 0.49 <= verdict.reward <= 0.51, verdict.breakdown


def test_a_diagnosis_that_names_everything_does_not_earn_the_half():
    """A text listing all four causes must not match the one that was injected.

    Otherwise the cheapest strategy is to recite the bug bank, and the diagnosis
    half of the reward measures nothing.
    """
    task = _task(3, 0)
    everything = " ".join(DIAGNOSES.values())
    source = REFERENCE.replace('DIAGNOSIS = "no diagnosis was supplied"', f"DIAGNOSIS = {everything!r}")
    verdict = FAM.verify(task, E.run_in_sandbox(task, source))
    assert not verdict.breakdown["diagnosis_named_the_cause"], (
        "reciting every cause earns the diagnosis half"
    )


# --- the injected bugs ------------------------------------------------------


@pytest.mark.parametrize("bug", list(E._INJECT))
def test_every_injected_bug_is_actually_detectable(bug):
    """A 'broken' solver that passes the gate is not a task.

    The first `grid_offset` shifted the grid by one cell out of 2000 and changed
    the answer by 3.1e-6 — under the 1e-4 tolerance. The candidate had nothing to
    find and the reward measured whether it wrote a diagnosis at all.
    """
    namespace: dict = {}
    exec(E.broken_solver(bug), namespace)  # noqa: S102 -- our own fixture
    got = namespace["eigenfrequencies"](6, 1.0, 1.0, 2.0)["omegas"]
    want = expo.omegas(6, 2.0, 1.0, 1.0)
    err = max(abs(g - w) / w for g, w in zip(got, want))
    assert err > E.TOL[3], f"{bug} is invisible at the gate's tolerance ({err:.2e})"


def test_the_level_three_prompt_shows_the_bug_it_grades():
    """They were independent once, so a correct diagnosis scored 0.5 half the time."""
    for seed in range(16):
        task = _task(3, seed)
        assert E.broken_solver(task.params["bug"]) in task.prompt


# --- the hard gate ----------------------------------------------------------


@pytest.mark.parametrize(
    "values,why",
    [
        ([], "no `omegas`"),
        ([1.0, 2.0], "wrong count"),
        ([-1.0] * 8, "not positive"),
        ([5.0] * 8, "not increasing"),
        ([float("nan")] * 8, "not finite"),
    ],
)
def test_the_gate_refuses_what_it_should(values, why):
    task = _task(1, 0)
    n = task.params["n_modes"]
    payload = values if len(values) in (0, 2) else values[:n]
    verdict = FAM.verify(task, AgentOutput(data={"omegas": payload}))
    assert not verdict.hard_gate and verdict.reward == 0.0, why


def test_a_missing_schema_is_a_zero_with_a_reason_rather_than_a_crash():
    """Spec §2.1: schema errors are training signal, so they must reach the reward."""
    verdict = FAM.verify(_task(1), AgentOutput(data={"answer": "42"}))
    assert verdict.reward == 0.0 and verdict.breakdown["schema_error"]


# --- anti-gaming ------------------------------------------------------------


def test_trivial_outputs_earn_nothing():
    """Spec §2.2's second defining test. The environment gives away no signal."""
    task = _task(1, 5)
    n = task.params["n_modes"]
    truth = E._truth_omegas(task.params)
    cheats = {
        "zeros": [0.0] * n,
        "ones": [1.0] * n,
        "integers": [float(i + 1) for i in range(n)],
        "the parameters echoed back": [task.params["c"] * (i + 1) for i in range(n)],
        "the textbook uniform formula": uni.omegas(n, task.params["c"], task.params["length"]),
    }
    for name, values in cheats.items():
        verdict = FAM.verify(task, AgentOutput(data={"omegas": values}))
        if task.params["profile"] == "uniform" and name == "the textbook uniform formula":
            continue  # on a uniform instance that is the correct answer, not a cheat
        assert verdict.reward < 0.05, f"{name} earned {verdict.reward}"


def test_the_textbook_formula_only_works_where_it_is_the_answer():
    """The point of sampling the profile, stated as a measurement.

    A solution that only knows `(2n-1)pi c / 2L` scores on the uniform draws and
    fails the exponential ones — which is the signal a training run should get,
    rather than a uniform zero or a uniform pass.
    """
    rng = np.random.default_rng(9)
    scores = {"uniform": [], "exponential": []}
    for _ in range(14):
        task = FAM.generate(1, rng)
        n = task.params["n_modes"]
        guess = uni.omegas(n, task.params["c"], task.params["length"])
        scores[task.params["profile"]].append(FAM.verify(task, AgentOutput(data={"omegas": guess})).reward)
    assert scores["uniform"] and min(scores["uniform"]) >= 0.99
    assert scores["exponential"] and max(scores["exponential"]) < 0.05


# --- level 4, the inverse problem -------------------------------------------


def _reference_module():
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "ref", pathlib.Path(__file__).parent / "fixtures" / "reference_eigenmodes.py"
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_inverse_problem_has_an_exact_solution_and_the_reference_finds_it(seed):
    """A task nobody can solve is not a task.

    The targets were first taken from the exponential closed form, where K varies
    too. Level 4 fixes K **constant**, so those eigenvalues are unreachable by
    any mu at all: the reference scored 0.0 with 24-52% error and the failure was
    in the generator. Targets now come from the same forward model the verifier
    uses, so an exact answer exists — and the reference lands at 1e-11.
    """
    task = _task(4, seed)
    ref = _reference_module()
    out = ref.profile(task.params["length"], 801, targets=task.params["targets"], K=task.params["c"] ** 2)
    verdict = FAM.verify(task, AgentOutput(data=out))
    assert verdict.reward >= 0.99, verdict.breakdown
    assert verdict.breakdown["max_relative_error"] < 1e-8


def test_the_forward_solve_the_verifier_uses_reproduces_the_targets():
    """The generator and the verifier must be the same model, not two models."""
    task = _task(4, 0)
    x = np.linspace(0.0, task.params["length"], 801)
    hidden = np.exp(-task.params["alpha"] * x)
    got = E.solve_profile(hidden, task.params["length"], task.params["c"] ** 2, 5)
    for g, w in zip(got, task.params["targets"]):
        assert abs(g - w) / w < 1e-12


@pytest.mark.parametrize(
    "mu,why",
    [
        ([1.0] * 800, "wrong length"),
        ([0.0] + [1.0] * 800, "not strictly positive"),
        ([-1.0] * 801, "negative"),
        (["x"] * 801, "not numeric"),
    ],
)
def test_the_inverse_gate_refuses_an_unphysical_profile(mu, why):
    verdict = FAM.verify(_task(4, 0), AgentOutput(data={"mu": mu}))
    assert not verdict.hard_gate and verdict.reward == 0.0, why


def test_a_flat_profile_does_not_accidentally_hit_the_targets():
    """The cheapest guess must not score. Anti-gaming, level 4."""
    task = _task(4, 0)
    verdict = FAM.verify(task, AgentOutput(data={"mu": [1.0] * 801}))
    assert verdict.hard_gate  # it is a legal profile
    assert verdict.reward < 0.05, verdict.breakdown
