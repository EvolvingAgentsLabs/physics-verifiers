"""The generator, as the product. Throughput, uniqueness, and a real difficulty knob.

`experiments/judge_vs_physics/` measured an LLM judge as good at detection as the
verifier — twice, on blatant fabrications and on subtle numerical defects. What
that leaves standing is the thing the judge cannot do: **create a task with a
known answer.** So the generator is what gets tested like a product.

Three claims, each measured rather than asserted:

1. it produces tasks faster than a training loop consumes them;
2. it does not repeat itself, so there is nothing to memorise;
3. `difficulty` is a property of the tasks, not a label on them.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from core import attest
from core.truth import mms_engine as M

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "scripts" / "generate_corpus.py"


@pytest.mark.parametrize("operator", sorted(M.OPERATORS))
@pytest.mark.parametrize("difficulty", M.DIFFICULTIES)
def test_every_operator_and_difficulty_manufactures_a_sound_instance(operator, difficulty):
    """Nine combinations, each self-checked through the code it ships."""
    rng = np.random.default_rng(hash((operator, difficulty)) % 2**31)
    m = M.manufacture(rng, M.sample_length(rng), operator=operator, difficulty=difficulty)
    assert M.self_check(m, rng) is None


def test_the_self_check_runs_the_rendered_code_and_not_only_the_symbols():
    """The stronger claim: it validates what a candidate actually receives.

    A code-generation bug — a dropped term, a `math.` that should be `np.` — is
    invisible to a symbolic residual and fatal to the task. Corrupting the
    rendered forcing must be caught.
    """
    rng = np.random.default_rng(3)
    m = M.manufacture(rng, M.sample_length(rng))
    assert M.self_check(m, rng) is None

    broken = M.Manufactured(
        operator=m.operator, u_expr=m.u_expr, mu_expr=m.mu_expr, K_expr=m.K_expr,
        b_expr=m.b_expr, length=m.length, difficulty=m.difficulty,
        code=m.code.replace("def F(x):\n    return ", "def F(x):\n    return 1.001 * ("
                            ).replace("\n", ")\n", -1) if False else m.code.replace(
            "def F(x):\n    return ", "def F(x):\n    return 0.999 * "),
        f_expr=m.f_expr, meta=m.meta,
    )
    why = M.self_check(broken, rng)
    assert why is not None and "disagrees" in why


def test_a_solution_that_breaks_a_boundary_is_rejected():
    """The envelope is even for a reason: an odd one breaks `u'(L) = 0`."""
    import sympy as sp

    rng = np.random.default_rng(1)
    good = M.manufacture(rng, sp.Integer(1))
    bad = M.Manufactured(
        operator=good.operator,
        u_expr=good.u_expr + sp.sin(sp.pi * M.X),   # slope at L is not zero
        mu_expr=good.mu_expr, K_expr=good.K_expr, b_expr=good.b_expr,
        length=good.length, difficulty=good.difficulty, code=good.code,
        f_expr=good.f_expr, meta=good.meta,
    )
    assert M.self_check(bad, rng) is not None


def test_it_does_not_repeat_itself():
    """Spec §1.3. A generator that collides is a dataset with extra steps."""
    rng = np.random.default_rng(11)
    seen = set()
    for i in range(60):
        op = sorted(M.OPERATORS)[i % 3]
        m = M.manufacture(rng, M.sample_length(rng), operator=op, difficulty=1 + i % 3)
        seen.add(attest.digest({"code": m.code, "u": str(m.u_expr)}))
    assert len(seen) == 60


def test_throughput_is_fast_enough_to_be_a_supply():
    """Measured, because a generator's unit cost is its contract.

    It was 0.28 s/task, then 14.9 s for a wave instance at difficulty 3 once the
    operator library grew — `simplify` on the forcing. Expanding instead, and
    checking numerically, took it to about 0.04 s. This asserts a loose bound so
    it fails on a regression of that kind rather than on a slow machine.
    """
    rng = np.random.default_rng(0)
    start = time.time()
    n = 20
    for i in range(n):
        op = sorted(M.OPERATORS)[i % 3]
        m = M.manufacture(rng, M.sample_length(rng), operator=op, difficulty=1 + i % 3)
        assert M.self_check(m, rng) is None
    per_task = (time.time() - start) / n
    assert per_task < 1.0, f"{per_task:.2f}s per task"


def test_difficulty_is_a_property_and_not_a_label():
    """One **fixed** solver, unchanged across levels. Its error must rise.

    Measured through the corpus script, which is also what publishes the number.
    A knob that does not move the error is a knob that names something.
    """
    proc = subprocess.run(
        [sys.executable, str(CORPUS), "--n", "120", "--seed", "5"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    errors: dict[tuple[str, int], float] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] in M.OPERATORS and parts[1].isdigit():
            errors[(parts[0], int(parts[1]))] = float(parts[3])

    steady = [op for op in M.OPERATORS if op not in M.TIME_DEPENDENT]
    assert steady, "no steady operator to measure"
    for op in steady:
        vals = [errors.get((op, d)) for d in (1, 2, 3)]
        assert all(v is not None for v in vals), f"{op}: {vals}"
        assert vals[0] < vals[1] < vals[2], f"{op}: difficulty is not monotone: {vals}"


def test_the_corpus_script_reports_rejections_rather_than_dropping_them():
    """A factory whose reject rate is invisible is a factory nobody can trust."""
    proc = subprocess.run(
        [sys.executable, str(CORPUS), "--n", "30"], capture_output=True, text=True, cwd=ROOT
    )
    assert proc.returncode == 0
    assert "rejected by the self-check:" in proc.stdout
    assert "collisions:" in proc.stdout
    assert "tasks/s" in proc.stdout
