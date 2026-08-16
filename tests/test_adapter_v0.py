"""The adapter against the `verifiers` API that installs. `docs/INTERFACE-NOTES.md`.

The translation is tested without `verifiers` itself wherever possible: the
physics has to run without the hub, and a test suite that needed the dependency
would make that claim untestable.
"""

from __future__ import annotations

import numpy as np
import pytest

from adapters import verifiers_v0 as A
from core.families import eigenmodes as E


def test_the_last_code_block_is_the_answer():
    """A model that reasons in code first must not be scored on its scratch work."""
    reply = "First I try:\n```python\nWRONG\n```\nBetter:\n```python\nRIGHT\n```"
    assert A.extract_code(reply).strip() == "RIGHT"
    assert A.extract_code("no fences here").strip() == "no fences here"
    assert A.extract_code([{"role": "assistant", "content": "```python\nX\n```"}]).strip() == "X"


def test_a_row_carries_the_prompt_and_the_manifest_and_no_truth():
    rows = A.build_rows("eigenmodes", levels=(1,), n=2, seed=3)
    assert len(rows) == 2
    for row in rows:
        info = row["info"]
        assert set(info) == {"family", "level", "prompt", "params", "seed", "manifest"}
        assert row["answer"] == "", "`answer` is a str and is the wrong channel for task data"
        truth = E._truth_omegas(info["params"])
        blob = str(info)
        for w in truth:
            assert f"{w:.6f}" not in blob


def test_the_reward_runs_the_real_verifier_end_to_end():
    row = A.build_rows("eigenmodes", levels=(1,), n=1, seed=5)[0]
    reference = (
        __import__("pathlib").Path(__file__).parent / "fixtures" / "reference_eigenmodes.py"
    ).read_text()
    reward = A.physics_reward(completion=f"```python\n{reference}\n```", info=row["info"])
    assert reward >= 0.99, reward


def test_a_fluent_wrong_answer_earns_nothing_through_the_adapter():
    """The same contrast the demo shows, but through the hub's own call path."""
    rows = A.build_rows("eigenmodes", levels=(1,), n=6, seed=11)
    row = next(r for r in rows if r["info"]["params"]["profile"] == "exponential")
    cheat = """
```python
import numpy as np
def eigenfrequencies(n_modes, length, c, alpha):
    # By Sturm-Liouville theory the fixed-free string admits odd quarter-waves.
    return {"omegas": [(2*n-1) * np.pi * c / (2*length) for n in range(1, n_modes+1)]}
```
"""
    assert A.physics_reward(completion=cheat, info=row["info"]) < 0.05


def test_missing_info_is_a_zero_rather_than_a_crash():
    """A hub runner that forgets `info` must not take the process down."""
    assert A.physics_reward(completion="```python\nx\n```", info=None) == 0.0


def test_mms_refuses_to_be_served_through_a_dataset_row():
    """Its manufactured solution cannot be rebuilt from public data, by design.

    Refused loudly rather than served with a verifier that would return zero for
    everyone — which is the shape this failure would otherwise take.
    """
    with pytest.raises(ValueError, match="one process"):
        A.load_environment(family="mms")
