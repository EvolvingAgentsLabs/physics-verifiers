"""Adapter for the `verifiers` API that actually installs: 0.1.5.

`docs/INTERFACE-NOTES.md` records the finding this file exists because of —
`pip install verifiers` gives an `Environment`/`Rubric` API with no `v1` module,
while the repository's own environments are written against an unreleased
`verifiers.v1`. Two live APIs. `core/` imports neither; this translates.

## The mapping, and the one decision in it

0.1.5 calls a reward function by keyword with whichever of
`prompt, completion, answer, state, task, info` its signature declares. Only
`info` carries arbitrary structure, so **the task rides in `info`**.

`answer` is a `str`. Forcing a parameter set or a manufactured solution through
it would mean serialising and re-parsing on both sides, and — worse — it would
send task data down the same channel the model's own answer arrives on. `info`
is the field designed for this and it is the one used.

## What crosses, and what does not

The dataset row carries `prompt` and `info = {family, level, params, manifest}`.
It does **not** carry the truth: `tests/test_no_truth_travels.py` asserts that
for every family, and this adapter adds no field of its own.

Import it only where `verifiers` is installed — `pip install -e ".[hub]"`.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from core.contracts import AgentOutput, TaskInstance
from core.families import conservation, convergence, eigenmodes, mms, stochastic

FAMILIES = {
    "eigenmodes": (eigenmodes.Eigenmodes(), eigenmodes.run_in_sandbox),
    "conservation": (conservation.Conservation(), conservation.run_in_sandbox),
    "convergence": (convergence.Convergence(), convergence.run_in_sandbox),
    "stochastic": (stochastic.Stochastic(), stochastic.run_in_sandbox),
    "mms": (mms.MMS(), mms.run_in_sandbox),
}


def extract_code(completion: Any) -> str:
    """The last fenced python block, or the whole reply.

    The **last** rather than the first: a model that reasons in code before
    giving its answer would otherwise be scored on its scratch work.
    """
    if isinstance(completion, list):
        text = "".join(
            m.get("content", "") for m in completion if isinstance(m, dict) and m.get("role") == "assistant"
        )
    else:
        text = str(completion or "")
    if "```" not in text:
        return text
    parts = text.split("```")
    blocks = []
    for i in range(1, len(parts), 2):
        block = parts[i]
        if block.startswith("python"):
            block = block[len("python") :]
        blocks.append(block.lstrip("\n"))
    return blocks[-1] if blocks else text


def _rebuild(info: dict[str, Any]) -> TaskInstance:
    """A `TaskInstance` from what the dataset row carried.

    Rebuilt rather than held in a side table because the reward function may run
    in a different process from the one that generated the row — which is the
    normal case for a hub runner, and a side table would silently return `None`
    there.

    MMS is the exception and says so: its manufactured solution cannot be
    reconstructed from public data, by design. Its environment must be generated
    and scored in one process.
    """
    return TaskInstance(
        family=info["family"],
        level=int(info["level"]),
        prompt=info.get("prompt", "(rebuilt)"),
        params=info["params"],
        seed=int(info.get("seed", 0)),
        manifest=info["manifest"],
    )


def physics_reward(completion: Any = None, info: dict[str, Any] | None = None, **_: Any) -> float:
    """The reward function 0.1.5's `Rubric` calls. Keyword-only, per its contract."""
    if not info:
        return 0.0
    family, runner = FAMILIES[info["family"]]
    task = _rebuild(info)
    output: AgentOutput = runner(task, extract_code(completion))
    return float(family.verify(task, output).reward)


def build_rows(family_name: str, levels: tuple[int, ...] = (1, 2, 3, 4), n: int = 8, seed: int = 0):
    """Dataset rows: `prompt` plus `info`. Nothing else, and no truth."""
    family, _ = FAMILIES[family_name]
    rng = np.random.default_rng(seed)
    rows = []
    for level in levels:
        for _ in range(n):
            task = family.generate(level, rng)
            rows.append(
                {
                    "question": task.prompt,
                    # `answer` is a str in 0.1.5 and is the wrong channel for task
                    # data. Left empty on purpose rather than stuffed.
                    "answer": "",
                    "info": {
                        "family": task.family,
                        "level": task.level,
                        "prompt": task.prompt,
                        "params": json.loads(json.dumps(task.params, default=float)),
                        "seed": task.seed,
                        "manifest": task.manifest,
                    },
                }
            )
    return rows


def load_environment(family: str = "eigenmodes", levels: tuple[int, ...] = (1, 2, 3, 4), n: int = 8, **kwargs):
    """Build a `SingleTurnEnv` over one family. Requires `verifiers==0.1.5`.

    Imported inside the function so that `core/` and the tests never need the
    dependency — the physics has to run without the hub.
    """
    import verifiers as vf  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415

    if family == "mms":
        raise ValueError(
            "mms cannot be served through a dataset row: its manufactured solution is held "
            "by the verifier and cannot be rebuilt from public data. Generate and score it "
            "in one process."
        )
    dataset = Dataset.from_list(build_rows(family, levels, n))
    rubric = vf.Rubric(funcs=[physics_reward], weights=[1.0])
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric, **kwargs)
