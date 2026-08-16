"""The internal contract. `core/` imports no hub interface, ever.

Spec §2.1 makes the core agnostic, and `docs/INTERFACE-NOTES.md` turned that from
tidiness into a load-bearing decision on day one: `pip install verifiers` gives
0.1.5, whose API is `Environment`/`Rubric`, while the repository's own
environments are already written against an unreleased `verifiers.v1` with
`Task`/`Runtime`/`@reward`. Two live APIs, one installable. The physics must not
move when that resolves.

So everything here is plain dataclasses and protocols. Adapters translate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

FAMILIES = ("eigenmodes", "conservation", "convergence", "stochastic", "mms")
LEVELS = (1, 2, 3, 4)


@dataclass(frozen=True)
class TaskInstance:
    """One generated task. **The truth is not in here.**

    Spec §1.1: the agent sees `prompt`; the verifier holds the truth. This object
    crosses to the agent, so anything in it is public — which is why `params`
    carries what the prompt already states and `manifest` carries only hashes.
    """

    family: str
    level: int
    prompt: str
    params: dict[str, Any]
    seed: int
    manifest: dict[str, Any]

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}; known: {', '.join(FAMILIES)}")
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {self.level}")
        if not self.prompt.strip():
            raise ValueError("a task with no prompt cannot be attempted")


@dataclass(frozen=True)
class Verdict:
    """What the verifier concluded, and enough to audit it without re-running.

    `hard_gate` is not advisory. Spec §1.2: a continuous score is only ever a
    *refinement* of a binary check, never a substitute — otherwise a solution
    that violates the physics still collects partial credit for being close, and
    "close" is precisely what a reward hacker optimises.
    """

    reward: float
    hard_gate: bool
    breakdown: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hard_gate and self.reward != 0.0:
            raise ValueError(
                f"hard_gate is False but reward is {self.reward}: a failed gate is a zero, "
                "and a Verdict that says otherwise is the bug this check exists to catch"
            )
        if not 0.0 <= self.reward <= 1.0:
            raise ValueError(f"reward must be in [0, 1], got {self.reward}")


@dataclass(frozen=True)
class AgentOutput:
    """What came back, already parsed. Schema errors are a `Verdict`, not a crash.

    Spec §2.1: an agent that ignores the schema gets reward 0 with
    `breakdown.schema_error` — which is training signal, so it must reach the
    reward rather than raising through it.
    """

    data: dict[str, Any]
    raw: str = ""


class Family(Protocol):
    """Generate a task, verify an answer. Four families will implement this."""

    name: str

    def generate(self, level: int, rng: Any) -> TaskInstance: ...

    def verify(self, task: TaskInstance, output: AgentOutput) -> Verdict: ...


def failed(reason: str, **breakdown: Any) -> Verdict:
    """A zero with a stated reason. Used for every refusal, so none is silent."""
    return Verdict(reward=0.0, hard_gate=False, breakdown={"reason": reason, **breakdown})
