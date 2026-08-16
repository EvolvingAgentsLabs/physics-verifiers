"""Light attestation: enough to reproduce and audit, not a ledger. Spec §1.7.

Each task instance carries `{env_version, family, level, params_hash, seed,
truth_hash}` and each evaluation emits `{manifest, agent_output_hash, reward,
breakdown}`. The hash-chained append-only ledger is v2 and is deliberately not
here — spec §7-R5 lists it under "what this spec does not include".

## Why `truth_hash` is in the manifest at all

It is not a channel to the agent: the manifest travels with the task, so anything
in it is public. A hash is public-safe by construction and answers the one
question an auditor asks — *was this scored against the truth the generator
produced, or against something that changed afterwards?* A number cannot answer
that; a hash of the number can.

## Canonical JSON, and why it is not `json.dumps`

Two processes hashing "the same" dict must produce the same bytes. Python's
default `json.dumps` sorts nothing and spaces everything, and floats print
differently across languages — `9.278e-06` in Python, `0.000009278` in
JavaScript. So: sorted keys, no spaces, and floats formatted with `repr`, which
round-trips exactly in Python and is the shortest string that does.

**Floats are hashed as their repr, not as text somebody typed.** A manifest whose
hash depends on how a number was printed is a manifest that changes when a
formatter does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

ENV_VERSION = "0.1.0"


def canonical(obj: Any) -> str:
    """Stable text for hashing. Sorted, compact, and float-safe."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_default)


def _default(o: Any) -> Any:
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    if hasattr(o, "tolist"):  # numpy arrays and scalars
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not canonicalisable; convert it before hashing")


def digest(obj: Any) -> str:
    """SHA-256 of the canonical form, hex."""
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def manifest(family: str, level: int, params: dict[str, Any], seed: int, truth: Any) -> dict[str, Any]:
    """The record that travels with a task.

    `truth` is hashed and discarded. Passing it here rather than its hash is
    deliberate: it makes the call site look like it is handing over the truth,
    which is exactly what a reader should check it is *not* doing further down.
    """
    return {
        "env_version": ENV_VERSION,
        "family": family,
        "level": level,
        "params_hash": digest(params),
        "seed": seed,
        "truth_hash": digest(truth),
    }


def evaluation(manifest_: dict[str, Any], agent_output: Any, reward: float, breakdown: dict[str, Any]) -> dict[str, Any]:
    """The record an evaluation emits. Reproducible from the manifest alone."""
    return {
        "manifest": manifest_,
        "agent_output_hash": digest(agent_output),
        "reward": reward,
        "breakdown": breakdown,
    }
