"""The child process. Copied beside the candidate; imports nothing of ours.

Runs with `python -I` (isolated: no user site-packages, no `PYTHONPATH`, cwd not
on `sys.path` implicitly) in a directory holding `candidate.py`, `call.json` and
this file. It writes `result.json` and exits.

**It knows no truth**, and that is structural rather than enforced: `core/` is
not reachable from here, so there is nothing to block. The blocking below is for
the two libraries a candidate might reach *through pip* — `mpmath` and `sympy`
are the scorer's own methods, and a candidate using them would be checked against
its own tool.
"""

from __future__ import annotations

import json
import resource
import sys
import traceback
from pathlib import Path

BLOCKED = ("mpmath", "sympy", "core")


class _Blocked:
    """A module object that raises on any use, with a reason.

    Set into `sys.modules` rather than removed from disk: `import mpmath` then
    fails at import with a message naming why, instead of succeeding and being
    silently wrong later.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, item: str):
        raise ImportError(
            f"{self._name} is not available to a solution: it is the verifier's own method, "
            "and a solution checked against its own tool proves agreement rather than correctness"
        )


def _limit_memory() -> None:
    import os

    mb = int(os.environ.get("PV_MEMORY_MB", "2048"))
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        want = mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (want, hard if hard != resource.RLIM_INFINITY else want))
    except (ValueError, OSError):
        # macOS refuses RLIMIT_AS in some configurations. Reported, not fatal:
        # a missing memory cap is a weaker sandbox, not a wrong answer, and
        # pretending it was applied would be the worse failure.
        print("warning: could not apply a memory limit", file=sys.stderr)


def _block_network() -> None:
    """Replace `socket` with a stub that raises. Not a security boundary.

    Stops an accidental download and an honest fetch. `core/sandbox.py` says
    plainly that a determined candidate can get around anything in-process, and
    that the real boundary is a container.
    """
    import types

    stub = types.ModuleType("socket")

    def _refuse(*_a, **_k):
        raise OSError("network access is disabled in this environment")

    stub.socket = _refuse  # type: ignore[attr-defined]
    stub.create_connection = _refuse  # type: ignore[attr-defined]
    stub.getaddrinfo = _refuse  # type: ignore[attr-defined]
    sys.modules["socket"] = stub


def main() -> int:
    _limit_memory()
    if __import__("os").environ.get("PV_NO_NETWORK") == "1":
        _block_network()
    for name in BLOCKED:
        sys.modules[name] = _Blocked(name)  # type: ignore[assignment]

    call = json.loads(Path("call.json").read_text())
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("candidate", "candidate.py")
        if spec is None or spec.loader is None:
            raise ImportError("candidate.py could not be loaded")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        fn = getattr(module, call["entrypoint"], None)
        if not callable(fn):
            raise AttributeError(f"the solution must define {call['entrypoint']}(...)")
        value = fn(**call["kwargs"])
        payload = {"ok": True, "value": _jsonable(value)}
    except BaseException as exc:  # noqa: BLE001 -- a candidate may fail in any way
        # Recorded, never raised. The caller must be able to tell "wrong answer"
        # from "did not run"; an exception here would look like infrastructure.
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }
    Path("result.json").write_text(json.dumps(payload, allow_nan=False))
    return 0


def _jsonable(value):
    """Convert numpy and friends; refuse anything that is not data.

    `allow_nan=False` at write time means a NaN becomes an error here rather than
    a token only Python reads. That is the same rule the ancestor project learned
    the hard way, where four cross-language tests reported themselves as skipped.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    raise TypeError(f"a solution returned {type(value).__name__}, which is not data")


if __name__ == "__main__":
    raise SystemExit(main())
