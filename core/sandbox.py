"""Run the agent's code where it can learn nothing. Spec §1.4.

The isolation is the product. A verifier the candidate can read is a verifier the
candidate can satisfy without solving anything, and every reward-hacking story in
RLVR reduces to that sentence.

## The split, stated once

* **This process** runs the candidate. It has numpy and scipy — everything needed
  to *solve* — and no truth: `core.truth`, `mpmath` and `sympy` are blocked
  before the candidate is imported. It runs in a temporary directory containing
  the candidate and nothing else.
* **The caller** holds the truth and never sees the candidate's source. The only
  thing that crosses is a JSON file with a fixed schema.

`core/` is not on the child's `sys.path`. That is not a filter to be bypassed by
a clever import: the files are not reachable.

## What "no network" means here, honestly

`PV_NO_NETWORK=1` is exported and the child's `socket` module is replaced by a
stub that raises. That stops an accidental download and an honest attempt to
fetch an answer. It is **not** a security boundary: a determined candidate can
`ctypes` its way around anything in-process. Spec §7-R4 says the same, and adds
the useful part — if a model breaks out, that is a publishable finding and a
patch, not an embarrassment. A real boundary is a container, and containers are
what the hub runtimes already provide.

Being precise about this matters more than sounding strong: a claim of "sandboxed"
that a reviewer can break in five minutes costs more credibility than the feature
buys.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNNER = Path(__file__).parent / "_runner.py"

#: What a candidate may import. Everything else it needs, it computes.
ALLOWED_HINT = "numpy, scipy, and the standard library"

#: Blocked because they are the *scorer's* tools. A candidate using mpmath would
#: be checked against its own method, and the gate would report agreement rather
#: than correctness.
BLOCKED = ("mpmath", "sympy", "core", "core.truth")


@dataclass(frozen=True)
class RunResult:
    ok: bool
    data: dict[str, Any]
    error: str = ""
    stdout: str = ""
    stderr: str = ""

    @property
    def failed_to_run(self) -> bool:
        """True when the candidate crashed, as opposed to answering wrongly.

        The caller must be able to tell these apart: a wrong answer is a zero
        reward, and a crash in *our* harness is a bug we must not train against.
        """
        return not self.ok


def run_candidate(
    source: str,
    entrypoint: str,
    kwargs: dict[str, Any],
    *,
    timeout_s: float = 60.0,
    memory_mb: int = 2048,
) -> RunResult:
    """Execute `source`, call `entrypoint(**kwargs)`, return what it produced.

    The result must be JSON-able. Anything a candidate prints is discarded —
    only the return value crosses, so a candidate cannot smuggle a claim through
    stdout and have it read as data.
    """
    with tempfile.TemporaryDirectory(prefix="pv-") as tmp:
        work = Path(tmp)
        (work / "candidate.py").write_text(source)
        (work / "call.json").write_text(json.dumps({"entrypoint": entrypoint, "kwargs": kwargs}))
        shutil.copy(RUNNER, work / "_runner.py")

        env = {
            k: v
            for k, v in os.environ.items()
            # PYTHONPATH would put the caller's `core/` back on the child's path,
            # which is the whole thing this function exists to prevent.
            if k not in ("PYTHONPATH", "PYTHONSTARTUP")
        }
        env["PV_NO_NETWORK"] = "1"
        env["PV_MEMORY_MB"] = str(memory_mb)

        try:
            proc = subprocess.run(
                [sys.executable, "-I", "_runner.py"],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return RunResult(ok=False, data={}, error=f"timed out after {timeout_s:g}s")

        out = work / "result.json"
        if not out.is_file():
            return RunResult(
                ok=False,
                data={},
                error=f"the candidate produced no result (exit {proc.returncode})",
                stdout=proc.stdout[-2000:],
                stderr=proc.stderr[-2000:],
            )
        payload = json.loads(out.read_text())
        return RunResult(
            ok=bool(payload.get("ok")),
            data=payload.get("value") if isinstance(payload.get("value"), dict) else {"value": payload.get("value")},
            error=str(payload.get("error", "")),
            stdout=proc.stdout[-2000:],
            stderr=proc.stderr[-2000:],
        )
