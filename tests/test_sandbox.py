"""The isolation, measured. Spec §1.4 and §7-R4.

Two properties, and only one of them is a security claim:

1. **A candidate cannot reach the verifier's tools.** `mpmath` and `sympy` are
   the scorer's own methods; a candidate using them would be checked against its
   own tool. `core` is not even on the child's path. This is structural and is
   asserted here.
2. **Network access is off.** Stated honestly in `core/sandbox.py` as *not* a
   security boundary — it stops an accident and an honest fetch, not a determined
   escape. Tested for what it claims, not for more.

The last test is the one that keeps the others meaningful: a control body that
scores fine on its own, so a zero can be attributed to the block rather than to
the body being wrong.
"""

from __future__ import annotations

from core import sandbox

GOOD = """
def solve(n):
    return {"values": [float(i) * 2 for i in range(n)]}
"""


def test_a_working_candidate_runs_and_its_return_value_crosses():
    r = sandbox.run_candidate(GOOD, "solve", {"n": 3})
    assert r.ok, r.error
    assert r.data == {"values": [0.0, 2.0, 4.0]}


def test_numpy_and_scipy_are_available_because_solving_needs_them():
    src = """
import numpy as np
from scipy.linalg import eigh_tridiagonal
def solve(n):
    d = np.full(n, 2.0); e = np.full(n - 1, -1.0)
    w, _ = eigh_tridiagonal(d, e, select="i", select_range=(0, 0))
    return {"lowest": float(w[0])}
"""
    r = sandbox.run_candidate(src, "solve", {"n": 8})
    assert r.ok, r.error
    assert 0.0 < r.data["lowest"] < 1.0


def test_the_control_body_scores_on_its_own():
    """Without this, every 'blocked' result below could be the body's fault."""
    body = 'def solve():\n    return {"x": 1.0}\n'
    r = sandbox.run_candidate(body, "solve", {})
    assert r.ok and r.data == {"x": 1.0}


def test_the_verifiers_own_tools_are_unreachable():
    """`mpmath` and `sympy` fail at import, naming why.

    The same body succeeds without the import (test above), so the failure is
    attributable to the block and not to the code.
    """
    for module in ("mpmath", "sympy"):
        src = f'import {module}\ndef solve():\n    return {{"x": 1.0}}\n'
        r = sandbox.run_candidate(src, "solve", {})
        assert not r.ok, f"{module} was importable"
        assert module in r.error


def test_the_truth_modules_are_not_on_the_path_at_all():
    """Not blocked — absent. There is nothing to bypass."""
    src = 'from core.truth import uniform_string\ndef solve():\n    return {"x": 1.0}\n'
    r = sandbox.run_candidate(src, "solve", {})
    assert not r.ok
    assert "core" in r.error or "No module named" in r.error


def test_network_access_is_off_for_what_it_claims_to_stop():
    src = """
import socket
def solve():
    socket.create_connection(("example.com", 80), timeout=1)
    return {"x": 1.0}
"""
    r = sandbox.run_candidate(src, "solve", {})
    assert not r.ok
    assert "network" in r.error.lower() or "disabled" in r.error.lower()


def test_a_crash_is_reported_as_a_crash_rather_than_raised():
    """The caller must tell 'wrong answer' from 'did not run'."""
    r = sandbox.run_candidate('def solve():\n    raise ValueError("boom")\n', "solve", {})
    assert not r.ok and "boom" in r.error
    assert r.failed_to_run


def test_a_missing_entrypoint_is_named():
    r = sandbox.run_candidate('def other():\n    return {}\n', "solve", {})
    assert not r.ok and "solve" in r.error


def test_an_infinite_loop_is_stopped():
    r = sandbox.run_candidate("def solve():\n    while True:\n        pass\n", "solve", {}, timeout_s=2.0)
    assert not r.ok and "timed out" in r.error


def test_a_non_data_return_value_is_refused():
    """A candidate returning an object, not data, is an error rather than a
    silent `str()` of something the verifier would then parse."""
    src = "class T:\n    pass\ndef solve():\n    return T()\n"
    r = sandbox.run_candidate(src, "solve", {})
    assert not r.ok and "not data" in r.error


def test_printing_does_not_reach_the_verifier():
    """Only the return value crosses, so nothing can be smuggled through stdout
    and read as data."""
    src = 'import json\ndef solve():\n    print(json.dumps({"values": [999.0]}))\n    return {"values": [1.0]}\n'
    r = sandbox.run_candidate(src, "solve", {})
    assert r.ok and r.data == {"values": [1.0]}
