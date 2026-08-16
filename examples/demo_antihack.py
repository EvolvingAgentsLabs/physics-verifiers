#!/usr/bin/env python3
"""Three agents, one task, three verdicts. Run this first.

    python examples/demo_antihack.py

The point in one screen: **an LLM judge scores confidence, and this scores
physics.** The middle agent below writes a fluent, correct-sounding derivation
citing Sturm-Liouville theory and returns the textbook quarter-wave formula. On
the instance it is handed, that formula is wrong — the profile is exponential and
its eigenvalues are the roots of a transcendental equation with a freshly sampled
parameter. A judge reading the prose would mark it well. The verifier gives it
zero, and says by how much it missed.

Nothing here is staged: the task is generated with a random seed, the truth is
computed at 50 digits by a module the candidates cannot reach, and the honest
agent's code runs in the same sandbox as the others.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import AgentOutput  # noqa: E402
from core.families import eigenmodes as E  # noqa: E402
from core.truth import uniform_string as uni  # noqa: E402

BAR = "=" * 72


HONEST = """
import numpy as np
from scipy.linalg import eigh_tridiagonal

def eigenfrequencies(n_modes, length, c, alpha):
    def eigs(n_cells):
        dx = length / n_cells
        x = np.arange(1, n_cells + 1, dtype=float) * dx
        K = lambda t: (c ** 2) * np.exp(-alpha * t)
        lower, upper = K(x - dx / 2), K(x[:-1] + dx / 2)
        volume = np.full(n_cells, dx); volume[-1] = dx / 2      # the apex owns half a cell
        d = np.empty(n_cells)
        d[:-1] = (upper + lower[:-1]) / dx
        d[-1] = lower[-1] / dx
        e = -upper / dx
        inv = 1.0 / np.sqrt(np.exp(-alpha * x) * volume)
        lam, _ = eigh_tridiagonal(d * inv * inv, e * inv[:-1] * inv[1:],
                                  select="i", select_range=(0, n_modes - 1))
        return np.sqrt(np.maximum(lam, 0.0))
    a, b = eigs(4000), eigs(8000)
    return {"omegas": ((4 * b - a) / 3).tolist()}       # Richardson: second order
"""

PLAUSIBLE_PROSE = (
    "By the Sturm-Liouville theory of the fixed-free string, the eigenvalue problem "
    "separates and the boundary conditions u(0)=0 and u'(L)=0 admit only the odd "
    "quarter-wave harmonics. Applying the standard result, the angular frequencies "
    "are w_n = (2n-1) pi c / 2L, which I evaluate below. The spectrum is real, "
    "discrete and simple, as the theory guarantees."
)


def main() -> int:
    rng = np.random.default_rng(20260816)
    family = E.Eigenmodes()

    # Draw until we get an exponential instance -- on a uniform one the textbook
    # formula IS the answer, and the demo would be showing the wrong thing.
    while True:
        task = family.generate(1, rng)
        if task.params["profile"] == "exponential":
            break

    print(BAR)
    print(" PHYSICS-VERIFIERS  ·  three agents, one task")
    print(BAR)
    print("\nTHE TASK (what the agent sees)\n" + "-" * 72)
    print(task.prompt.strip()[:900] + "\n...")
    print("-" * 72)
    p = task.params
    print(f"sampled: L={p['length']:.6f}  c={p['c']:.6f}  alpha={p['alpha']:.6f}  n={p['n_modes']}")
    print("The eigenvalues are roots of a transcendental equation with that alpha.")
    print("They are in no table and no training set. Either you compute them or you do not.\n")

    n = p["n_modes"]
    truth = E._truth_omegas(p)

    agents: list[tuple[str, str, AgentOutput]] = []

    # A -- solves it.
    agents.append(("A · honest solver", "wrote a second-order solver and extrapolated",
                   E.run_in_sandbox(task, HONEST)))

    # B -- writes a confident derivation and returns the wrong formula.
    agents.append((
        "B · fluent and wrong",
        "cited Sturm-Liouville and used the uniform-string formula",
        AgentOutput(data={
            "explanation": PLAUSIBLE_PROSE,
            "omegas": uni.omegas(n, p["c"], p["length"]),
        }),
    ))

    # C -- ignores the schema.
    agents.append(("C · malformed", "returned a different key, as a string",
                   AgentOutput(data={"eigenvalues": str([1.2, 2.3, 3.4])})))

    print("THE VERDICTS\n" + "-" * 72)
    for name, what, output in agents:
        verdict = family.verify(task, output)
        print(f"\n{name}")
        print(f"  did:      {what}")
        print(f"  reward:   {verdict.reward:.4f} / 1.0     hard gate: {'PASS' if verdict.hard_gate else 'FAIL'}")
        err = verdict.breakdown.get("max_relative_error")
        if err is not None:
            print(f"  error:    {err:.3e}  (full marks below {verdict.breakdown['tolerance']:.0e})")
        if "reason" in verdict.breakdown:
            print(f"  refused:  {verdict.breakdown['reason']}")
        if name.startswith("B"):
            print("  note:     a judge reading the prose above would score this well.")
            print(f"            its first eigenvalue is {output.data['omegas'][0]:.4f}"
                  f" against the true {truth[0]:.4f}.")

    print("\n" + "-" * 72)
    print("AUDIT")
    print(f"  truth hash:   {task.manifest['truth_hash'][:32]}...")
    print(f"  params hash:  {task.manifest['params_hash'][:32]}...")
    print("  The manifest carries hashes and no values: it travels with the task.")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
