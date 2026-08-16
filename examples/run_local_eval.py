#!/usr/bin/env python3
"""Evaluate a solution against the catalogue, locally, with a manifest per task.

    python examples/run_local_eval.py --family eigenmodes --levels 1 2 --n 3
    python examples/run_local_eval.py --all --solution my_solver.py
    python examples/run_local_eval.py --all --json report.json

The week-one gate from the spec: *a third party can run this and get a report
with manifests*. No API key, no network, no account.

With no `--solution` it runs the repo's reference fixtures, which is a smoke test
of the catalogue itself: every family should come back near 1. With one, it runs
your file against every selected task.

## What this is not

It does not call a model. Wiring an API key in would make the first thing a
visitor runs cost them money and require a login, and the thing worth seeing —
that the reward is computed by physics — does not need a model at all.
`examples/demo_refinement.py` shows the loop where a model is in it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import attest  # noqa: E402
from core.families import conservation, convergence, eigenmodes, mms, stochastic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

CATALOGUE = {
    "eigenmodes": (eigenmodes.Eigenmodes(), eigenmodes.run_in_sandbox, "reference_eigenmodes.py"),
    "conservation": (conservation.Conservation(), conservation.run_in_sandbox, "reference_conservation.py"),
    "convergence": (convergence.Convergence(), convergence.run_in_sandbox, "reference_convergence.py"),
    "stochastic": (stochastic.Stochastic(), stochastic.run_in_sandbox, "reference_stochastic.py"),
    "mms": (mms.MMS(), mms.run_in_sandbox, "reference_mms.py"),
}

#: Levels the bundled references are written for. Level 3 of `eigenmodes` needs a
#: diagnosis matched to the injected bug and level 4 needs the inverse solve, so
#: the smoke test skips them rather than reporting a low score as a catalogue
#: problem. Named here rather than silently filtered.
REFERENCE_LEVELS = {
    "eigenmodes": (1, 2),
    "conservation": (1, 2, 3, 4),
    "convergence": (1, 2, 3),
    "stochastic": (1, 2, 3, 4),
    "mms": (1, 2, 3, 4),
}


def evaluate(name: str, levels: list[int], n: int, source: str | None, seed: int) -> list[dict]:
    family, runner, fixture = CATALOGUE[name]
    code = source if source is not None else (FIXTURES / fixture).read_text()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for level in levels:
        for _ in range(n):
            task = family.generate(level, rng)
            output = runner(task, code)
            verdict = family.verify(task, output)
            rows.append(
                {
                    **attest.evaluation(task.manifest, output.data, verdict.reward, verdict.breakdown),
                    "family": name,
                    "level": level,
                    "hard_gate": verdict.hard_gate,
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(CATALOGUE), action="append")
    ap.add_argument("--all", action="store_true", help="every family")
    ap.add_argument("--levels", type=int, nargs="+")
    ap.add_argument("--n", type=int, default=2, help="instances per level")
    ap.add_argument("--solution", type=Path, help="a Python file; omit to run the bundled references")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, help="write the full report here")
    args = ap.parse_args()

    names = sorted(CATALOGUE) if args.all or not args.family else args.family
    source = args.solution.read_text() if args.solution else None

    print(f"{'family':<14}{'level':>6}{'reward':>9}{'gate':>7}   detail")
    print("-" * 78)
    rows: list[dict] = []
    for name in names:
        levels = args.levels or list(REFERENCE_LEVELS[name] if source is None else (1, 2, 3, 4))
        for row in evaluate(name, levels, args.n, source, args.seed):
            rows.append(row)
            b = row["breakdown"]
            detail = b.get("reason") or ", ".join(
                f"{k}={v:.2e}" for k, v in b.items()
                if isinstance(v, float) and k not in ("tolerance",)
            )[:44]
            print(
                f"{row['family']:<14}{row['level']:>6}{row['reward']:>9.4f}"
                f"{'PASS' if row['hard_gate'] else 'FAIL':>7}   {detail}"
            )

    print("-" * 78)
    mean = sum(r["reward"] for r in rows) / len(rows) if rows else 0.0
    gated = sum(1 for r in rows if not r["hard_gate"])
    print(f"{len(rows)} tasks · mean reward {mean:.4f} · {gated} refused at the gate")
    if source is None:
        print("(bundled references; levels the fixtures are written for — see REFERENCE_LEVELS)")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, sort_keys=True))
        print(f"report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
