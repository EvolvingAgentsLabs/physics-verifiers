#!/usr/bin/env python3
"""Generate a corpus of manufactured tasks, and measure what a buyer would ask.

    python scripts/generate_corpus.py --n 300
    python scripts/generate_corpus.py --n 2000 --out corpus.jsonl

Three numbers, none of them asserted anywhere else:

1. **Throughput** — tasks per second, including the per-instance self-check.
   A generator's unit cost is its contract.
2. **Collisions** — distinct tasks over total. A generator that repeats itself is
   a dataset with extra steps, and a dataset is a thing to memorise.
3. **Difficulty** — the observed error of one **fixed** second-order solver at a
   fixed resolution, per difficulty level. If it does not rise, the difficulty
   knob is a label rather than a property.

Every task is self-checked before it counts. A rejected instance is reported, not
dropped: a factory whose reject rate is invisible is a factory nobody can trust.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import attest  # noqa: E402
from core.truth import mms_engine as M  # noqa: E402


def reference_error(m: M.Manufactured, n_points: int = 401) -> float:
    """Max relative error of a fixed second-order finite-volume solve.

    Deliberately **one** solver, unchanged across difficulties — the measurement
    is of the tasks, not of a solver tuned per level. Steady operators only;
    the wave family needs a time integrator and is reported separately.
    """
    if m.time_dependent:
        return float("nan")
    ns: dict = {}
    exec(compile(m.code, "<mms>", "exec"), ns)  # noqa: S102
    mu, K = ns["mu"], ns["K"]
    b = ns.get("b")

    length = float(m.length)
    x = np.linspace(0.0, length, n_points)
    dx = x[1] - x[0]
    xi = x[1:]
    n = xi.size

    lower, upper = K(xi - dx / 2.0), K(xi[:-1] + dx / 2.0)
    volume = np.full(n, dx)
    volume[-1] = dx / 2.0

    main = np.empty(n)
    main[:-1] = (upper + lower[:-1]) / dx
    main[-1] = lower[-1] / dx
    main = main + mu(xi) * volume

    A = np.diag(main) + np.diag(-upper / dx, 1) + np.diag(-upper / dx, -1)
    if b is not None:
        # Centred first derivative, integrated over the cell like every other term.
        #
        # The apex row needs the Neumann ghost: with `u'(L) = 0` the ghost node
        # satisfies `u_{n+1} = u_{n-1}`, so the centred difference there is
        # `(u_{n-1} - u_{n-1}) / 2dx = 0` — the advection term **vanishes** at the
        # free end.
        #
        # Omitting that left the apex row computing `b (0 - u_{n-2}) / 2dx`, and
        # the measured error for the advection family came out at 1.9e-2 — a
        # thousand times the reaction-diffusion family at the same difficulty,
        # and non-monotone in difficulty. That reads as "these tasks are hard"
        # and is in fact "this solver is wrong at one node".
        bv = np.asarray(b(xi), dtype=float) * volume
        upper_b = bv[:-1] / (2 * dx)
        lower_b = bv[1:] / (2 * dx)
        lower_b[-1] = 0.0            # the apex: ghost node cancels the term
        A += np.diag(upper_b, 1) - np.diag(lower_b, -1)

    u = np.linalg.solve(A, np.asarray(ns["F"](xi), dtype=float) * volume)
    got = np.concatenate(([0.0], u))
    want = m.u(x)
    scale = float(np.max(np.abs(want))) or 1.0
    return float(np.max(np.abs(got - want)) / scale)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    operators = list(M.OPERATORS)
    hashes: set[str] = set()
    errors: dict[tuple[str, int], list[float]] = defaultdict(list)
    rejected: list[str] = []
    rows = []

    start = time.time()
    for i in range(args.n):
        op = operators[i % len(operators)]
        difficulty = int(rng.integers(1, 4))
        m = M.manufacture(rng, M.sample_length(rng), operator=op, difficulty=difficulty)
        why = M.self_check(m, rng)
        if why:
            rejected.append(f"{op} d{difficulty}: {why}")
            continue
        digest = attest.digest({"code": m.code, "u": str(m.u_expr)})
        hashes.add(digest)
        err = reference_error(m)
        if err == err:
            errors[(op, difficulty)].append(err)
        rows.append(
            {
                "operator": op,
                "difficulty": difficulty,
                "length": str(m.length),
                "task_hash": digest,
                "code": m.code,
                "reference_error": None if err != err else err,
                "meta": m.meta,
            }
        )
    elapsed = time.time() - start

    print(f"generated {len(rows)} tasks in {elapsed:.1f}s  ->  {len(rows)/elapsed:.1f} tasks/s")
    print(f"distinct: {len(hashes)}/{len(rows)}  collisions: {len(rows) - len(hashes)}")
    print(f"rejected by the self-check: {len(rejected)}")
    for r in rejected[:5]:
        print(f"  {r}")

    print(f"\n{'operator':<32}{'d':>3}{'n':>5}{'median error of ONE fixed solver':>36}")
    for op in operators:
        for d in (1, 2, 3):
            vals = errors.get((op, d), [])
            if vals:
                print(f"{op:<32}{d:>3}{len(vals):>5}{np.median(vals):>36.3e}")
            elif op in M.TIME_DEPENDENT:
                print(f"{op:<32}{d:>3}{'-':>5}{'(time-dependent; needs an integrator)':>36}")

    print("\nIf the error does not rise with d, the difficulty knob is a label.")
    if args.out:
        args.out.write_text("\n".join(json.dumps(r) for r in rows))
        print(f"\ncorpus written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
