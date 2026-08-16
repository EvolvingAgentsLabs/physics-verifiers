#!/usr/bin/env python3
"""The verifier as a signal a model can act on: best-of-N, then refinement.

    python examples/demo_refinement.py                  # scripted, no API key
    python examples/demo_refinement.py --model <name>   # a real model

## Why this and not a training run

Closing an RL loop — PPO, GRPO, weight updates — would spend the whole budget on
CUDA and hyperparameters and would prove the wrong thing. What a buyer needs to
see is that **the reward is dense, discriminating and actionable**: that it
separates a correct solve from a fluent one, and that its `breakdown` is specific
enough for a model to fix its own answer from.

That is testable in inference, in seconds, for the price of a few API calls, and
it is the property any RL algorithm downstream depends on.

## The two loops

**Best-of-N.** Sample N attempts at temperature, score them all, keep the best.
The verifier is the selector. If the scores are all identical the reward carries
no signal; if they spread, it does.

**Refinement.** Feed the verdict back into the prompt and ask for a fix. The
measurement is whether the reward *rises* — which is only possible if the
breakdown says something a model can act on. A reward that only ever says
"wrong" cannot drive this.

## What runs without a key

The scripted mode replays three attempts — a plausible-but-wrong one, a nearly
right one, and a correct one — through the **real** verifier. The loop mechanics
and every number are real; only the source of the attempts is scripted. With
`--model` the attempts come from a model instead, and the rest is unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.families import eigenmodes as E  # noqa: E402

BAR = "=" * 72

# --- the scripted attempts, in the order a model typically produces them ----

ATTEMPT_TEXTBOOK = """
import numpy as np
def eigenfrequencies(n_modes, length, c, alpha):
    # The fixed-free string admits odd quarter-wave harmonics.
    return {"omegas": [(2*n-1) * np.pi * c / (2*length) for n in range(1, n_modes+1)]}
"""

ATTEMPT_FIRST_ORDER = """
import numpy as np
from scipy.linalg import eigh_tridiagonal
def eigenfrequencies(n_modes, length, c, alpha):
    N = 800
    dx = length / N
    x = np.arange(1, N + 1, dtype=float) * dx
    K = lambda t: (c**2) * np.exp(-alpha * t)
    lower, upper = K(x - dx/2), K(x[:-1] + dx/2)
    volume = np.full(N, dx)                      # the apex gets a full cell
    d = np.empty(N)
    d[:-1] = (upper + lower[:-1]) / dx
    d[-1] = lower[-1] / dx
    e = -upper / dx
    inv = 1.0 / np.sqrt(np.exp(-alpha * x) * volume)
    lam, _ = eigh_tridiagonal(d*inv*inv, e*inv[:-1]*inv[1:], select="i", select_range=(0, n_modes-1))
    return {"omegas": np.sqrt(np.maximum(lam, 0.0)).tolist()}
"""

ATTEMPT_CORRECT = ATTEMPT_FIRST_ORDER.replace(
    "volume = np.full(N, dx)                      # the apex gets a full cell",
    "volume = np.full(N, dx); volume[-1] = dx/2   # the apex owns HALF a cell",
)


def _call_model(model: str, prompt: str, temperature: float) -> str:
    """One completion from an OpenAI-compatible endpoint.

    Not exercised in this repository's tests — there is no key in CI, and a test
    that silently skipped would be worse than one that does not exist. Marked
    `[read]` rather than `[ran]` until somebody runs it with a key.
    """
    import json
    import urllib.request

    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("--model needs OPENROUTER_API_KEY or OPENAI_API_KEY in the environment")
    base = os.environ.get("PV_BASE_URL", "https://openrouter.ai/api/v1")
    body = json.dumps(
        {"model": model, "temperature": temperature, "messages": [{"role": "user", "content": prompt}]}
    ).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
        payload = json.load(resp)
    return payload["choices"][0]["message"]["content"] or ""


def _extract(text: str) -> str:
    if "```" not in text:
        return text
    parts = text.split("```")
    blocks = [p[len("python"):].lstrip("\n") if p.startswith("python") else p for p in parts[1::2]]
    return blocks[-1] if blocks else text


def _score(family, task, source: str):
    verdict = family.verify(task, E.run_in_sandbox(task, source))
    return verdict


def _feedback(verdict) -> str:
    """The verdict, as text a model can act on.

    Only what the verifier measured. Adding a hint here would make the loop a
    demonstration of the hint rather than of the reward.
    """
    b = verdict.breakdown
    if "reason" in b:
        return f"The verifier refused your answer: {b['reason']}"
    return (
        f"Your answer was accepted but scored {verdict.reward:.4f}. "
        f"Its largest relative error against the exact eigenvalues is "
        f"{b['max_relative_error']:.3e}; full marks require below {b['tolerance']:.0e}. "
        f"The per-mode errors were {['%.1e' % e for e in b['per_mode_relative_error'][:5]]}."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="an OpenAI-compatible model name; omit for the scripted run")
    ap.add_argument("--n", type=int, default=3, help="best-of-N samples")
    ap.add_argument("--rounds", type=int, default=2, help="refinement rounds")
    args = ap.parse_args()

    family = E.Eigenmodes()
    rng = np.random.default_rng(7)
    while True:
        task = family.generate(2, rng)
        if task.params["profile"] == "exponential":
            break

    print(BAR)
    print(" THE VERIFIER AS A SIGNAL  ·  best-of-N, then refinement")
    print(BAR)
    p = task.params
    print(f"\ntask: eigenmodes level 2, exponential profile, alpha={p['alpha']:.4f}, n={p['n_modes']}")

    # --- best of N ---------------------------------------------------------
    print("\n1. BEST-OF-N — the verifier as selector\n" + "-" * 72)
    if args.model:
        sources = [_extract(_call_model(args.model, task.prompt, 0.8)) for _ in range(args.n)]
    else:
        sources = [ATTEMPT_TEXTBOOK, ATTEMPT_FIRST_ORDER, ATTEMPT_CORRECT][: args.n]

    scored = []
    for i, source in enumerate(sources, start=1):
        verdict = _score(family, task, source)
        scored.append((verdict.reward, source, verdict))
        note = verdict.breakdown.get("reason") or f"error {verdict.breakdown.get('max_relative_error', float('nan')):.2e}"
        print(f"  attempt {i}: reward {verdict.reward:.4f}   {note}")
    best = max(scored, key=lambda r: r[0])
    spread = max(r[0] for r in scored) - min(r[0] for r in scored)
    print(f"\n  best {best[0]:.4f}, spread {spread:.4f}")
    print("  A spread means the reward carries signal. Identical scores would mean it does not.")

    # --- refinement --------------------------------------------------------
    print("\n2. REFINEMENT — the verdict fed back\n" + "-" * 72)
    source = sources[1] if not args.model else sources[0]
    verdict = _score(family, task, source)
    print(f"  round 0: reward {verdict.reward:.4f}")
    print(f"           feedback -> {_feedback(verdict)[:120]}")

    for r in range(1, args.rounds + 1):
        if args.model:
            prompt = f"{task.prompt}\n\nYour previous answer:\n```python\n{source}\n```\n\n{_feedback(verdict)}\n\nFix it."
            source = _extract(_call_model(args.model, prompt, 0.2))
        else:
            # The scripted stand-in for "the model acted on the feedback".
            source = ATTEMPT_CORRECT
        verdict = _score(family, task, source)
        print(f"  round {r}: reward {verdict.reward:.4f}")
        if verdict.reward >= 0.99:
            break

    print("\n  The reward rose because the breakdown said something actionable.")
    print("  A verifier that only ever answered 'wrong' could not drive this loop.")
    if not args.model:
        print("\n  (scripted attempts; every number above came from the real verifier.")
        print("   `--model <name>` replaces the attempts with a real one.)")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
