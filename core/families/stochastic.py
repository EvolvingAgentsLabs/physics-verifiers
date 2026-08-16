"""Family 4 — STOCHASTIC. Semi-analytic truth, no LLM judge. Spec §3.

The differentiating family. Domains with stochastic rewards are the open problem
the market patches with LLM judges — soft verifiers watching soft verifiers — and
this is a clean answer for the case where the statistics are known exactly.

## The trick, and it is publishable

**The reward does not compare trajectories.** Trajectories are irreproducible;
demanding one would mean demanding a specific RNG. It compares **estimators
against their own sampling distribution**:

    z = (theta_hat - theta_exact) / SE(M, T)

An honest simulation with finite samples lands at |z| of order 1. That is the
whole point: the environment rewards *being right within your error bars*, not
being lucky.

## Why a fabricated exact answer fails

A candidate that skips simulating and returns the exact variance M times has
`z = 0` — better than honest. So `z` alone is not the check. Two more:

* **Inter-replicate spread.** The M replicates must scatter by the amount theory
  predicts. Returning the exact value M times gives zero spread, which is as
  impossible as being wildly off; the gate refuses `|z_spread| > 4` in *either*
  direction.
* **The autocorrelation shape.** A candidate that adds fabricated noise to hit
  the spread gets the variance and the scatter right and the correlation time
  wrong: OU decays as `exp(-|s|/tau)` with `tau = 1/gamma`, and white noise does
  not decay at all.

Being *too good* is a failure mode here, and that is unusual enough to be worth
saying plainly: a verifier that only checks "close enough" rewards fabrication
over simulation.

## The conventions, written out everywhere they matter

    q'' + 2 G q' + w^2 q = xi(t),      <xi(t) xi(t')> = 2 D delta(t - t')

Both factors of two are conventions. The ancestor of this project lost a correct
solution to exactly this — relative error 1.0 — because a prompt did not say.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core import attest, sandbox
from core.contracts import AgentOutput, TaskInstance, Verdict, failed
from core.truth import ou_lyapunov as ou
from core.truth import rice_rates as rice

NAME = "stochastic"

#: How many standard errors of disagreement still counts as honest sampling.
Z_GATE = 4.0

#: Disagreement, in standard errors, that costs nothing.
#:
#: **One.** An honest simulation lands at |z| of order 1 by definition — that is
#: what a standard error is. The first version scored `exp(-|z|/3)` per statistic
#: and multiplied three of them, so a *perfect* exact sampler came out at 0.27:
#: it was penalising the sampling noise it exists to tolerate.
#:
#: Beyond one standard error the reward decays, because at that point the
#: disagreement is no longer explained by sampling.
Z_FREE = 1.0
Z_SOFT = 1.5

#: The correlation time has no clean standard error, so its tolerance comes from
#: the estimator's **measured** floor rather than from theory.
#:
#: Measured on the exact sampler — no simulation error at all — the estimator
#: overestimates `tau` by 10% to 26% systematically, with 16-28% scatter across
#: replicates. The bias grows with `omega/gamma` and does **not** come from the
#: ACF normalisation: switching to the unbiased `1/(n-k)` form moved it from
#: 0.253 to 0.257, which is nothing. It is not chased further here.
#:
#: So `TAU_FREE` sits above the measured bias and the gate at roughly twice it.
#: The check's job in the trio is to catch a process with the wrong *shape* —
#: white noise has no decay at all and comes out at 100% — not to measure `G` to
#: a few percent. A tolerance below the instrument's floor would fail correct
#: solutions and tell nobody why.
TAU_FREE = 0.30
TAU_SOFT = 0.25
TAU_GATE = 0.60

BROKEN_SIMULATOR = """
import numpy as np

def simulate(omega, gamma, noise_intensity, duration, n_replicates, seed):
    rng = np.random.default_rng(seed)
    dt = 0.01 / omega
    n = int(duration / dt)
    out = []
    for r in range(n_replicates):
        q, v = 0.0, 0.0
        qs = np.empty(n)
        for k in range(n):
            # Euler-Maruyama. The noise increment for <xi xi> = 2 D delta
            # should have standard deviation sqrt(2 D dt) -- this one does not.
            xi = rng.normal(0.0, np.sqrt(noise_intensity * dt))
            v += (-2 * gamma * v - omega ** 2 * q) * dt + xi
            q += v * dt
            qs[k] = q
        out.append(qs[n // 2:].tolist())
    return {"q": out, "dt": dt}
"""


@dataclass(frozen=True)
class Stochastic:
    name: str = NAME

    def generate(self, level: int, rng: np.random.Generator) -> TaskInstance:
        if level not in (1, 2, 3, 4):
            raise ValueError(f"level must be 1..4, got {level}")
        params = _sample(level, rng)
        truth = {
            "variance": ou.variance(params["omega"], params["gamma"], params["noise_intensity"]),
            "tau": ou.autocorrelation_time(params["gamma"]),
        }
        return TaskInstance(
            family=NAME,
            level=level,
            prompt=_prompt(level, params),
            params=params,
            seed=params["seed"],
            manifest=attest.manifest(NAME, level, params, params["seed"], truth),
        )

    def verify(self, task: TaskInstance, output: AgentOutput) -> Verdict:
        return _verify(task, output)


def _sample(level: int, rng: np.random.Generator) -> dict[str, Any]:
    omega = round(float(rng.uniform(0.8, 3.0)), 6)
    # Underdamped, and slow enough that a run of a few hundred correlation times
    # is affordable: `tau = 1/gamma`.
    gamma = round(float(rng.uniform(0.05, 0.25)), 6)
    return {
        "omega": omega,
        "gamma": gamma,
        "noise_intensity": round(float(rng.uniform(0.5, 4.0)), 6),
        "duration": round(400.0 / gamma, 3),
        "n_replicates": 12,
        "seed": int(rng.integers(1, 2**31)),
        # `theta/sigma` above 1.5: Rice's Poisson approximation needs the level
        # to be high enough that crossings are rare and nearly independent.
        "threshold_ratio": round(float(rng.uniform(1.5, 2.5)), 4),
    }


_SCHEMA = (
    "Reply with a single Python module in one ```python block and nothing else. "
    "numpy and scipy are available; `mpmath` and `sympy` are not."
)


def _prompt(level: int, p: dict[str, Any]) -> str:
    extra = {
        1: "",
        2: "Use a scheme whose stationary statistics are correct, not merely stable.",
        3: (
            "The simulator below produces a stationary variance about **twice** the "
            "theoretical value. Find the missing factor and fix it.\n\n"
            f"```python{BROKEN_SIMULATOR}```\n"
        ),
        4: (
            "Use the **exact** Gaussian transition of the linear system rather than a "
            "time-stepping approximation, so that your statistics do not depend on `dt` "
            "at all. Report the `dt` you sampled at; a correct exact sampler gives the "
            "same statistics whatever it is."
        ),
    }[level]
    return f"""Simulate a damped harmonic oscillator driven by white noise:

    q'' + 2 G q' + w^2 q = xi(t),      <xi(t) xi(t')> = 2 D delta(t - t')

**Note both factors of two exactly as written**: the damping term carries a 2,
and the noise autocorrelation carries 2D.

    def simulate(omega, gamma, noise_intensity, duration, n_replicates, seed):
        ...
        return {{"q": [[...], ...], "dt": float}}

`q` is `n_replicates` independent trajectories of the displacement, each sampled
at a fixed step `dt`, **after** any burn-in you discard. You will be called with
omega={p['omega']:.6f}, gamma={p['gamma']:.6f} (this is G),
noise_intensity={p['noise_intensity']:.6f} (this is D), duration={p['duration']:.3f},
n_replicates={p['n_replicates']}, seed={p['seed']}.

{extra}

You are **not** scored on matching a reference trajectory — that would be
impossible. You are scored on three statistics against their exact values, in
units of their own sampling error:

  1. the stationary variance of q, against the Lyapunov solution;
  2. the scatter of that variance **across your replicates**, against the
     theoretical spread of the estimator;
  3. the autocorrelation time, against 1/G.

Point 2 means returning the exact answer with no spread fails as surely as being
wrong: a set of replicates that agree perfectly did not come from a simulation.

{_SCHEMA}"""


# --- the checks -------------------------------------------------------------


def autocorrelation_time(q: np.ndarray, dt: float) -> float:
    """Decay time of the ACF's **envelope**, fitted through its local maxima.

    For an underdamped oscillator the ACF is `exp(-G|s|) cos(w_d s)`, so two
    obvious approaches both fail:

    * *Integrating the ACF* gives nearly zero — the oscillation cancels it.
    * *Cutting the record where |acf| first falls below a floor* finds the first
      **zero of the cosine**, at a quarter period, long before the envelope has
      decayed at all. Measured: the cut landed at lag 2.49 for a process with
      `tau = 4.17`, leaving one peak in the window and returning `nan`.

    So the peaks are found first, over the whole record, and the fit runs through
    the ones still above a floor. The envelope is what carries `G`; the cosine
    carries the damped frequency and is not what is being asked for.
    """
    x = q - q.mean()
    n = x.size
    raw_acf = np.correlate(x, x, mode="full")[n - 1 :]
    if raw_acf[0] <= 0:
        return float("nan")
    # Unbiased normalisation: lag `k` has only `n - k` overlapping pairs, and
    # dividing them all by `n` shrinks the tail. Measured on exact data, the
    # biased form overestimated `tau` by 10-25% systematically — the shrunken
    # tail sits closer to the noise floor, the truncation keeps fewer peaks, and
    # the fit through them is flatter than the true decay.
    counts = np.arange(n, 0, -1, dtype=float)
    a = np.abs((raw_acf / counts) / (raw_acf[0] / counts[0]))

    raw = [i for i in range(1, a.size - 1) if a[i] >= a[i - 1] and a[i] >= a[i + 1]]

    # Keep only the leading run in which the envelope is actually decaying.
    #
    # The previous version collected every peak above 0.05 across the whole
    # record. Past a few correlation times those are the **noise floor**, which
    # is flat, so the fit ran through thousands of samples of nothing and
    # returned tau 700x too large. The comment said to stop at the first peak
    # that breaks the decay; the code did not do it.
    peaks: list[int] = []
    for i in raw:
        if a[i] < 0.05:
            break
        if peaks and a[i] > a[peaks[-1]] * 1.05:
            break  # rising again: the envelope has reached the noise
        peaks.append(i)
    if len(peaks) < 3:
        return float("nan")
    lags = np.asarray(peaks, dtype=float) * dt
    slope = np.polyfit(lags, np.log(a[peaks]), 1)[0]
    return float(-1.0 / slope) if slope < 0 else float("nan")


def _verify(task: TaskInstance, output: AgentOutput) -> Verdict:
    p = task.params
    q_raw = output.data.get("q")
    dt = output.data.get("dt")
    if not isinstance(q_raw, list) or not q_raw or not isinstance(dt, (int, float)):
        return failed("expected `q` (a list of trajectories) and `dt`", schema_error=True)
    if dt <= 0 or not math.isfinite(dt):
        return failed("`dt` must be a positive finite step", schema_error=True)
    try:
        q = np.asarray([[float(v) for v in row] for row in q_raw], dtype=float)
    except (TypeError, ValueError):
        return failed("the trajectories are not numeric", schema_error=True)
    if q.ndim != 2 or q.shape[0] != p["n_replicates"]:
        return failed(f"expected {p['n_replicates']} trajectories, got {q.shape}", schema_error=True)
    if not np.all(np.isfinite(q)):
        return failed("a trajectory is not finite")
    if q.shape[1] < 64:
        return failed("the trajectories are too short to estimate anything from")

    exact = ou.variance(p["omega"], p["gamma"], p["noise_intensity"])
    tau_exact = ou.autocorrelation_time(p["gamma"])

    # --- 1. the variance, in units of its own sampling error.
    per_replicate = q.var(axis=1, ddof=1)
    var_hat = float(per_replicate.mean())
    # Independent samples for a **variance** estimator: `q^2` decorrelates in
    # `tau/2`, not `tau`, so a run of length T carries about `2T/tau` of them.
    #
    # This was `T / (2 tau)` — a factor of four short — and the exact sampler
    # came out at `z_spread = -2.3`, i.e. accused of scattering half as much as
    # it should. Measured against 40 replicates of the exact sampler, the implied
    # `n_eff / (T/tau)` is 1.63, 1.71, 1.76 at T/tau = 100, 200, 400: close to
    # the derivable 2 and slightly under it, as expected since the ACF of `q^2`
    # is not exactly `exp(-2s/tau)`.
    #
    # Using the step count instead of an effective count is the classic way to
    # make an honest estimator look like a fraud.
    duration = q.shape[1] * float(dt)
    n_indep = max(2.0, 2.0 * duration / tau_exact)
    se_single = ou.standard_error_of_variance(exact, int(n_indep))
    se_mean = se_single / math.sqrt(p["n_replicates"])
    z_var = (var_hat - exact) / se_mean if se_mean > 0 else float("inf")

    # --- 2. the spread across replicates. Too small is as wrong as too large.
    observed_spread = float(per_replicate.std(ddof=1))
    # The sampling distribution of a spread: SE(s) ~ s / sqrt(2(M-1)).
    spread_se = se_single / math.sqrt(2.0 * (p["n_replicates"] - 1))
    z_spread = (observed_spread - se_single) / spread_se if spread_se > 0 else float("inf")

    # --- 3. the correlation time.
    taus = [autocorrelation_time(row, float(dt)) for row in q]
    good = [t for t in taus if math.isfinite(t) and t > 0]
    tau_hat = float(np.median(good)) if good else float("nan")
    tau_rel = abs(tau_hat - tau_exact) / tau_exact if math.isfinite(tau_hat) else float("inf")

    breakdown = {
        "variance_hat": var_hat,
        "variance_exact": exact,
        "z_variance": z_var,
        "spread_observed": observed_spread,
        "spread_expected": se_single,
        "z_spread": z_spread,
        "tau_hat": tau_hat,
        "tau_exact": tau_exact,
        "tau_relative_error": tau_rel,
        "n_independent_per_replicate": n_indep,
    }

    if not math.isfinite(z_var) or abs(z_var) > Z_GATE:
        return failed("the variance is not consistent with honest sampling", **breakdown)
    if not math.isfinite(z_spread) or abs(z_spread) > Z_GATE:
        # Both directions. Zero scatter is the signature of a fabricated answer,
        # and it is *further* from honest than being somewhat off.
        return failed(
            "the scatter across replicates is not what sampling would produce",
            fabrication_suspected=z_spread < -Z_GATE,
            **breakdown,
        )
    if not math.isfinite(tau_rel) or tau_rel > TAU_GATE:
        return failed("the autocorrelation does not decay at the rate 1/G", **breakdown)

    def _beyond(value: float, free: float, scale: float) -> float:
        return math.exp(-max(0.0, abs(value) - free) / scale)

    soft = (
        _beyond(z_var, Z_FREE, Z_SOFT)
        * _beyond(z_spread, Z_FREE, Z_SOFT)
        * _beyond(tau_rel, TAU_FREE, TAU_SOFT)
    )
    return Verdict(reward=float(min(1.0, soft)), hard_gate=True, breakdown=breakdown, evidence={})


def run_in_sandbox(task: TaskInstance, source: str, *, timeout_s: float = 300.0) -> AgentOutput:
    p = task.params
    result = sandbox.run_candidate(
        source,
        "simulate",
        {
            "omega": p["omega"],
            "gamma": p["gamma"],
            "noise_intensity": p["noise_intensity"],
            "duration": p["duration"],
            "n_replicates": p["n_replicates"],
            "seed": p["seed"],
        },
        timeout_s=timeout_s,
    )
    if not result.ok:
        return AgentOutput(data={"_error": result.error}, raw=source)
    return AgentOutput(data=result.data, raw=source)


__all__ = ["Stochastic", "BROKEN_SIMULATOR", "autocorrelation_time", "rice"]
