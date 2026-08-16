# INTERFACE-NOTES — step 0 of the spec, done by reading installed code

**Date:** 2026-08-16 · **Everything below is `[ran]`**, not recalled.

Spec §2.1 makes this the first thing to do and says to read the installed code
rather than tutorials. Doing so found the thing §7-R1 predicted, on day one.

## The finding: there are two `verifiers` APIs, and only one installs

```
pip index versions verifiers
  verifiers (0.1.5)
  Available versions: 0.1.5, 0.1.5.dev1, 0.1.5.dev0, 0.0.0
```

`pip install verifiers` gives **0.1.5**, whose package layout is

```
verifiers/{envs,parsers,rubrics,trainers,inference,utils,types.py}
```

with no `v1` module and no `Task`, `Runtime` or `@reward`. Its exports are
`Environment`, `MultiTurnEnv`, `SingleTurnEnv`, `Rubric`, `JudgeRubric`, `Parser`.

**But the repository's own environments are written against a different API.**
`PrimeIntellect-ai/verifiers@main:environments/gsm8k/gsm8k/taskset.py` opens with
`import verifiers.v1 as vf` and uses `vf.TaskData`, `vf.Task`, `vf.Taskset`,
`vf.Runtime`, `@vf.reward(weight=...)`, `runtime.run_uv_script(...)`,
`Task.setup()` and `Task.validate()`. **None of that exists in 0.1.5**, and no
released version on PyPI has it.

So: the released API is the old rubric/environment shape; `v1` is an unreleased
rewrite that the repo has already migrated its own examples to. A tutorial, a
model's memory, or the docs site will each confidently describe one of the two.

## What this decides

1. **`core/` imports neither.** Spec §2.1 already required an interface-agnostic
   core; this makes it load-bearing rather than tidy. The physics does not move
   when an unreleased rewrite lands.
2. **Two adapters, not one.** `adapters/verifiers_v0.py` against 0.1.5 — what a
   third party can `pip install` and run **today** — and `adapters/verifiers_v1.py`
   against the shape the repo's own environments use, which is what the hub will
   want. Neither is speculative: both were read.
3. **The version is pinned and the pin is dated.** `verifiers==0.1.5` in
   `pyproject.toml`. When `v1` releases, this file gets a new section rather than
   an edit — the point is to be able to see *when* the ground moved.

## The 0.1.5 contract, verbatim from the installed source

`verifiers/types.py:38`

```python
RewardFunc = Callable[..., float | Awaitable[float]]
```

Rubric calls a reward function by **keyword**, passing whichever of these its
signature declares (`rubrics/rubric.py`):

```python
prompt: Messages, completion: Messages, answer: str, state: State,
task: str = "default", info: Info | None = None, **kwargs
```

`Environment.__init__` (`envs/environment.py:47`) takes
`dataset`, `eval_dataset`, `system_prompt`, `few_shot`, `parser`, `rubric`,
`sampling_args`, `message_type`, `oai_tools`, `max_workers`.

`SingleTurnEnv` is `MultiTurnEnv` with `is_completed` true after one response —
seven lines, no logic of its own.

**Where our per-instance data goes:** `info`. It is the only field carrying
arbitrary structure into a reward function, so a `TaskInstance`'s `params`,
`seed` and `manifest` ride there. `answer` is a `str` and is the wrong shape for
a manufactured solution or a parameter set; forcing them through it would mean
serialising and re-parsing, with the truth passing through the same channel as
the prompt.

## The v1 contract, verbatim from the repo

```python
import verifiers.v1 as vf

class GSM8KData(vf.TaskData): answer: str
class GSM8KTask(vf.Task[GSM8KData]):
    async def setup(self, runtime: vf.Runtime) -> None: ...
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace, runtime: vf.Runtime) -> float: ...
    async def validate(self, runtime: vf.Runtime) -> bool: ...
class GSM8KTaskset(vf.Taskset[GSM8KTask, GSM8KConfig]):
    def load(self) -> list[GSM8KTask]: ...
```

Two things in it are worth taking whether or not we use the API:

* **`validate()`** — "is this task well-formed", checked model-free by running the
  verifier on the ground truth. It is the same idea as our
  `test_reference_agent`, and it is the check that catches a task whose scorer
  disagrees with its own prompt.
* **`runtime.run_uv_script`** — the verifier's dependencies are installed by `uv`
  *inside the rollout's runtime*, so they never touch the eval process. That is
  a cleaner version of our §1.4 isolation and we should copy the shape: the
  scorer and the candidate as two processes with disjoint dependency sets.

## Prior art we already have, and should not rebuild

`ai-os/projects/coclea-sr/environments/coclea_sr/` (paused repo) is a working
v1-shaped environment with four graded physics tasks. It is the direct ancestor
of this repo and it already paid for three lessons:

1. A scorer that **re-derived** its truth got the transcendental roots wrong and
   scored a *correct* solution 0.0. One derivation, bundled and drift-checked.
2. A prompt that never stated its noise convention cost a factor of two —
   relative error exactly 1.0 on a correct answer. **Every convention gets
   written out.**
3. The reference solution's free end was first-order (a full cell of mass at the
   boundary instead of half). It scored 0.89 where 1.0 was needed, and Richardson
   extrapolation made it *worse* — the giveaway. The fix took the error from
   2.5e-5 to 5.4e-9.

Measured there: reference `1.0 1.0 1.0 1.0`, a cheating solution trying three
real routes `0.0 0.0 0.0 0.0`.
