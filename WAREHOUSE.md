# Email Triage RL -- Observability Warehouse

Event log + DuckDB/dbt warehouse for the Email Triage RL environment. This
is the #1 item in the data-engineering pivot (see project progress notes):
turn the eval harness's in-memory `reset()`/`step()` calls into durable,
queryable data, without paid services.

Zero new dependencies in the core server or `inference.py` -- the telemetry
module has zero third-party imports. Everything that needs `pyarrow` /
`duckdb` / `dbt-duckdb` lives in a separate `requirements-warehouse.txt`
and is only needed to run the batch/warehouse steps below, not to run the
graded environment itself.

## Architecture

```
EmailTriageEnvironment.step()  ──┐
  (server/Email_RL_environment)  ├─► telemetry/event_sink.py ─► data/raw/{env_steps,client_steps}/dt=YYYY-MM-DD/events.jsonl
run_episodes.py (recommended)  ──┤        (JSONL, append-only,
  or inference.py (hackathon's,    │        fail-open, per-stream lock)
  frozen, own scores still OK)  ──┘
                                                    │
                                    telemetry/compact_to_parquet.py
                                    (batch; moves source files into
                                     _compacted/ once written)
                                                    ▼
                                   data/lake/{env_steps,client_steps}/dt=.../*.parquet
                                                    │
                                          warehouse/ (dbt-duckdb)
                                          staging → intermediate → marts
                                                    ▼
                                     warehouse/warehouse.duckdb (query this)
```

Two independent event streams, joined downstream rather than coupled at
the source:

- **`env_steps`** -- logged server-side, inside `EmailTriageEnvironment.step()`,
  in the same process, before the HTTP boundary. Carries every reward
  component the environment actually computed for that email, whether or
  not anything client-side was listening.
- **`client_steps`** -- logged client-side, inside `run_episodes.py` (or
  `inference.py`)'s per-step loop. Carries the model name, task, the LLM's
  raw predicted fields, the task-specific grader's reward, and LLM call
  latency.

### Why `email_id` is the join key, not `episode_id`

`email_id` is a `uuid4()` generated server-side and present on every
observation regardless of session/HTTP mode. `episode_id` lives in
`metadata`, which is stripped by OpenEnv's HTTP layer in stateless mode. A
top-level `session_id` does survive over HTTP (captured by
`inference.py`'s client into `self._session_id`) and is logged as a
secondary field on `client_steps`, but `email_id` is the reliable primary
key across both streams -- see `warehouse/models/intermediate/int_steps_joined.sql`.

## Setup

```bash
pip install -r requirements-warehouse.txt
```

### 0. Pick an LLM provider

`inference.py`/`run_episodes.py` are provider-agnostic by design -- they
just build a plain `OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)`
client and call `MODEL_NAME`. Any OpenAI-compatible endpoint works with
zero code changes; only `.env` needs to change.

**Recommended: Groq** (`console.groq.com`, no credit card required) --
fast (300-1,000 tok/s on purpose-built inference hardware, so a full
multi-episode run doesn't take forever) and free-tier usage isn't flagged
for model training the way some alternatives are.

**Do not use a hidden-reasoning model (e.g. `openai/gpt-oss-120b`) on this
harness.** Confirmed the hard way: these models can spend their entire
`max_tokens` budget on an invisible reasoning channel before writing any
visible answer, returning a genuine HTTP 200 with empty `content` --
indistinguishable from a rate limit or connection failure except that
`diagnose_llm.py` reports `SUCCESS` with nothing after it. Raising
`MAX_TOKENS` does **not** reliably fix this -- in one test, going from 500
to 2000 made it *worse* (10/10 empty responses instead of a partial mix),
because the model just reasons longer with more room rather than reasoning
the same amount and using the rest for the answer. `inference.py`'s
`_call_llm` never sets `reasoning_effort`/`include_reasoning`, and it's
frozen, so there's no clean way to cap reasoning depth short of
monkeypatching the function itself. Simplest fix: don't use a
hidden-reasoning model here.

```
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
HF_TOKEN=<your Groq API key>
```

(`HF_TOKEN` as a name is a holdover from the original hackathon setup --
it's not validated against any particular key format, any provider's API
key works there.)

**Free-tier rate limits differ a lot by model, and it matters which task
you're running** (confirmed against Groq's published limits page):

| model | RPM | RPD | TPM | TPD |
|---|---|---|---|---|
| `llama-3.3-70b-versatile` | 30 | 1,000 | 12,000 | 100,000 |
| `llama-3.1-8b-instant` | higher | 14,400 | comparable/higher | 500,000 |

`llama-3.3-70b-versatile`'s 12K TPM ceiling is fine for the four
short-output tasks (spam-detection, priority-classification, full-triage,
critical-escalation -- just three XML tags), but is a **sustained, not
occasional**, wall for `action-orchestrator` and `threat-assessment`,
which both require a real JSON `action_plan`/`threat_report` in the
output. Confirmed empirically: even with `run_episodes.py`'s retry/backoff
(see below), individual steps needed the full 5s+15s+30s retry sequence
just to get a real response, and some still failed all three retries.

**If you hit this on those two tasks specifically**, you have two options:

1. Slow down and stay on `llama-3.3-70b-versatile` for a single-model
   comparison across all six tasks:
   ```bash
   python run_episodes.py --episodes 4 --tasks action-orchestrator,threat-assessment --request-delay 12
   ```
   (`--request-delay` paces LLM calls; 12s keeps you comfortably under the
   12K TPM ceiling for this task's heavier per-call token cost -- the
   default 1s pacing, tuned for the lighter tasks, is nowhere near enough
   here.)
2. Switch `MODEL_NAME` to `llama-3.1-8b-instant` for just these two tasks,
   which has a meaningfully larger free-tier quota (14,400 RPD / 500K TPD
   vs. 1,000 RPD / 100K TPD). **This project's own dataset uses this
   option** -- see "Rate limits and the two-model split" below for why
   that's a documented, deliberate choice rather than an inconsistency.

**Whatever provider you use, run `python diagnose_llm.py` before
`run_episodes.py`** and confirm it prints `SUCCESS` *and* a non-empty raw
response. `SUCCESS` alone isn't enough -- see the hidden-reasoning-model
warning above. This has now caught three separate real failures that
otherwise looked identical to "the model just isn't very compliant": an
exhausted API key, a `.env` override that silently didn't apply, and a
reasoning model quietly burning its budget -- see "Parser robustness"
below.

### 1. Generate telemetry

**Use `python run_episodes.py [--episodes N] [--tasks name1,name2]`**
(same env vars as `inference.py`: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`,
`EMAIL_RL_SERVER_URL`). This drives real, continuous multi-email episodes
over `client.py`'s WebSocket client -- see finding #0 above for why this
exists as a separate script from `inference.py`.

`inference.py` itself is left **completely untouched** -- byte-identical
to what the hackathon provided, with zero telemetry hooks added to it. You
can still run it directly for its own console-reported scores (those are
unaffected by finding #0), but it won't write any `client_steps` telemetry
at all, and the `env_steps` telemetry the server logs while `inference.py`
is the caller isn't analytically meaningful (see finding #0). Use
`run_episodes.py` for anything you actually want to feed into this
warehouse.

Every graded `step()` call appends a line to
`data/raw/env_steps/dt=<today>/events.jsonl`; every `run_episodes.py` step
appends a line to `data/raw/client_steps/dt=<today>/events.jsonl`. **The
server process and `run_episodes.py` are separate processes** -- make sure
they either share `EMAIL_RL_TELEMETRY_ROOT`, or are both left at the
default (`data/raw`, relative to each process's own CWD) and run from the
same working directory, or the compaction step below won't find both
streams together.

Override the landing-zone location with `EMAIL_RL_TELEMETRY_ROOT` (defaults
to `data/raw`, relative to wherever the process's CWD is).

### 2. Compact to Parquet

```bash
python telemetry/compact_to_parquet.py --data-root data/raw --lake-root data/lake
```

Idempotent: each `events.jsonl` is moved into a `_compacted/` sibling
folder once processed, so re-running never double-counts rows. Safe to run
on a cron/schedule against a live `data/raw/` directory.

### 3. Build the warehouse

```bash
cd warehouse
dbt run --profiles-dir .
```

This uses the project-local `profiles.yml` in `warehouse/`, so it never
touches `~/.dbt/`. Output lands in `warehouse/warehouse.duckdb`.

```bash
# Query it directly:
python3 -c "import duckdb; print(duckdb.connect('warehouse.duckdb').sql('select * from agg_model_leaderboard').df())"
# or:
duckdb warehouse.duckdb
```

## dbt model layering

- `staging/stg_env_steps.sql`, `staging/stg_client_steps.sql` -- thin
  selects off the Parquet lake via `read_parquet(..., hive_partitioning=true)`.
- `intermediate/int_steps_joined.sql` -- `stg_env_steps` LEFT JOIN
  `stg_client_steps` on `email_id` (LEFT, not INNER -- env_steps is
  authoritative; a client_steps write can silently fail since the sink is
  fail-open by design, and that shouldn't drop the env-side row).
- `marts/fct_reward_components.sql` -- unpacks the `reward_components` JSON
  string into individual numeric columns via `json_extract_string(...)::double`.
- `marts/fct_confusion_matrix.sql` -- predicted-vs-true counts across
  priority/category/route, grouped by `model_name` x `task`.
- `marts/agg_model_leaderboard.sql` -- accuracy per field, perfect-match
  rate, phishing catch rate, per `model_name` x `task`.
- `marts/agg_episode_summary.sql` -- per-episode rollups, including
  `reached_done`/`emails_remaining_at_cutoff` (see finding #6). **Lower
  confidence mart** for episode *length* specifically -- see finding #1
  below -- but the completion-status columns are correct regardless.

## Parser robustness: `_parse_action`'s silent XML-only fallback masked two real LLM-connectivity failures

`inference.py`'s `_parse_action` only recognizes strict `<priority>...
</priority>` XML tags and silently defaults to `low`/`spam`/`trash` on any
mismatch, with no logging. Combined with `_call_llm`'s bare `except
Exception: return ""`, this made two different real problems -- an
exhausted API key, and (separately) a model that doesn't reliably produce
XML tags -- produce the exact same symptom: every single action
identical, no error visible anywhere.

`run_episodes.py` now uses `_parse_action_lenient()` instead of
`inference._parse_action` (added there, not in `inference.py`): same
strict-XML regexes tried first (identical behavior for compliant models),
then a `Key: value` / `**Key:** value` line parser, then a last-resort
loose whole-word search, and -- critically -- every fallback path now
prints a specific warning to stderr (including a snippet of the raw
text), instead of failing silently. An empty LLM response specifically
prints a pointer to `diagnose_llm.py` rather than looking like a
legitimate classification.

### Diagnosing a "every action looks identical" run

If you ever see `run_episodes.py` produce the same `priority`/`category`/
`route` on every single step across every episode, don't assume it's a
grading or telemetry problem -- check stderr first. As of the fix above,
a genuine LLM-call failure now prints an explicit warning there rather
than hiding inside a plausible-looking default action. Run
`python diagnose_llm.py` (prints the real `API_BASE_URL`/`MODEL_NAME`/
masked `HF_TOKEN` and the actual exception `_call_llm` would otherwise
swallow) to confirm before changing anything else. One easy-to-miss
gotcha: `load_dotenv()` does not override an already-exported shell
environment variable, so editing `.env` after having exported
`MODEL_NAME`/`HF_TOKEN` directly in the same shell session earlier will
silently have no effect.

### A real regression, caught by the data itself, not by reading the code

At one point, a copy of `run_episodes.py` in active use had silently
reverted to calling `inference._parse_action` directly -- the exact strict
XML-only parser this section describes replacing -- despite this file
already documenting `_parse_action_lenient()` as the fix. The regression
wasn't visible from the run's console output alone (no exceptions, no
crash); it was caught by querying `client_steps` telemetry directly:
`error` was `NULL` on 100% of 240 rows (should be informative on the ones
that fell back), and the literal fallback triple `low`/`spam`/`trash`
appeared on 100% of rows for the harder tasks (full-triage,
critical-escalation, action-orchestrator, threat-assessment) -- statistically
impossible if those were genuine classifications given the same email
distribution showed only ~15% real spam in the easier tasks. Restored and
re-verified against realistic malformed outputs (markdown-bold, plain
`Key: value`, loose prose, and a real JSON `action_plan` block) before
trusting it again. Lesson worth keeping: documentation describing a fix
is not evidence the fix shipped -- query the actual telemetry the code
produced.

## Rate limits, dropped connections, and the two-model split

Two separate failure modes showed up once real multi-episode runs got
long enough, and `run_episodes.py` now handles both -- but neither is
free, and both are worth understanding rather than just trusting the
retry logic blindly.

**Empty LLM responses mid-run (`_call_llm_with_retry`).** `_call_llm`
swallows every exception and returns `""`, so an empty response could be
a genuine rate limit, a dropped connection to the provider, or (as above)
a reasoning model burning its budget. `run_episodes.py` now retries an
empty response with a 5s -> 15s -> 30s backoff before accepting it as
real, plus a small proactive pacing delay between every call
(`--request-delay`, default 1s) to avoid tripping the ceiling in the first
place. The default is tuned for the four short-output tasks; it is **not**
enough for `action-orchestrator`/`threat-assessment` on a tightly
TPM-capped model -- see the rate-limit table above.

**Dropped WebSocket connections (`main()`'s retry loop).** A long run can
lose its connection mid-episode
(`websockets.exceptions.ConnectionClosedError` / `OSError` /
`asyncio.TimeoutError`) for reasons unrelated to the LLM provider entirely
-- confirmed once to be the local connection between the client and the
project's own `uvicorn` server dropping, not a Groq-side failure.
`run_episode()` calls are now wrapped in a retry loop: on a connection
error, wait 10s and reconnect with a fresh WebSocket session (the failed
episode restarts from step 1 -- steps already logged before the drop stay
in the telemetry), up to 3 attempts, then log the episode as skipped and
**continue with the rest of the run** rather than crashing the whole
script.

**One concrete, plausible cause of that dropped connection, worth ruling
out before assuming it's random flakiness:** if you're running the server
with `uvicorn --reload`, WatchFiles watches the *entire* project
directory by default, including `data/raw/`, which the server's own
telemetry sink writes a new line into on every single step. It's possible
for the reloader to see its own telemetry write as a "code change" and
restart the server process mid-request. **Don't use `--reload` for an
actual data-collection run** -- it's a development convenience, not
something you want fighting a long eval run:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000   # no --reload
```

**Rate limits and the two-model split.** This project's own collected
dataset uses `llama-3.3-70b-versatile` for spam-detection,
priority-classification, full-triage, and critical-escalation, but
`llama-3.1-8b-instant` for action-orchestrator and threat-assessment --
specifically because the latter two tasks' JSON output pushed
consistently against `llama-3.3-70b-versatile`'s 12K TPM free-tier
ceiling (see the rate-limit table above), while `llama-3.1-8b-instant`'s
much larger free-tier budget handled them cleanly. This is a **deliberate,
evidence-based choice**, not an inconsistency: `model_name` is preserved
per-row through `client_steps` -> `int_steps_joined` ->
`agg_model_leaderboard`, so the leaderboard never silently blends the two
models' results together. If you want a clean single-model comparison
across all six tasks instead, re-run those two tasks on
`llama-3.3-70b-versatile` with a much longer pacing delay:
```bash
python run_episodes.py --episodes 4 --tasks action-orchestrator,threat-assessment --request-delay 12
```

## Known findings (confirmed by running the actual, unmodified-arithmetic environment)

These are properties of the current codebase's runtime behavior, verified
by actually executing `EmailTriageEnvironment` end-to-end and inspecting
the resulting warehouse tables -- not guesses. Findings #0 and #2 have
since been fixed (see below); findings #1 and #3 are flagged for a
decision and have **not** been changed.

### 0. `inference.py`'s HTTP client can't produce trustworthy env-side telemetry at all -- root cause of #1, fixed via a new script

This supersedes finding #1 below with something more fundamental, found by
actually running `inference.py`'s real `run_task()` against a live
`uvicorn` server (not just calling the environment as a plain Python
object, which is all the other verification in this file did before this).

**Mechanism, confirmed directly in `openenv-core`'s installed source**
(`openenv/core/env_server/http_server.py`): the plain HTTP `/reset` and
`/step` routes each do `_env = self._env_factory()` -- a **brand-new,
throwaway `EmailTriageEnvironment` instance, every single call**, with no
session token linking a `/step` call back to the instance a preceding
`/reset` used. `EmailTriageEnvironment.step()` also has a "stateless HTTP
guard": if called on an instance whose queue is empty (i.e. never reset),
it silently calls `self.reset()` internally first, sampling a **brand-new
random episode**.

Net effect: `inference.py`'s hand-rolled HTTP client shows the agent email
A via `/reset`, then the very next `/step` call hits a different,
auto-resetting instance and grades the action against unrelated email B.
Confirmed live: pulling matching-position `client_steps`/`env_steps`
events from the same real `run_task()` execution showed identical
predicted fields (the action really was relayed correctly) but completely
different `email_id`s.

The silver lining: `inference.py`'s own reported `task_reward` is computed
client-side from `reset_obs_data` (the ground truth captured from its own
`/reset` call), not from the broken `/step` response -- so its
console-reported scores are trustworthy. What's **not** trustworthy is
anything from the `env_steps` side of telemetry when `inference.py` is the
caller: `true_priority`/`category`/`route`, `priority_ok`/`category_ok`/
`route_ok`, `shaped_reward`, all of `reward_components` are computed
against an unrelated random email, and `email_id` won't match between
`client_steps` and `env_steps`, so `int_steps_joined.sql`'s LEFT JOIN
produces essentially no real matches.

**Root cause:** the project ships two clients. `client.py`'s
`EmailTriageEnv(EnvClient)` is the correct one -- it connects over
WebSocket to `/ws`, which really does keep one persistent, session-bound
environment instance alive (confirmed in the same `openenv-core` source).
`inference.py` doesn't use it; it reimplements its own simplified client
against the stateless REST routes instead.

**inference.py was provided by the hackathon for automated grading and is
left completely untouched** -- it's a record of what was actually
submitted, and its own reported scores are unaffected by any of this. The
fix is a **new script, `run_episodes.py`**, that reuses `inference.py`'s
`TASKS`/graders/system-prompts/LLM-call/action-parser verbatim (imported,
not copied) and drives them through `client.py`'s WebSocket client
instead, with real per-episode continuity (`reset()` once, then `step()`
through all 10 emails). See "Running `run_episodes.py`" below.

**A second, related bug found and fixed along the way:** `client.py`'s own
`_parse_result()` only mapped 9 of `EmailTriageObservation`'s 15 fields --
it silently dropped `true_priority`, `true_category`, `true_route`,
`is_business_critical`, `is_phishing`, and `linked_incident`, so any
observation built from a WebSocket response defaulted those to
`None`/`False` regardless of what the server actually sent. This is fixed
(see the diff in `client.py`) since it's the project's own code, not the
frozen hackathon file.

**Confirmed after both fixes:** running `run_episodes.py`'s real code path
against a live server, `done` only turns `True` on step 10 (not step 1),
reward values reflect genuine streak-bonus accumulation across the
episode, and `client_steps`/`env_steps` `email_id`s match exactly, in
order, for every step of the same episode.

### 1. `inference.py` resets before every single graded email

(Still relevant specifically to `inference.py`'s own behavior, even though
finding #0 above is the deeper reason its telemetry can't be trusted.)

`run_task()` calls `env.reset()` before *every* graded email, not once per
10-email episode -- so `current_streak`, `_cluster_routes`, and
`_current_idx` never persist across emails in the harness as actually run.
Practical effect, confirmed by querying `fct_reward_components` after a
mixed run:

| pattern | `n_steps` in `agg_episode_summary` | `streak_bonus` / `coherence_bonus` / `dependency_bonus` |
|---|---|---|
| `inference.py` as-is (reset-per-email) | 1 | always 0 |
| full multi-email episode (reset-per-episode) | up to 14 | nonzero on ~15-20% of graded emails, as designed |

If you run `inference.py` unmodified against this warehouse, expect
`agg_episode_summary` to show `n_steps = 1` for nearly every row, and
`streak_bonus`, `coherence_bonus`, and `dependency_bonus` to read ~0 across
the board in `fct_reward_components`. That's the harness, not the
telemetry.

**Refinement, confirmed with a real (non-oracle) model via
`run_episodes.py`:** `coherence_bonus` only computes when `done and not
stateless_http_mode` (see `step()`), and `done` requires the episode to
reach its natural end. A real, imperfect model makes at least one
urgent/high -> low/medium mistake somewhere in a 10-email episode often
enough that the escalation-injection mechanic (see the README's
"Escalation Consequences") keeps extending the queue past what
`MAX_STEPS = 10` can exhaust. In a real 9-episode sample across
full-triage/action-orchestrator/threat-assessment, **0 of 9 episodes
reached `done=True`** -- every one was cut off by the step cap instead of
completing naturally. Don't be surprised if `coherence_bonus` reads ~0
across most real runs, even with `run_episodes.py`'s correct continuity --
that's a property of the escalation mechanic interacting with an
imperfect policy, not a telemetry problem. `streak_bonus` and
`dependency_bonus` don't depend on `done` and do fire mid-episode as
expected.

**Also worth knowing about `threat-assessment`'s own grading (not
something to fix -- it's the hackathon's grader, same status as
`inference.py`):** `_grade_threat_assessment` gives non-phishing emails
(the ~90% majority) a flat `0.30` (correct "not a threat" classification)
+ `0.10` (report has >=3 keys) = `0.40` ceiling, with no further credit for
report quality -- the attack-vector/indicators/recommended-actions/risk-
score checks are only reached for actual phishing emails. Expect
`task_reward` for this task to cluster tightly around `0.40` for most
steps in any episode, with real variation only showing up on the ~1-in-10
emails that are genuinely phishing.

### 2. Phishing emails never actually appeared in any episode -- FIXED

`EmailTriageEnvironment._sample_episode()` used to append the phishing
email to the candidate list immediately before a `.pop()` call whose
comment said it was removing "the last standard email" to make room for
the two-email dependency cluster. The actual last item at that point was
the just-appended phishing email, not a standard one -- so every episode
silently ended up with 7 standard + 1 critical + 2 dependency-cluster
emails, and the phishing email was discarded every time.

**Confirmed before the fix:** 0 phishing emails across 200 sampled
episodes (2,000 emails), and `is_phishing = false` for every row across a
live 111-row `env_steps` sample.

**Fix applied:** the pop now targets index `len(CATEGORIES) - 1` --
the last of the 7 standard-category emails added in step (1) of
`_sample_episode()`, which is what the pre-existing comment always said
this was supposed to do -- instead of the overall last item in the list.

**Confirmed after the fix:** 200/200 sampled episodes now contain exactly
1 phishing email. Rebuilding the warehouse against fresh telemetry (171
graded emails) shows 14 phishing emails, `phishing_bonus` /
`phishing_miss_penalty` nonzero on all 14 of them, and
`agg_model_leaderboard.phishing_catch_rate` populated with real values
instead of `NULL` across every task.

### 3. Ground-truth leak in `_make_observation()`

The file's own header docstring claims `observation.metadata` no longer
contains `true_priority`/`category`/`route` before the agent acts. In
practice, the top-level `true_priority`, `true_category`, `true_route`,
`is_business_critical`, and `is_phishing` fields are set from the email
being shown *next* -- including on `reset()`, before any action is taken --
not from the previously-graded email. As long as the LLM prompt-builder
only reads `email_subject`/`email_sender`/`email_body` (which it currently
does), the policy itself never sees it, but any code with direct
observation access does. Not triggered by the warehouse pipeline itself
(the telemetry hook logs these fields deliberately, for grading, from
inside the trusted server process), but worth knowing if you build
anything else against the raw observation object.

### 4. Run the server locally, not on a hosted Space, for this pipeline

Considered and rejected: deploying the FastAPI server to a free-tier
Hugging Face Space instead of running it locally. Free Spaces are limited
to 16GB RAM, 2 CPU cores and 50GB of **non-persistent** disk by default --
persistent storage is a paid add-on. Since this whole pipeline assumes
`event_sink.py`'s JSONL lands on a filesystem `compact_to_parquet.py` and
`dbt` can both read locally, a hosted Space would mean either paying for
persistent storage (against this project's free-tooling constraint) or
re-architecting the sink to push to a Hugging Face Dataset instead of
local disk -- real additional work, not a config change. Local also avoids
Space cold-starts corrupting `llm_latency_ms` telemetry. Worth revisiting
once the pipeline itself is a finished, stable artifact and the goal
shifts to "make the environment a public demo" rather than "build and
debug the data pipeline."

**A real, related mistake this caused:** for a while, the client was
pointed at a remote Space URL (`EMAIL_RL_SERVER_URL` was set once,
somewhere, and outlived its usefulness -- `load_dotenv()` not overriding
an already-exported shell variable, again) while the server the developer
was actually watching and editing ran locally and never received any
traffic. The symptom was `env_steps` telemetry staying empty indefinitely
even after the CWD/path fix below was applied and looked correct in
isolation -- the server code was never wrong, it just never got called.
Confirmed by adding a startup print of the resolved `SERVER_URL` /
telemetry root to both processes and comparing them side by side.

### 5. Column aliasing in `int_steps_joined` -- easy to guess wrong when querying by hand

Both `env_steps` and `client_steps` log their own `step` and `done`
fields, so `int_steps_joined.sql` disambiguates them: the raw JSONL/mart
column names are `env_step` / `client_step` and `env_done` / `client_done`,
not bare `step` / `done`. Run `DESCRIBE main.int_steps_joined` before
writing an ad-hoc query against this mart rather than assuming a name --
this tripped up more than one debugging session here.

### 6. `agg_episode_summary` couldn't distinguish a clean finish from a step-cap cutoff -- fixed

`n_steps` / `final_step` alone can't tell "episode reached its natural
end" apart from "the client's fixed `MAX_STEPS` cut it off while the
environment's own queue had grown past that." Both can read `n_steps=10`.
This matters because of the escalation-injection mechanic (see finding #1
above): every missed urgent/high email inserts a follow-up 2 positions
ahead, so the environment's real queue can grow well past 10 mid-episode.

Confirmed on a real single-episode run: `emails_remaining` traced across
steps 1-10 showed the queue growing from 10 to 13 emails (three separate
overload penalties firing, at steps 3, 4, and 8), with `done=False` on
every single logged step -- the client stopped at its own step cap with 2
of 13 emails never graded, not because the episode finished.

**Fix:** `agg_episode_summary` now includes `reached_done` and
`emails_remaining_at_cutoff`, sourced from the *last* logged step of each
episode (via `qualify row_number() over (partition by episode_id order by
env_step desc) = 1`), so a step-cap cutoff and a genuine finish are no
longer silently averaged together on the leaderboard.

### 7. Parquet schema drift across days with an all-null column -- fixed

`compact_to_parquet.py` used to let `pyarrow` infer each day's schema
from that day's batch alone. A day where `action_plan`/`threat_report`
happened to be `None` in every row (any run of the four non-JSON tasks)
got those columns typed as something other than string -- confirmed as
`INTEGER` via `DESCRIBE main.int_steps_joined` -- while a day that
actually populated them typed `VARCHAR`. `read_parquet(glob,
hive_partitioning=true)` then has to reconcile `INTEGER` vs `VARCHAR` for
the same column across files.

**Fix:** both streams now have an explicit, pinned `pyarrow.schema(...)`
in `compact_to_parquet.py`, and every row is normalized to exactly that
schema's field list before being handed to `pa.Table.from_pylist(...,
schema=...)` -- missing fields (an older JSONL file predating a newer
telemetry field) default to `None` rather than erroring, and unexpected
fields (a newer telemetry field this schema hasn't been updated for yet)
print a warning rather than silently vanishing or crashing the batch job.
`cluster_id` got the same fix pre-emptively -- it's `None` on the ~80% of
emails outside a dependency cluster, the identical failure mode waiting
to happen on a day with no clustered emails.

## Example questions this warehouse can answer

```sql
-- Leaderboard across tasks
select * from agg_model_leaderboard order by task, avg_task_reward desc;

-- Where does the model actually get priority wrong?
select * from fct_confusion_matrix where field = 'priority' and true_value <> predicted_value order by n desc;

-- How much of the shaped reward is coming from partial credit vs shaping bonuses?
select
    avg(base_score * urgency_multiplier) as avg_base_component,
    avg(streak_bonus) as avg_streak,
    avg(overload_penalty) as avg_overload,
    avg(coherence_bonus) as avg_coherence
from fct_reward_components;

-- Confirm finding #1 yourself: episode length distribution
select n_steps, count(*) as n_episodes from agg_episode_summary group by 1 order by 1;

-- Confirm finding #6 yourself: how often does an episode actually finish
-- vs. get cut off mid-escalation-spiral by the client's step cap?
select
    reached_done,
    count(*) as n_episodes,
    avg(emails_remaining_at_cutoff) as avg_emails_left_when_cut_off
from agg_episode_summary
group by 1;

-- Confirm the phishing fix yourself: roughly 1 in 10 graded emails should
-- now be phishing (was exactly 0% before the fix, regardless of harness mode)
select
    round(avg(case when is_phishing then 1.0 else 0.0 end), 3) as phishing_rate,
    count(*) as n_graded_emails
from stg_env_steps;
```

## Limitations

- **Single-process assumption.** `event_sink.py`'s `threading.Lock` is
  per-process. Under `uvicorn --workers > 1`, each worker gets its own lock
  and its own file handle; writes across workers are not coordinated. Fine
  for local/single-worker use; if you scale workers up, either route
  through a single writer process or switch the sink to something with
  cross-process locking.
- **Fail-open by design.** A telemetry write failure (disk full, bad
  permissions, an unexpected argument) is caught, logged to stderr, and
  swallowed -- it will never break a graded RL step or an eval run, but it
  also means telemetry gaps are silent unless you're watching stderr.
- **`agg_episode_summary`'s `n_steps`/`final_step` is lower-confidence for
  episode *length* specifically**, given finding #1 above -- it's correct
  for whatever episode structure actually occurred, but that structure is
  currently "1 email" for anything produced by unmodified `inference.py`.
  Its `reached_done`/`emails_remaining_at_cutoff` columns (finding #6) are
  not affected by this and are trustworthy regardless.
- **Two different models across tasks in this project's own dataset.**
  action-orchestrator/threat-assessment were collected on
  `llama-3.1-8b-instant`, not `llama-3.3-70b-versatile` like the other
  four tasks, due to free-tier rate limits -- see "Rate limits and the
  two-model split" above. `model_name` is preserved per-row so nothing is
  silently blended, but don't read `agg_model_leaderboard` as one model's
  performance across all six tasks without checking `model_name` first.

  