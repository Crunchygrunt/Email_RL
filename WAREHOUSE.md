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
multi-episode run doesn't take forever), a strong free-tier model tuned
for structured/agentic output, and free-tier usage isn't flagged for
model training the way some alternatives are.

```
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=openai/gpt-oss-120b
HF_TOKEN=<your Groq API key>
```

(`HF_TOKEN` as a name is a holdover from the original hackathon setup --
it's not validated against any particular key format, any provider's API
key works there.) `llama-3.3-70b-versatile` on the same endpoint is a
solid fallback if `gpt-oss-120b` ever gets rate-limited or deprecated.

**Whatever provider you use, run `python diagnose_llm.py` before
`run_episodes.py`** and confirm it prints `SUCCESS`. This has caught two
separate real failures already (an exhausted API key, and a `.env`
override that silently didn't apply) that otherwise looked identical to
"the model just isn't very compliant" -- see "Parser robustness" below.

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
- `marts/agg_episode_summary.sql` -- per-episode rollups. **Lower
  confidence mart** -- see findings below.

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
- **`agg_episode_summary` is lower-confidence** given finding #1 above --
  it's correct for whatever episode structure actually occurred, but that
  structure is currently "1 email" for anything produced by unmodified
  `inference.py`.
