# Email Triage RL Environment + Observability Warehouse

An OpenEnv-based reinforcement learning environment where LLM agents triage
synthetic emails -- spam detection, priority classification, full triage,
critical escalation, action orchestration, and threat assessment -- built
on a hackathon submission and extended into a full telemetry + data
warehouse pipeline: **event log -> DuckDB/dbt warehouse -> queryable
leaderboard**, using only free/open-source tooling.

The interesting part of this project isn't just the RL environment itself
-- it's the process of instrumenting it, finding that several of its own
metrics were structurally unreachable or silently wrong, and fixing that
with evidence rather than assumption. See `WAREHOUSE.md` for the full,
warts-and-all writeup; this file is the short version.

## Architecture

```
LLM agent (via run_episodes.py, WebSocket)
        │
        ▼
EmailTriageEnvironment.step()  ──►  telemetry/event_sink.py  ──►  data/raw/*/dt=YYYY-MM-DD/events.jsonl
        │                                  (JSONL, fail-open)
        ▼
graded reward + observation                       │
                                     telemetry/compact_to_parquet.py
                                                    ▼
                                   data/lake/*/dt=.../*.parquet
                                                    │
                                          warehouse/ (dbt-duckdb)
                                          staging → intermediate → marts
                                                    ▼
                                     warehouse/warehouse.duckdb (query this)
```

- **`server/Email_RL_environment.py`** -- the FastAPI/OpenEnv environment:
  synthetic email generation, phishing injection, cross-email dependency
  clusters, escalation consequences, and the shaped reward function.
- **`inference.py`** -- **frozen**, byte-identical to the original
  hackathon submission. Its own reported scores are trustworthy and
  untouched; it is never modified, only imported.
- **`run_episodes.py`** -- the real evaluation harness. Drives the
  environment over `client.py`'s WebSocket client with genuine
  multi-email episode continuity, a model-agnostic action parser, and
  retry/backoff resilience against rate limits and dropped connections.
- **`telemetry/`** -- zero-dependency, fail-open JSONL event sink plus a
  batch Parquet compactor.
- **`warehouse/`** -- a dbt-duckdb project: staging -> intermediate ->
  marts, queryable directly with DuckDB.

## Quick start

```bash
pip install -r server/requirements.txt
pip install -r requirements-warehouse.txt

# .env
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
HF_TOKEN=<your Groq API key>
EMAIL_RL_SERVER_URL=http://localhost:8000

# terminal 1
uvicorn server.app:app --host 0.0.0.0 --port 8000   # no --reload for real runs

# terminal 2
python diagnose_llm.py                              # confirm SUCCESS + non-empty response first
python run_episodes.py --episodes 4
python telemetry/compact_to_parquet.py
cd warehouse && dbt run --profiles-dir .
```

Then query it:
```bash
python -c "import duckdb; print(duckdb.connect('warehouse/warehouse.duckdb').sql('select * from agg_model_leaderboard').df())"
```

Full setup detail, including the LLM-provider gotchas below, is in
`WAREHOUSE.md`.

## What this project actually demonstrates

Rather than a clean success story, this is a record of finding and fixing
real problems empirically -- which is the more honest (and more
interesting) version of a data engineering project:

- **The evaluation harness itself was broken.** `inference.py`'s
  hand-rolled HTTP client hit stateless routes that spun up a brand-new
  environment instance on every call -- an agent was shown one email via
  `/reset` and graded against a completely different one on `/step`.
  Confirmed by directly comparing mismatched `email_id`s across the same
  execution, and fixed with a new harness (`run_episodes.py`) that drives
  the environment's real WebSocket client instead of duplicating or
  patching the frozen hackathon file.
- **A phishing-sampling bug meant 0 of 2,000 sampled emails, across 200
  episodes, were ever actually phishing** -- a `.pop()` call was removing
  the wrong list index. Confirmed before and after the one-line fix.
- **A silent parser fallback made two unrelated failures produce the
  identical symptom.** An exhausted API key and a model that just doesn't
  reliably emit strict XML both defaulted to the same hardcoded action
  with zero logging. Replaced with a layered, model-agnostic parser that
  reports exactly which fallback path fired, plus a diagnostic script
  (`diagnose_llm.py`) that surfaces the real exception `_call_llm` was
  swallowing.
- **A reasoning model (`gpt-oss-120b`) returned genuine HTTP 200s with
  empty content**, spending its entire token budget on invisible
  reasoning tokens -- indistinguishable from a rate limit until traced
  down to the actual API response. Raising `max_tokens` made it *worse*,
  not better, which was itself a useful (if initially wrong) finding.
- **Free-tier rate limits turned out to differ meaningfully by model and
  by task**, not just by provider -- confirmed against Groq's published
  limits and handled with real retry/backoff and pacing logic rather than
  a bigger hammer.
- **A reward-shaping mart was quietly built on structurally unreachable
  data.** The original harness reset before every single email, so
  streak/dependency/coherence bonuses -- fully implemented, correctly
  coded -- could never fire. Fixing the harness surfaced a second,
  subtler finding: even with real continuity, an imperfect model rarely
  reaches natural episode completion at all, because missed
  urgent/high-priority emails inject escalation follow-ups that can push
  the queue past the step cap. The warehouse now tracks this explicitly
  (`reached_done`, `emails_remaining_at_cutoff`) instead of conflating a
  clean finish with a cutoff.
- **A Parquet schema silently drifted across days** when a column
  happened to be all-null in one day's batch. Fixed with an explicit,
  pinned schema instead of per-batch type inference.

Every finding above was confirmed by running the actual code and
inspecting the resulting data -- not by reading the source and reasoning
about what it probably did. See `WAREHOUSE.md` for the evidence behind
each one, including specific row counts and query results.

## Honest limitations

- **`train.py` (the GRPO training script) is not wired into the active
  project run.** It's not invoked by Docker, the OpenEnv manifest, or any
  script here, and it imports several packages not in `requirements.txt`.
  It exists as a reference implementation, not a script that's actually
  been run end-to-end. Deprioritized in favor of the data-engineering
  work above; revisit before claiming this project trained anything.
- **A ground-truth leak exists in `_make_observation()`**: the file's own
  header comment claims this was already fixed, but the top-level
  `true_*` fields are populated for the *next* email before the agent
  acts on it. Not exploitable through the current LLM prompt (which only
  reads subject/sender/body), but flagged and not yet fixed.
- **`threat-assessment`'s grader has a flat 0.40 ceiling** for the ~90% of
  emails that aren't phishing -- a property of the frozen hackathon
  grader, not a bug, but worth knowing before reading too much into that
  task's average reward.
- **This project's own dataset mixes two models** across the six tasks,
  for a documented, rate-limit-driven reason -- see `WAREHOUSE.md`'s
  "Rate limits and the two-model split."

## Credits

`inference.py` and the original environment design were provided as part
of a hackathon submission and are preserved byte-identical as a record of
what was actually submitted. Everything under `telemetry/`, `warehouse/`,
`run_episodes.py`, and `diagnose_llm.py` was built afterward as an
independent extension.

