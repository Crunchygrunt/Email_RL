# Orchestration layer — confirmed vs. still open

Same confirmed-vs-pending convention as `WAREHOUSE.md`. This version was
built against your actual uploaded files (`run_episodes.py`, `inference.py`,
`client.py`, `models.py`, the real `telemetry/`, `warehouse/`, and
`server/` folders, the real `warehouse.duckdb`, and the existing
`.github/workflows/data-quality.yml`) — not the progress-doc descriptions
of them. The previous draft's guesses are gone; every claim below was
either read directly from your source or exercised end to end.

## What was actually run this session, not just written

- **Full pipeline, fully green, end to end**, using the real
  `telemetry/event_sink.py`, real `telemetry/compact_to_parquet.py`, a
  real `dbt build` (installed `dbt-duckdb` fresh) against your real
  `warehouse/` project, and a stub in place of only the one thing that
  can't run without live API/network access: the actual LLM calls inside
  `run_episodes.py`. Result: `PASS=22, WARN=0, ERROR=0` — all 8 models, 14
  data-quality tests. README table and badge JSON came out correctly
  formatted from that run.
- **The real, already-collected `warehouse.duckdb` you uploaded** (240
  rows, `llama-3.1-8b-instant` across all 6 tasks) was queried directly
  with the exact SQL `pipeline_flow.py` uses — confirmed the column names
  and the rendered table match your actual mart output, not an assumed
  schema.
- **A real edge case found by testing, not reasoning:** if `dbt build`
  runs against an empty Parquet lake (zero files under `data/lake/`),
  `read_parquet(glob)` raises a hard `IO Error: No files found...` instead
  of an empty result — this is DuckDB's own behavior, not something in
  this pipeline. In practice this only bites on a from-scratch clone with
  no prior collection ever run, or a scheduled run where every single
  model fails before logging one step. Once a run compacts even a
  handful of real rows, `dbt build` proceeds normally — confirmed by
  writing one fake `client_step`/`env_step` pair through the real
  `event_sink` functions and watching the full `dbt build` go green.
  Worth knowing so a first-ever cold run failing here doesn't read as a
  new bug in `pipeline_flow.py`.
- **The literal env var name issue.** `inference.py` reads its API key via
  `HF_TOKEN = os.getenv("Grok API")` — a real, space-containing key name,
  confirmed by grepping the source (matches the odd `Grok API = ...` line
  in your `_env.example`). `pipeline_flow.py` translates a normally-named
  `GROQ_API_KEY` into that exact key before spawning each subprocess, so
  the frozen file never needs touching and the workflow never has to set
  an env var with a space in its YAML.
- **`run_episodes.py` only takes `--episodes` and `--tasks`.** The
  `--request-delay` flag in the earlier draft doesn't exist — pacing is a
  fixed internal constant (`_REQUEST_PACING_DELAY = 1.0`). Dropped from
  both `model_config.yaml` and the flow.
- **Repo folder name matters, and it already lines up.** `run_episodes.py`
  falls back to `from Email_RL.client import EmailTriageEnv` if the flat
  import fails, which only resolves if the checked-out folder is
  literally named `Email_RL`. Per the v3 progress doc
  (`github.com/Crunchygrunt/Email_RL`), that's the actual repo name, and
  `actions/checkout@v4`'s default behavior checks out into a folder named
  after the repo — so this lines up with no extra workflow config needed.

## What's still genuinely open (not guessed, not faked)

None of these were in the files you sent, so none of this is invented —
just flagged:

1. **README.md's real content.** `update_readme()` looks for
   `<!-- BASELINE_SCORES_START -->` / `<!-- BASELINE_SCORES_END -->` and
   raises (doesn't guess) if they're missing. Add them once, anywhere you
   want the table to live.
2. **Whether `requirements-warehouse.txt` / a root `requirements.txt`
   already exist.** The workflow installs `openai`, `websockets`,
   `pyarrow`, `dbt-duckdb` explicitly since I can't confirm those files'
   contents — harmless if redundant, worth consolidating once you check.
3. **`.gitignore`** — matters for whether `data/` and `warehouse.duckdb`
   are already excluded from commits (relevant to the Streamlit plan
   below, which wants `warehouse.duckdb` committed).
4. **Groq model availability.** `llama-3.3-70b-versatile` /
   `llama-3.1-8b-instant` are what your real `warehouse.duckdb` already
   shows data for; verify current names at console.groq.com/docs/models
   before adding a third model to `model_config.yaml`.

## Why Prefect over Dagster, unchanged from the earlier draft

This deployment shape — one-shot, cron-triggered by GitHub Actions, no
always-on infrastructure — doesn't exercise Dagster's actual advantage
(its asset catalog + UI, which wants `dagster dev` or a webserver running
persistently to pay off). Prefect's `@flow`/`@task` decorators run
standalone, confirmed this session with zero Prefect server or daemon:
`python orchestration/pipeline_flow.py` spins up a throwaway local API,
does its job, exits. (One related finding: Prefect pings an anonymous
telemetry endpoint by default; `pipeline_flow.py` sets
`PREFECT_API_TELEMETRY_ENABLED=false` so a CI runner with a locked-down
network doesn't produce a spurious warning.)

