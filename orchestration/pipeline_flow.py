"""
Prefect orchestration pipeline for the Email Triage RL evaluation loop.

Reuses the existing, already-verified project scripts as subprocesses rather
than re-implementing their logic:
  - run_episodes.py                    (episode generation + LLM eval + grading + telemetry)
  - telemetry/compact_to_parquet.py    (JSONL -> Parquet compaction)
  - dbt build (warehouse/)             (staging -> intermediate -> marts, + data quality tests)

Flow: for each configured model -> run its assigned task set through
run_episodes.py -> compact telemetry -> dbt build -> query
agg_model_leaderboard (+ agg_data_quality) -> render the README's Baseline
Scores table -> write a last-evaluated badge JSON.

No Prefect server or daemon is used or needed. GitHub Actions' cron
supplies the schedule, compute, and run history; this file only needs
Prefect's Python API (@flow/@task) for retries, structured logging, and a
readable task graph. `python orchestration/pipeline_flow.py` runs the whole
thing synchronously, locally, in CI, or anywhere else.

Confirmed against the real project files (not guessed):
  - run_episodes.py's only CLI flags are --episodes and --tasks (no
    --request-delay -- pacing is a fixed internal constant). Model
    switching works by setting MODEL_NAME before each subprocess call,
    since inference.py reads it at import time via
    `os.getenv("MODEL_NAME", ...)` and each subprocess re-imports fresh.
  - inference.py's API key is read via `os.getenv("Grok API")` -- a
    literal, space-containing env var name baked into the frozen file
    (see _translate_api_key below for how this is handled without
    touching inference.py).
  - agg_model_leaderboard's real columns: model_name, task,
    n_graded_emails, priority_accuracy, category_accuracy, route_accuracy,
    perfect_match_rate, avg_task_reward, n_phishing_emails_seen,
    phishing_catch_rate.
  - agg_data_quality's real shape: metric_type / metric_key / total_rows /
    n_rows rows, with metric_type='quality_flag_summary' holding the
    overall violation rate as n_rows/total_rows.
  - warehouse/dbt_project.yml's profile is 'email_rl_warehouse', target
    'dev' (from profiles.yml) -- `dbt build --profiles-dir .` run from
    inside warehouse/ is correct.
  - compact_to_parquet.py's --data-root/--lake-root defaults are relative
    to CWD, not anchored to __file__ (unlike event_sink.py) -- so it MUST
    be run with cwd=PROJECT_ROOT, which this flow does.

Still unconfirmed (flagged, not guessed -- see ORCHESTRATION.md):
  - README.md's real content / whether Baseline Scores markers exist yet.
  - Whether requirements-warehouse.txt exists at the repo root.

Run locally:  python orchestration/pipeline_flow.py
Run in CI:    see .github/workflows/eval-pipeline.yml
"""

from __future__ import annotations

import json
import os

# Skip Prefect's anonymous usage telemetry ping (sens-o-matic.prefect.io) --
# this is a one-shot CI script, not worth a network call that can fail
# noisily (harmlessly, but noisily) in a restricted-network runner.
os.environ.setdefault("PREFECT_API_TELEMETRY_ENABLED", "false")

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import yaml
from prefect import flow, get_run_logger, task

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE_DIR = PROJECT_ROOT / "warehouse"
DUCKDB_PATH = WAREHOUSE_DIR / "warehouse.duckdb"
README_PATH = PROJECT_ROOT / "README.md"
BADGE_JSON_PATH = PROJECT_ROOT / "docs" / "assets" / "last_evaluated_badge.json"
MODEL_CONFIG_PATH = Path(__file__).resolve().parent / "model_config.yaml"

README_TABLE_START = "<!-- BASELINE_SCORES_START -->"
README_TABLE_END = "<!-- BASELINE_SCORES_END -->"

# inference.py (frozen) reads its API key via os.getenv("Grok API") -- an
# unusual literal env var name, confirmed directly from the source
# (`HF_TOKEN = os.getenv("Grok API")`). We never edit inference.py; instead
# we translate a normally-named secret into that exact key before spawning
# each subprocess. Locally, if a real "Grok API" value is already present
# (e.g. via a local .env file that inference.py's own load_dotenv() call
# picks up), that's left alone unless GROQ_API_KEY is also set.
_INFERENCE_API_KEY_ENV_NAME = "Grok API"


def _translate_api_key(env: dict) -> dict:
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        env[_INFERENCE_API_KEY_ENV_NAME] = groq_key
    return env


@task
def load_model_config() -> list[dict]:
    with open(MODEL_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["models"]


@task(retries=2, retry_delay_seconds=60, timeout_seconds=3600)
def run_eval_for_model(model_cfg: dict) -> None:
    """Runs run_episodes.py for one model's assigned task set.

    MODEL_NAME is injected via env var so run_episodes.py / inference.py
    pick it up through the existing provider-agnostic .env-driven config
    -- zero code changes to either frozen file. Each subprocess re-imports
    inference.py fresh, so the env var set here is what that subprocess
    sees regardless of what any local .env file says (an explicitly-set
    env var always wins over load_dotenv(), which never overrides an
    already-set variable).
    """
    logger = get_run_logger()
    env = os.environ.copy()
    env["MODEL_NAME"] = model_cfg["name"]
    env = _translate_api_key(env)

    cmd = [
        sys.executable,
        "run_episodes.py",
        "--episodes", str(model_cfg.get("episodes", 4)),
        "--tasks", ",".join(model_cfg["tasks"]),
    ]
    logger.info("Evaluating %s on tasks: %s", model_cfg["name"], model_cfg["tasks"])
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"run_episodes.py failed for model {model_cfg['name']!r}")


@task
def compact_telemetry() -> None:
    """
    Must run with cwd=PROJECT_ROOT: compact_to_parquet.py's --data-root /
    --lake-root defaults ("data/raw" / "data/lake") are relative to CWD,
    not anchored to the file's own location the way event_sink.py's are.
    """
    logger = get_run_logger()
    result = subprocess.run(
        [sys.executable, "telemetry/compact_to_parquet.py"],
        cwd=PROJECT_ROOT, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("telemetry/compact_to_parquet.py failed")
    logger.info("Telemetry compacted.")


@task
def dbt_build() -> None:
    result = subprocess.run(
        ["dbt", "build", "--profiles-dir", "."],
        cwd=WAREHOUSE_DIR, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("dbt build failed")


@task
def query_leaderboard() -> list[dict]:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        rows = con.execute(
            """
            select
                model_name, task, n_graded_emails,
                priority_accuracy, category_accuracy, route_accuracy,
                perfect_match_rate, avg_task_reward,
                n_phishing_emails_seen, phishing_catch_rate
            from main.agg_model_leaderboard
            order by task, model_name
            """
        ).fetchall()
        cols = [d[0] for d in con.description]
    finally:
        con.close()
    return [dict(zip(cols, row)) for row in rows]


@task
def query_data_quality_violation_rate() -> float | None:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        row = con.execute(
            """
            select n_rows, total_rows
            from main.agg_data_quality
            where metric_type = 'quality_flag_summary'
            limit 1
            """
        ).fetchone()
    except duckdb.CatalogException:
        return None
    finally:
        con.close()
    if not row or not row[1]:
        return None
    n_rows, total_rows = row
    return n_rows / total_rows


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


@task
def render_markdown_table(rows: list[dict], violation_rate: float | None) -> str:
    if not rows:
        return "_No evaluation data yet._"

    header = (
        "| Task | Model | N | Avg reward | Perfect-match | "
        "Priority acc. | Category acc. | Route acc. | Phishing catch |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['task']} | {r['model_name']} | {r['n_graded_emails']} | "
            f"{_pct(r['avg_task_reward'])} | {_pct(r['perfect_match_rate'])} | "
            f"{_pct(r['priority_accuracy'])} | {_pct(r['category_accuracy'])} | "
            f"{_pct(r['route_accuracy'])} | {_pct(r['phishing_catch_rate'])} |"
        )

    footer = (
        f"\n_Data quality violation rate across all logged steps: {_pct(violation_rate)}._"
        if violation_rate is not None else ""
    )
    return "\n".join(lines) + footer


@task
def update_readme(table_md: str) -> None:
    text = README_PATH.read_text(encoding="utf-8")
    if README_TABLE_START not in text or README_TABLE_END not in text:
        raise RuntimeError(
            f"README.md is missing the {README_TABLE_START} / {README_TABLE_END} "
            "markers -- add them once around the Baseline Scores section, then re-run."
        )
    before, rest = text.split(README_TABLE_START, 1)
    _, after = rest.split(README_TABLE_END, 1)
    new_text = f"{before}{README_TABLE_START}\n{table_md}\n{README_TABLE_END}{after}"
    README_PATH.write_text(new_text, encoding="utf-8")


@task
def write_last_evaluated_badge() -> None:
    """Writes a shields.io 'endpoint' badge JSON.

    README should reference it as:
    ![last evaluated](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/<owner>/<repo>/master/docs/assets/last_evaluated_badge.json)
    (branch is `master`, confirmed from the existing data-quality.yml workflow)
    """
    BADGE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": 1,
        "label": "last evaluated",
        "message": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "color": "blue",
    }
    BADGE_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@flow(name="email-rl-eval-pipeline")
def eval_pipeline() -> None:
    logger = get_run_logger()
    models = load_model_config()

    # Sequential on purpose: Groq's free-tier TPM/RPD limits are per-account,
    # shared across every model you evaluate -- running these concurrently
    # would just make all of them hit 429s sooner, not finish faster.
    for model_cfg in models:
        run_eval_for_model(model_cfg)

    compact_telemetry()
    dbt_build()

    rows = query_leaderboard()
    violation_rate = query_data_quality_violation_rate()
    table_md = render_markdown_table(rows, violation_rate)
    update_readme(table_md)
    write_last_evaluated_badge()

    logger.info("Pipeline complete: %d leaderboard rows written to README.", len(rows))


if __name__ == "__main__":
    eval_pipeline()
    
    