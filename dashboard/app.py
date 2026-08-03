"""
dashboard/app.py -- read-only Streamlit dashboard over the Email Triage RL
warehouse.

Design:
  - Reads warehouse/warehouse.duckdb directly. Zero new ETL -- every tab
    below is a query against a mart or view that already exists (see
    warehouse/models/marts/*.sql). This file adds no new dbt models.
  - Materialized TABLE marts (agg_model_leaderboard, agg_data_quality,
    fct_confusion_matrix, fct_reward_components, agg_episode_summary) are
    self-contained in the .duckdb file and always queryable.
  - The staging/intermediate VIEWS (stg_env_steps, stg_client_steps,
    int_steps_joined) are NOT self-contained: their SQL embeds relative
    Parquet globs (e.g. "../data/lake/env_steps/dt=*/*.parquet",
    resolved against dbt_project.yml's vars), so querying them requires
    the process's CWD to be warehouse/, same as running `dbt build` does.
    This file chdir's into warehouse/ once at startup specifically so
    those views resolve correctly regardless of how you launch
    `streamlit run` -- see get_connection() below. The one place this
    file queries a view directly (the episode step-by-step trace) is
    wrapped in a try/except and degrades to a friendly message if the
    Parquet lake isn't reachable, rather than crashing the whole page --
    same fail-open spirit as telemetry/event_sink.py.

Run:
    streamlit run dashboard/app.py

Deploying for free (Streamlit Community Cloud):
    Point it at this repo/branch, entrypoint dashboard/app.py. Since
    warehouse.duckdb is a single small file, commit it (check
    .gitignore first) -- then every scheduled orchestration/pipeline_flow.py
    run commits a fresh warehouse.duckdb, and Streamlit Cloud redeploys
    automatically. The dashboard stays current with zero extra wiring.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE_DIR = PROJECT_ROOT / "warehouse"
DUCKDB_PATH = WAREHOUSE_DIR / "warehouse.duckdb"
BADGE_JSON_PATH = PROJECT_ROOT / "docs" / "assets" / "last_evaluated_badge.json"

REWARD_COMPONENT_COLUMNS = [
    "base_score", "urgency_multiplier", "streak_bonus", "overload_penalty",
    "response_time_penalty", "escalation_multiplier_delta",
    "phishing_bonus", "phishing_miss_penalty", "dependency_bonus", "coherence_bonus",
]

# Components documented (WAREHOUSE.md) as structurally near-unreachable
# under the reset-per-email pattern -- surfaced explicitly in the UI
# rather than left as a silent flat zero someone has to notice themselves.
_STRUCTURALLY_LIMITED = {"streak_bonus", "dependency_bonus", "coherence_bonus"}


st.set_page_config(page_title="Email Triage RL — Dashboard", layout="wide")


@st.cache_resource
def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


@contextlib.contextmanager
def _resolved_from_warehouse_dir():
    """
    Temporarily chdir into warehouse/ for the duration of one query.

    Only needed for queries against stg_env_steps / stg_client_steps /
    int_steps_joined: those are dbt VIEWS whose SQL embeds relative
    Parquet globs (e.g. "../data/lake/env_steps/dt=*/*.parquet"), resolved
    the same way `dbt build --profiles-dir .` resolves them -- against
    the process CWD at query time, not at view-creation time. Every other
    mart this dashboard reads is a materialized TABLE, self-contained in
    the .duckdb file, and doesn't need this.

    Deliberately scoped to a single `with` block rather than chdir'd once
    at startup: a permanent chdir mutates global process state that
    Streamlit's own file-watcher/rerun machinery (and anything else
    relying on a relative path) doesn't expect -- confirmed by this
    breaking `streamlit run`'s own rerun-on-change if done at import time.
    """
    prev = os.getcwd()
    os.chdir(WAREHOUSE_DIR)
    try:
        yield
    finally:
        os.chdir(prev)


@st.cache_data(ttl=300)
def load_leaderboard() -> pd.DataFrame:
    con = get_connection()
    return con.execute(
        "select * from main.agg_model_leaderboard order by task, model_name"
    ).df()


@st.cache_data(ttl=300)
def load_confusion_matrix() -> pd.DataFrame:
    con = get_connection()
    return con.execute("select * from main.fct_confusion_matrix").df()


@st.cache_data(ttl=300)
def load_reward_components() -> pd.DataFrame:
    con = get_connection()
    return con.execute("select * from main.fct_reward_components").df()


@st.cache_data(ttl=300)
def load_data_quality() -> pd.DataFrame:
    con = get_connection()
    return con.execute("select * from main.agg_data_quality").df()


@st.cache_data(ttl=300)
def load_episode_summary() -> pd.DataFrame:
    con = get_connection()
    return con.execute(
        "select * from main.agg_episode_summary order by episode_dt desc"
    ).df()


def load_episode_trace(episode_id: str) -> pd.DataFrame | None:
    """Best-effort: needs the Parquet lake reachable from warehouse/.

    Returns None (not an exception) if int_steps_joined can't be read, so
    the caller can show a friendly fallback instead of a stack trace.
    """
    con = get_connection()
    try:
        with _resolved_from_warehouse_dir():
            return con.execute(
                """
                select
                    env_step, email_id, model_name, task,
                    true_priority, client_predicted_priority,
                    true_category, client_predicted_category,
                    true_route, client_predicted_route,
                    shaped_reward, task_reward, cluster_id, is_escalation,
                    emails_remaining, env_done
                from main.int_steps_joined
                where episode_id = ?
                order by env_step
                """,
                [episode_id],
            ).df()
    except duckdb.Error:
        return None


def render_last_evaluated_banner() -> None:
    if not BADGE_JSON_PATH.exists():
        return
    try:
        payload = json.loads(BADGE_JSON_PATH.read_text())
        st.caption(f"Last evaluated: {payload.get('message', 'unknown')}")
    except (json.JSONDecodeError, OSError):
        pass


def pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.1%}"


st.title("📬 Email Triage RL — Warehouse Dashboard")
render_last_evaluated_banner()

tab_leaderboard, tab_confusion, tab_rewards, tab_quality, tab_episodes = st.tabs(
    ["🏆 Leaderboard", "🔀 Confusion Matrix", "⚙️ Reward Diagnostics",
     "🛡️ Data Quality", "📖 Episode Explorer"]
)

# --- Leaderboard -----------------------------------------------------------
with tab_leaderboard:
    df = load_leaderboard()
    if df.empty:
        st.info("No evaluation data yet. Run `python orchestration/pipeline_flow.py`.")
    else:
        tasks = sorted(df["task"].unique())
        task_filter = st.multiselect("Filter by task", tasks, default=tasks)
        view = df[df["task"].isin(task_filter)] if task_filter else df

        display = view.copy()
        for col in ["priority_accuracy", "category_accuracy", "route_accuracy",
                    "perfect_match_rate", "avg_task_reward", "phishing_catch_rate"]:
            display[col] = display[col].map(pct)
        st.dataframe(display, width='stretch', hide_index=True)

        chart_df = view.melt(
            id_vars=["model_name", "task"],
            value_vars=["priority_accuracy", "category_accuracy", "route_accuracy"],
            var_name="field", value_name="accuracy",
        )
        fig = px.bar(
            chart_df, x="task", y="accuracy", color="field", barmode="group",
            facet_col="model_name" if view["model_name"].nunique() > 1 else None,
            title="Per-field accuracy by task",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width='stretch')

# --- Confusion matrix -------------------------------------------------------
with tab_confusion:
    df = load_confusion_matrix()
    if df.empty:
        st.info("No confusion matrix data yet.")
    else:
        c1, c2, c3 = st.columns(3)
        model = c1.selectbox("Model", sorted(df["model_name"].unique()))
        task_opts = sorted(df.loc[df["model_name"] == model, "task"].unique())
        task = c2.selectbox("Task", task_opts)
        field_opts = sorted(
            df.loc[(df["model_name"] == model) & (df["task"] == task), "field"].unique()
        )
        field = c3.selectbox("Field", field_opts)

        sub = df[(df["model_name"] == model) & (df["task"] == task) & (df["field"] == field)]
        if sub.empty:
            st.info("No rows for this combination.")
        else:
            pivot = sub.pivot_table(
                index="true_value", columns="predicted_value", values="n",
                fill_value=0, aggfunc="sum",
            )
            fig = px.imshow(
                pivot, text_auto=True, aspect="auto",
                labels=dict(x="Predicted", y="True", color="Count"),
                title=f"{field} — {model} / {task}",
            )
            st.plotly_chart(fig, width='stretch')

# --- Reward diagnostics ------------------------------------------------------
with tab_rewards:
    df = load_reward_components()
    if df.empty:
        st.info("No reward component data yet.")
    else:
        nonzero = (
            df[REWARD_COMPONENT_COLUMNS].fillna(0).ne(0).mean()
            .rename("nonzero_fraction").reset_index().rename(columns={"index": "component"})
        )
        nonzero["flagged"] = nonzero["component"].isin(_STRUCTURALLY_LIMITED)

        st.caption(
            "Fraction of logged steps where each reward component is nonzero. "
            "Components marked ⚠️ below are documented (WAREHOUSE.md) as "
            "structurally near-unreachable under a reset-per-email eval "
            "pattern — a near-zero bar here is expected, not a bug."
        )
        nonzero_display = nonzero.copy()
        nonzero_display["component"] = nonzero_display.apply(
            lambda r: f"⚠️ {r['component']}" if r["flagged"] else r["component"], axis=1
        )
        fig = px.bar(
            nonzero_display, x="component", y="nonzero_fraction",
            title="Reward component activity",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width='stretch')

        component = st.selectbox("Component histogram", REWARD_COMPONENT_COLUMNS)
        fig2 = px.histogram(df, x=component, nbins=30, title=f"Distribution: {component}")
        st.plotly_chart(fig2, width='stretch')

# --- Data quality -------------------------------------------------------------
with tab_quality:
    df = load_data_quality()
    if df.empty:
        st.info("No data quality metrics yet.")
    else:
        summary = df[df["metric_type"] == "quality_flag_summary"]
        if not summary.empty and summary.iloc[0]["total_rows"]:
            rate = summary.iloc[0]["n_rows"] / summary.iloc[0]["total_rows"]
            st.metric("Violation rate", pct(rate),
                       help=f"{summary.iloc[0]['n_rows']} flagged / {summary.iloc[0]['total_rows']} total rows")

        flag_types = df[df["metric_type"] == "quality_flag_type"]
        if not flag_types.empty:
            st.plotly_chart(
                px.bar(flag_types.sort_values("n_rows", ascending=False),
                       x="metric_key", y="n_rows", title="Violations by flag type"),
                width='stretch',
            )

        reuse = df[df["metric_type"] == "template_reuse"]
        if not reuse.empty:
            st.subheader("Template reuse (top 20)")
            st.dataframe(
                reuse.sort_values("n_rows", ascending=False)
                     .head(20)[["template_pool", "template_idx", "n_rows"]],
                width='stretch', hide_index=True,
            )

# --- Episode explorer ----------------------------------------------------------
with tab_episodes:
    df = load_episode_summary()
    if df.empty:
        st.info("No episodes logged yet.")
    else:
        display = df.copy()
        display["avg_shaped_reward"] = display["avg_shaped_reward"].round(3)
        st.dataframe(display, width='stretch', hide_index=True)

        chosen = st.selectbox("Inspect an episode", df["episode_id"])
        row = df[df["episode_id"] == chosen].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Steps", int(row["n_steps"]))
        c2.metric("Avg shaped reward", f"{row['avg_shaped_reward']:.3f}")
        c3.metric("Reached done", "Yes" if row["reached_done"] else "No")
        c4.metric("Emails remaining at cutoff", int(row["emails_remaining_at_cutoff"] or 0))

        trace = load_episode_trace(chosen)
        if trace is None:
            st.info(
                "Step-by-step trace needs the Parquet lake (data/lake/) "
                "reachable from the warehouse/ directory — showing the "
                "episode-level summary above only."
            )
        elif trace.empty:
            st.info("No step rows found for this episode.")
        else:
            st.subheader("Step-by-step trace")
            st.dataframe(trace, width='stretch', hide_index=True)
            
            