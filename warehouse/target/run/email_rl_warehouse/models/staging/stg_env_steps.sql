
  
  create view "warehouse"."main"."stg_env_steps__dbt_tmp" as (
    -- Thin select off the raw Parquet lake for server-side env_steps events
-- (logged from inside EmailTriageEnvironment.step() -- see
-- telemetry/event_sink.py::log_env_step).
--
-- hive_partitioning=true turns the dt=YYYY-MM-DD folder name into a
-- queryable `dt` column automatically. Each row also carries its own `dt`
-- field written at log time (see event_sink.py) -- harmless duplication,
-- kept as a cross-check in case partition-folder naming ever drifts from
-- the logged timestamp.

with source as (
    select *
    from read_parquet('../data/lake/env_steps/dt=*/*.parquet', hive_partitioning = true)
)

select
    event_type,
    logged_at,
    episode_id,
    step,
    email_id,
    predicted_priority,
    predicted_category,
    predicted_route,
    true_priority,
    true_category,
    true_route,
    is_business_critical,
    is_phishing,
    cluster_id,
    is_escalation,
    priority_ok,
    category_ok,
    route_ok,
    is_perfect,
    base_score,
    urgency_multiplier,
    reward_components,   -- JSON string; unpacked in marts/fct_reward_components.sql
    shaped_reward,
    current_streak,
    done,
    stateless_http_mode,
    emails_remaining,
    template_pool,          -- data quality gate Layer 1/3: which template pool generated this email
    template_idx,           -- nullable (cluster emails are identified by cluster_id instead)
    email_quality_flags,    -- JSON list string; data quality gate Layer 2 runtime invariant checks
    dt
from source
  );
