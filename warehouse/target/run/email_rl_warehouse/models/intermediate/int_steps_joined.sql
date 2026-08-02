
  
  create view "warehouse"."main"."int_steps_joined__dbt_tmp" as (
    -- Joins client_steps onto env_steps by email_id -- see WAREHOUSE.md
-- "Join key" section for why email_id (not episode_id/session_id) is the
-- primary key across these two streams.
--
-- LEFT JOIN, not INNER: env_steps is authoritative. Every graded email
-- produces an env_step row regardless of who called step() or whether
-- client-side telemetry logged successfully (event_sink is fail-open by
-- design, so a client_steps write can silently fail without breaking the
-- run). An INNER JOIN would silently drop those rows instead of surfacing
-- them as env-only.

select
    env.episode_id,
    env.step                 as env_step,
    env.email_id,
    env.predicted_priority   as env_predicted_priority,
    env.predicted_category   as env_predicted_category,
    env.predicted_route      as env_predicted_route,
    env.true_priority,
    env.true_category,
    env.true_route,
    env.is_business_critical,
    env.is_phishing,
    env.cluster_id,
    env.is_escalation,
    env.priority_ok,
    env.category_ok,
    env.route_ok,
    env.is_perfect,
    env.base_score,
    env.urgency_multiplier,
    env.reward_components,
    env.shaped_reward,
    env.current_streak,
    env.done                 as env_done,
    env.stateless_http_mode,
    env.emails_remaining,
    env.template_pool,
    env.template_idx,
    env.email_quality_flags,
    env.dt                   as env_dt,
    cli.model_name,
    cli.task,
    cli.step                 as client_step,
    cli.predicted_priority   as client_predicted_priority,
    cli.predicted_category   as client_predicted_category,
    cli.predicted_route      as client_predicted_route,
    cli.action_plan,
    cli.threat_report,
    cli.task_reward,
    cli.llm_latency_ms,
    cli.session_id,
    cli.error                as client_error,
    cli.dt                   as client_dt
from "warehouse"."main"."stg_env_steps" env
left join "warehouse"."main"."stg_client_steps" cli
    on env.email_id = cli.email_id
  );
