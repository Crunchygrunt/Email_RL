-- Thin select off the raw Parquet lake for client-side client_steps events
-- (logged from inside inference.py's run_task() -- see
-- telemetry/event_sink.py::log_client_step).

with source as (
    select *
    from read_parquet('{{ var("client_steps_glob") }}', hive_partitioning = true)
)

select
    event_type,
    logged_at,
    model_name,
    task,
    step,
    email_id,
    predicted_priority,
    predicted_category,
    predicted_route,
    action_plan,
    threat_report,
    task_reward,
    done,
    llm_latency_ms,
    session_id,
    error,
    parse_ok,
    raw_response_snippet,
    dt
from source

