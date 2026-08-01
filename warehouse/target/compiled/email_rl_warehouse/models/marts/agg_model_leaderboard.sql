-- Accuracy per field, perfect-match rate, and phishing catch rate, per
-- model_name x task.
--
-- phishing_catch_rate is populated now that the _sample_episode() bug
-- that dropped phishing emails from every episode has been fixed (see
-- WAREHOUSE.md finding #2). If you're rebuilding this against telemetry
-- generated before that fix, expect n_phishing_emails_seen = 0 and
-- phishing_catch_rate = NULL instead -- that reflects the data, not this
-- mart.

select
    model_name,
    task,
    count(*)                                          as n_graded_emails,
    avg(case when priority_ok then 1.0 else 0.0 end)  as priority_accuracy,
    avg(case when category_ok then 1.0 else 0.0 end)  as category_accuracy,
    avg(case when route_ok    then 1.0 else 0.0 end)  as route_accuracy,
    avg(case when is_perfect  then 1.0 else 0.0 end)  as perfect_match_rate,
    avg(task_reward)                                  as avg_task_reward,
    sum(case when is_phishing then 1 else 0 end)      as n_phishing_emails_seen,
    avg(
        case when is_phishing
             then case when category_ok and route_ok then 1.0 else 0.0 end
        end
    )                                                  as phishing_catch_rate
from "warehouse"."main"."int_steps_joined"
where model_name is not null
group by 1, 2
order by model_name, task