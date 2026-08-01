-- Per-episode rollups.
--
-- LOWER CONFIDENCE MART: given the reset-per-email finding in
-- WAREHOUSE.md, most "episodes" produced by running inference.py as-is
-- are exactly one email long, so n_steps will mostly read 1 here. This
-- mart is still correct for whatever episode structure actually occurred
-- -- full multi-email episodes if you drive the server directly (or fix
-- run_task's reset-per-email pattern), single-email "episodes" if you ran
-- inference.py unmodified.

select
    episode_id,
    min(env_dt)                    as episode_dt,
    count(*)                       as n_steps,
    sum(shaped_reward)             as total_shaped_reward,
    avg(shaped_reward)             as avg_shaped_reward,
    max(current_streak)            as max_streak,
    bool_or(is_business_critical)  as had_business_critical_email,
    bool_or(is_phishing)           as had_phishing_email,
    max(env_step)                  as final_step
from {{ ref('int_steps_joined') }}
group by 1
order by episode_dt, episode_id
