-- Per-episode rollups.
--
-- LOWER CONFIDENCE MART: given the reset-per-email finding in
-- WAREHOUSE.md, most "episodes" produced by running inference.py as-is
-- are exactly one email long, so n_steps will mostly read 1 here. This
-- mart is still correct for whatever episode structure actually occurred
-- -- full multi-email episodes if you drive the server directly (or fix
-- run_task's reset-per-email pattern), single-email "episodes" if you ran
-- inference.py unmodified.
--
-- reached_done / emails_remaining_at_cutoff (added after the
-- escalation-injection finding -- see WAREHOUSE.md): the environment can
-- grow its own email queue mid-episode (every missed urgent/high email
-- injects a follow-up escalation 2 positions ahead), but run_episodes.py
-- stops at a fixed MAX_STEPS regardless of whether the queue has grown
-- past that. n_steps / final_step alone can't distinguish "episode
-- genuinely finished" from "client gave up mid-escalation-spiral with
-- emails still queued" -- both can show n_steps=10. reached_done and
-- emails_remaining_at_cutoff, taken from the LAST logged step of each
-- episode, make that distinction explicit instead of silently averaging
-- the two outcomes together on the leaderboard.
--
-- Note: this is env_done (int_steps_joined aliases the env-side `done`
-- to disambiguate it from client_steps' own `done` field) -- NOT the
-- client's done, which just reflects run_episodes.py's local MAX_STEPS
-- cutoff rather than the environment's real completion state.

with last_step as (
    select
        episode_id,
        env_done          as reached_done,
        emails_remaining  as emails_remaining_at_cutoff
    from "warehouse"."main"."int_steps_joined"
    qualify row_number() over (
        partition by episode_id
        order by env_step desc
    ) = 1
)

select
    s.episode_id,
    min(s.env_dt)                     as episode_dt,
    count(*)                          as n_steps,
    sum(s.shaped_reward)              as total_shaped_reward,
    avg(s.shaped_reward)              as avg_shaped_reward,
    max(s.current_streak)             as max_streak,
    bool_or(s.is_business_critical)   as had_business_critical_email,
    bool_or(s.is_phishing)            as had_phishing_email,
    max(s.env_step)                   as final_step,
    any_value(l.reached_done)         as reached_done,
    any_value(l.emails_remaining_at_cutoff) as emails_remaining_at_cutoff
from "warehouse"."main"."int_steps_joined" s
join last_step l using (episode_id)
group by 1
order by episode_dt, episode_id