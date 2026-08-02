-- Regression guard for a real bug found and fixed in
-- server/Email_RL_environment.py: _sample_episode() used to always drop
-- the same standard category ("security") from every single episode, so
-- it was never represented in any collected data (confirmed empirically:
-- 0/300 sampled episodes). This test fails if any of the 7 standard
-- categories drops below 5% of standard-slot rows in the WHOLE collected
-- dataset -- loose enough to tolerate normal sampling variance (each
-- category is expected at ~14.3%), tight enough to catch a total or
-- near-total omission like the original bug.
--
-- Only evaluated once there's a reasonably sized dataset (>50 standard
-- rows), so a handful of debug episodes doesn't produce a flaky failure.
-- "Standard" rows exclude phishing (always mislabeled true_category=
-- 'security' by design) and cluster/critical rows, which aren't part of
-- the per-category sampling this test is guarding.
--
-- IMPORTANT: a category with ZERO occurrences never produces a row from
-- a plain GROUP BY -- which is exactly the failure mode this test exists
-- to catch. All 7 categories are enumerated explicitly and LEFT JOINed
-- against the observed counts so a fully-absent category still shows up
-- as n=0 instead of silently not appearing at all.

with all_categories as (
    select unnest(['spam', 'newsletter', 'support', 'sales', 'internal', 'billing', 'security']) as true_category
),
standard_rows as (
    select true_category
    from "warehouse"."main"."stg_env_steps"
    where not is_phishing
      and not is_business_critical
      and cluster_id is null
),
category_counts as (
    select
        all_categories.true_category,
        count(standard_rows.true_category) as n
    from all_categories
    left join standard_rows
        on all_categories.true_category = standard_rows.true_category
    group by all_categories.true_category
),
total as (
    select count(*) as total_n from standard_rows
)
select category_counts.true_category, category_counts.n, total.total_n
from category_counts, total
where total.total_n > 50
  and category_counts.n < 0.05 * total.total_n