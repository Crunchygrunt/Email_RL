-- Generator / data-quality health metrics over the full collected
-- dataset. Complements the pass/fail dbt tests in warehouse/tests/ (which
-- only tell you IF something's wrong) with actual numbers to look at: how
-- often email_quality_flags fires, which specific invariants get
-- violated and how often, and how evenly the synthetic email template
-- pool is actually being sampled across every real collection run (not
-- just a one-off validate_synthetic_emails.py sampling pass).
--
-- Two grains in one mart, unioned together under `metric_type` so both
-- are queryable from a single table:
--   'quality_flag_summary' -- one row: overall violation rate
--   'quality_flag_type'    -- one row per distinct flag string, with count
--   'template_reuse'       -- one row per (template_pool, template_idx),
--                              with how many times it was actually served

with base as (
    select *
    from {{ ref('stg_env_steps') }}
),

flag_summary as (
    select
        'quality_flag_summary'  as metric_type,
        cast(null as varchar)   as metric_key,
        count(*)                as total_rows,
        count(*) filter (
            where email_quality_flags is not null
              and email_quality_flags != '[]'
        )                       as n_rows,
        cast(null as varchar)   as template_pool,
        cast(null as integer)   as template_idx
    from base
),

flags_exploded as (
    select unnest(from_json(email_quality_flags, '["VARCHAR"]')) as flag
    from base
    where email_quality_flags is not null
      and email_quality_flags != '[]'
),

flag_type_counts as (
    select
        'quality_flag_type'     as metric_type,
        flag                    as metric_key,
        cast(null as bigint)    as total_rows,
        count(*)                as n_rows,
        cast(null as varchar)   as template_pool,
        cast(null as integer)   as template_idx
    from flags_exploded
    group by flag
),

template_reuse as (
    select
        'template_reuse'        as metric_type,
        template_pool || ':' || cast(template_idx as varchar) as metric_key,
        cast(null as bigint)    as total_rows,
        count(*)                as n_rows,
        template_pool,
        template_idx
    from base
    where template_pool is not null
      and template_idx is not null
    group by template_pool, template_idx
)

select * from flag_summary
union all
select * from flag_type_counts
union all
select * from template_reuse
