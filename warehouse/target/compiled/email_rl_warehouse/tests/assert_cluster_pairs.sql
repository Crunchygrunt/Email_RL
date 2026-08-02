-- Every dependency cluster is authored as exactly 2 linked emails (see
-- _DEPENDENCY_CLUSTERS in server/Email_RL_environment.py). A dbt test
-- passes when the query returns ZERO rows -- so this returns any
-- (episode_id, cluster_id) pair that does NOT have exactly 2 rows.

select
    episode_id,
    cluster_id,
    count(*) as n_rows
from "warehouse"."main"."stg_env_steps"
where cluster_id is not null
group by episode_id, cluster_id
having count(*) > 2