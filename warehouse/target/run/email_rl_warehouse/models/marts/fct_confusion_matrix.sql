
  
    
    

    create  table
      "warehouse"."main"."fct_confusion_matrix__dbt_tmp"
  
    as (
      -- Predicted-vs-true counts across priority/category/route, grouped by
-- model_name x task. Uses the client-reported prediction (what the LLM
-- actually output for that task's action space); rows with no client_steps
-- match (model_name is null) are excluded here since there's no prediction
-- to compare -- see fct_reward_components / int_steps_joined for the
-- env-only view of those rows.

with base as (
    select model_name, task, 'priority' as field,
           true_priority as true_value, client_predicted_priority as predicted_value
    from "warehouse"."main"."int_steps_joined"
    where model_name is not null

    union all

    select model_name, task, 'category' as field,
           true_category as true_value, client_predicted_category as predicted_value
    from "warehouse"."main"."int_steps_joined"
    where model_name is not null

    union all

    select model_name, task, 'route' as field,
           true_route as true_value, client_predicted_route as predicted_value
    from "warehouse"."main"."int_steps_joined"
    where model_name is not null
)

select
    model_name,
    task,
    field,
    true_value,
    predicted_value,
    count(*) as n
from base
group by 1, 2, 3, 4, 5
    );
  
  