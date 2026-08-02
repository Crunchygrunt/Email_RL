
    
    

with all_values as (

    select
        true_priority as value_field,
        count(*) as n_records

    from "warehouse"."main"."stg_env_steps"
    group by true_priority

)

select *
from all_values
where value_field not in (
    'low','medium','high','urgent'
)


