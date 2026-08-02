
    
    

with all_values as (

    select
        true_route as value_field,
        count(*) as n_records

    from "warehouse"."main"."stg_env_steps"
    group by true_route

)

select *
from all_values
where value_field not in (
    'inbox','archive','support_team','sales_team','security_team','billing_team','trash','human_review'
)


