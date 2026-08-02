
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        true_category as value_field,
        count(*) as n_records

    from "warehouse"."main"."stg_env_steps"
    group by true_category

)

select *
from all_values
where value_field not in (
    'spam','newsletter','support','sales','internal','billing','security'
)



  
  
      
    ) dbt_internal_test