
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select true_priority
from "warehouse"."main"."stg_env_steps"
where true_priority is null



  
  
      
    ) dbt_internal_test