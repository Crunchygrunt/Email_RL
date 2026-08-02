
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select step
from "warehouse"."main"."stg_env_steps"
where step is null



  
  
      
    ) dbt_internal_test