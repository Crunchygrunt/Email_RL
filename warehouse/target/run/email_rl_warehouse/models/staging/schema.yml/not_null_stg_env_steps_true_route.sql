
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select true_route
from "warehouse"."main"."stg_env_steps"
where true_route is null



  
  
      
    ) dbt_internal_test