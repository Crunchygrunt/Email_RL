
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select true_category
from "warehouse"."main"."stg_env_steps"
where true_category is null



  
  
      
    ) dbt_internal_test