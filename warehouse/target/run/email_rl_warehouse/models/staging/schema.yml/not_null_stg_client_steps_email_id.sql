
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select email_id
from "warehouse"."main"."stg_client_steps"
where email_id is null



  
  
      
    ) dbt_internal_test