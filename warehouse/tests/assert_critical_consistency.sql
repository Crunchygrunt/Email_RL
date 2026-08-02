-- Business-critical emails are always constructed with
-- true_route='human_review' regardless of category (see _generate_email()
-- in server/Email_RL_environment.py). Any row violating that is either a
-- generator regression or a telemetry-field mixup.

select *
from {{ ref('stg_env_steps') }}
where is_business_critical = true
  and true_route != 'human_review'
