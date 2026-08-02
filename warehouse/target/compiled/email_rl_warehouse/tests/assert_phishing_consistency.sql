-- Phishing emails are always constructed with true_category='security' and
-- true_route='security_team' (see _generate_email() in
-- server/Email_RL_environment.py). Any row violating that is either a
-- generator regression or a telemetry-field mixup.

select *
from "warehouse"."main"."stg_env_steps"
where is_phishing = true
  and (true_category != 'security' or true_route != 'security_team')