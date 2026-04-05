# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# """Email Rl Environment Client."""

# from typing import Dict

# from openenv.core import EnvClient
# from openenv.core.client_types import StepResult
# from openenv.core.env_server.types import State

# from .models import EmailRlAction, EmailRlObservation


# class EmailRlEnv(
#     EnvClient[EmailRlAction, EmailRlObservation, State]
# ):
#     """
#     Client for the Email Rl Environment.

#     This client maintains a persistent WebSocket connection to the environment server,
#     enabling efficient multi-step interactions with lower latency.
#     Each client instance has its own dedicated environment session on the server.

#     Example:
#         >>> # Connect to a running server
#         >>> with EmailRlEnv(base_url="http://localhost:8000") as client:
#         ...     result = client.reset()
#         ...     print(result.observation.echoed_message)
#         ...
#         ...     result = client.step(EmailRlAction(message="Hello!"))
#         ...     print(result.observation.echoed_message)

#     Example with Docker:
#         >>> # Automatically start container and connect
#         >>> client = EmailRlEnv.from_docker_image("Email_RL-env:latest")
#         >>> try:
#         ...     result = client.reset()
#         ...     result = client.step(EmailRlAction(message="Test"))
#         ... finally:
#         ...     client.close()
#     """

#     def _step_payload(self, action: EmailRlAction) -> Dict:
#         """
#         Convert EmailRlAction to JSON payload for step message.

#         Args:
#             action: EmailRlAction instance

#         Returns:
#             Dictionary representation suitable for JSON encoding
#         """
#         return {
#             "message": action.message,
#         }

#     def _parse_result(self, payload: Dict) -> StepResult[EmailRlObservation]:
#         """
#         Parse server response into StepResult[EmailRlObservation].

#         Args:
#             payload: JSON response data from server

#         Returns:
#             StepResult with EmailRlObservation
#         """
#         obs_data = payload.get("observation", {})
#         observation = EmailRlObservation(
#             echoed_message=obs_data.get("echoed_message", ""),
#             message_length=obs_data.get("message_length", 0),
#             done=payload.get("done", False),
#             reward=payload.get("reward"),
#             metadata=obs_data.get("metadata", {}),
#         )

#         return StepResult(
#             observation=observation,
#             reward=payload.get("reward"),
#             done=payload.get("done", False),
#         )

#     def _parse_state(self, payload: Dict) -> State:
#         """
#         Parse server response into State object.

#         Args:
#             payload: JSON response from state request

#         Returns:
#             State object with episode_id and step_count
#         """
#         return State(
#             episode_id=payload.get("episode_id"),
#             step_count=payload.get("step_count", 0),
#         )



# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Email Triage RL Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import EmailTriageAction, EmailTriageObservation


class EmailTriageEnv(
    EnvClient[EmailTriageAction, EmailTriageObservation, State]
):
    """
    Client for the Email Triage RL Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling low-latency multi-step interactions.  Each client instance has
    its own dedicated environment session on the server.

    Example — connect to a running server::

        with EmailTriageEnv(base_url="http://localhost:8000") as client:
            result = client.reset()
            obs = result.observation
            print(obs.email_subject)    # first email to triage

            action = EmailTriageAction(
                priority="medium",
                category="sales",
                route="sales_team",
            )
            result = client.step(action)
            print(result.reward)         # shaped reward for that decision

    Example — start container automatically then connect::

        client = EmailTriageEnv.from_docker_image("email_rl-env:latest")
        try:
            result = client.reset()
            result = client.step(EmailTriageAction(
                priority="urgent",
                category="security",
                route="security_team",
            ))
        finally:
            client.close()
    """

    def _step_payload(self, action: EmailTriageAction) -> Dict:
        """
        Serialise EmailTriageAction to JSON payload for the step message.

        Args:
            action: EmailTriageAction instance.

        Returns:
            Dictionary suitable for JSON encoding.
        """
        return {
            "priority": action.priority,
            "category": action.category,
            "route":    action.route,
        }

    def _parse_result(self, payload: Dict) -> StepResult[EmailTriageObservation]:
        """
        Deserialise the server response into StepResult[EmailTriageObservation].

        Args:
            payload: Raw JSON response from the server.

        Returns:
            StepResult containing EmailTriageObservation.
        """
        obs_data = payload.get("observation", {})

        observation = EmailTriageObservation(
            # Current email
            email_id      = obs_data.get("email_id", ""),
            email_subject = obs_data.get("email_subject", ""),
            email_sender  = obs_data.get("email_sender", ""),
            email_body    = obs_data.get("email_body", ""),
            # Feedback
            last_priority_correct = obs_data.get("last_priority_correct"),
            last_category_correct = obs_data.get("last_category_correct"),
            last_route_correct    = obs_data.get("last_route_correct"),
            # Episode info
            emails_remaining = obs_data.get("emails_remaining", 0),
            current_streak   = obs_data.get("current_streak", 0),
            # Standard Observation fields
            done     = payload.get("done", False),
            reward   = payload.get("reward"),
            metadata = obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Deserialise server response into a State object.

        Args:
            payload: JSON response from the /state endpoint.

        Returns:
            State with episode_id and step_count.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )