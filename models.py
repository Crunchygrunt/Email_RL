# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.


# """
# Data models for the Email Triage RL Environment.

# The agent receives an email (subject, sender, body) and must output
# three structured fields: priority, category, and route.

# Valid values
# ------------
# priority  : low | medium | high | urgent
# category  : spam | newsletter | support | sales | internal | billing | security
# route     : inbox | archive | support_team | sales_team |
#             security_team | billing_team | trash
# """

# from typing import Optional
# from openenv.core.env_server.types import Action, Observation
# from pydantic import Field


# # ── Domain constants (single source of truth, shared with server package) ──

# PRIORITIES = ("low", "medium", "high", "urgent")
# CATEGORIES = ("spam", "newsletter", "support", "sales", "internal", "billing", "security")
# ROUTES = (
#     "inbox",
#     "archive",
#     "support_team",
#     "sales_team",
#     "security_team",
#     "billing_team",
#     "trash",
# )

# # Canonical route for each category
# ROUTE_MAP = {
#     "security":   "security_team",
#     "billing":    "billing_team",
#     "support":    "support_team",
#     "sales":      "sales_team",
#     "internal":   "inbox",
#     "newsletter": "archive",
#     "spam":       "trash",
# }

# # Per-priority urgency multiplier applied on top of base correctness score
# URGENCY_BONUS = {
#     "urgent": 2.0,
#     "high":   1.5,
#     "medium": 1.0,
#     "low":    0.8,
# }


# class EmailTriageAction(Action):
#     """
#     Triage decision for a single email.

#     The agent must classify the email along three orthogonal axes:
#       priority  — how urgently should this be handled?
#       category  — what type of email is this?
#       route     — which queue / team should receive it?
#     """

#     priority: str = Field(
#         ...,
#         description="Urgency level. One of: low, medium, high, urgent",
#     )
#     category: str = Field(
#         ...,
#         description=(
#             "Type of email. "
#             "One of: spam, newsletter, support, sales, internal, billing, security"
#         ),
#     )
#     route: str = Field(
#         ...,
#         description=(
#             "Destination queue. "
#             "One of: inbox, archive, support_team, sales_team, "
#             "security_team, billing_team, trash"
#         ),
#     )


# class EmailTriageObservation(Observation):
#     """
#     Observation returned by the environment after reset() or step().

#     Contains the next email to triage plus feedback about the previous action
#     and episode-level bookkeeping.
#     """

#     # ── Current email to triage ────────────────────────────────────────────
#     email_id: str = Field(default="", description="Opaque unique ID for this email")
#     email_subject: str = Field(default="", description="Subject line of the email")
#     email_sender: str = Field(default="", description="Sender address")
#     email_body: str = Field(default="", description="Full body text of the email")

#     # ── Feedback about the immediately preceding action (None on first obs) ─
#     last_priority_correct: Optional[bool] = Field(
#         default=None,
#         description="Was the priority field correct in the previous action?",
#     )
#     last_category_correct: Optional[bool] = Field(
#         default=None,
#         description="Was the category field correct in the previous action?",
#     )
#     last_route_correct: Optional[bool] = Field(
#         default=None,
#         description="Was the route field correct in the previous action?",
#     )

#     # ── Episode bookkeeping ────────────────────────────────────────────────
#     emails_remaining: int = Field(
#         default=0,
#         description="Emails left to process after this one (0 = this is the last)",
#     )
#     current_streak: int = Field(
#         default=0,
#         description="Consecutive perfectly-triaged emails so far this episode",
#     )




# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Email Triage RL Environment.

The agent receives an email (subject, sender, body) and must output
three structured fields: priority, category, and route.

Valid values
------------
priority  : low | medium | high | urgent
category  : spam | newsletter | support | sales | internal | billing | security
route     : inbox | archive | support_team | sales_team |
            security_team | billing_team | trash | human_review
"""

from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ── Domain constants (single source of truth, shared with server package) ──

PRIORITIES = ("low", "medium", "high", "urgent")
CATEGORIES = ("spam", "newsletter", "support", "sales", "internal", "billing", "security")
ROUTES = (
    "inbox",
    "archive",
    "support_team",
    "sales_team",
    "security_team",
    "billing_team",
    "trash",
    "human_review",          # for business-critical emails requiring human sign-off
)

# Canonical route for each category (standard, non-critical emails)
ROUTE_MAP = {
    "security":   "security_team",
    "billing":    "billing_team",
    "support":    "support_team",
    "sales":      "sales_team",
    "internal":   "inbox",
    "newsletter": "archive",
    "spam":       "trash",
}

# Per-priority urgency multiplier applied on top of base correctness score
URGENCY_BONUS = {
    "urgent": 2.0,
    "high":   1.5,
    "medium": 1.0,
    "low":    0.8,
}


class EmailTriageAction(Action):
    """
    Triage decision for a single email.

    The agent must classify the email along three orthogonal axes:
      priority  — how urgently should this be handled?
      category  — what type of email is this?
      route     — which queue / team should receive it?

    For business-critical emails (legal disputes, large contract negotiations,
    compliance violations, insurance claims, policy changes) the correct route
    is 'human_review' regardless of category.
    """

    priority: str = Field(
        ...,
        description="Urgency level. One of: low, medium, high, urgent",
    )
    category: str = Field(
        ...,
        description=(
            "Type of email. "
            "One of: spam, newsletter, support, sales, internal, billing, security"
        ),
    )
    route: str = Field(
        ...,
        description=(
            "Destination queue. "
            "One of: inbox, archive, support_team, sales_team, "
            "security_team, billing_team, trash, human_review"
        ),
    )


class EmailTriageObservation(Observation):
    """
    Observation returned by the environment after reset() or step().

    Contains the next email to triage plus feedback about the previous action
    and episode-level bookkeeping.
    """

    # ── Current email to triage ────────────────────────────────────────────
    email_id: str = Field(default="", description="Opaque unique ID for this email")
    email_subject: str = Field(default="", description="Subject line of the email")
    email_sender: str = Field(default="", description="Sender address")
    email_body: str = Field(default="", description="Full body text of the email")

    # ── Feedback about the immediately preceding action (None on first obs) ─
    last_priority_correct: Optional[bool] = Field(
        default=None,
        description="Was the priority field correct in the previous action?",
    )
    last_category_correct: Optional[bool] = Field(
        default=None,
        description="Was the category field correct in the previous action?",
    )
    last_route_correct: Optional[bool] = Field(
        default=None,
        description="Was the route field correct in the previous action?",
    )

    # ── Episode bookkeeping ────────────────────────────────────────────────
    emails_remaining: int = Field(
        default=0,
        description="Emails left to process after this one (0 = this is the last)",
    )
    current_streak: int = Field(
        default=0,
        description="Consecutive perfectly-triaged emails so far this episode",
    )