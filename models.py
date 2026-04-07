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


# -- Domain constants (single source of truth, shared with server package) --

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
    "human_review",
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

# Task difficulty definitions
TASK_DIFFICULTIES = {
    "spam-detection":         "easy",
    "priority-classification": "medium",
    "full-triage":            "hard",
    "critical-escalation":    "hard",
}


class EmailTriageAction(Action):
    """
    Triage decision for a single email.

    The agent must classify the email along three orthogonal axes:
      priority  -- how urgently should this be handled?
      category  -- what type of email is this?
      route     -- which queue / team should receive it?

    For advanced tasks, the agent also outputs:
      action_plan    -- JSON string describing what actions to take
      threat_report  -- JSON string with security threat assessment

    For business-critical emails (legal disputes, large contract negotiations,
    compliance violations, insurance claims, policy changes) the correct route
    is 'human_review' regardless of category.

    For phishing / social engineering emails, the correct category is 'security'
    and the correct route is 'security_team', even if the email appears to be
    a billing invoice, internal request, or IT notification.
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
    action_plan: Optional[str] = Field(
        default=None,
        description="JSON action plan for orchestrator task (optional)",
    )
    threat_report: Optional[str] = Field(
        default=None,
        description="JSON threat assessment for security task (optional)",
    )


class EmailTriageObservation(Observation):
    """
    Observation returned by the environment after reset() or step().

    Contains the next email to triage plus feedback about the previous action
    and episode-level bookkeeping.

    NOTE: metadata does NOT contain ground truth for the current email.
    Ground truth is only provided for the previously-graded email
    (under graded_true_* keys) so client-side graders can score correctly
    without leaking answers to the agent.
    """

    # -- Current email to triage ----------------------------------------
    email_id: str = Field(default="", description="Opaque unique ID for this email")
    email_subject: str = Field(default="", description="Subject line of the email")
    email_sender: str = Field(default="", description="Sender address")
    email_body: str = Field(default="", description="Full body text of the email")

    # -- Feedback about the immediately preceding action (None on first obs)
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

    # -- Episode bookkeeping --------------------------------------------
    emails_remaining: int = Field(
        default=0,
        description="Emails left to process after this one (0 = this is the last)",
    )
    current_streak: int = Field(
        default=0,
        description="Consecutive perfectly-triaged emails so far this episode",
    )

    # -- Ground truth for client-side graders ----------------------------
    # These fields carry the correct labels for the CURRENT email.
    # The LLM agent prompt must NOT include these fields.
    # They exist solely so that inference.py graders can score correctly
    # in stateless HTTP mode where each request hits a fresh env instance.
    true_priority: Optional[str] = Field(
        default=None,
        description="Ground truth priority for grader use (not shown to agent)",
    )
    true_category: Optional[str] = Field(
        default=None,
        description="Ground truth category for grader use (not shown to agent)",
    )
    true_route: Optional[str] = Field(
        default=None,
        description="Ground truth route for grader use (not shown to agent)",
    )
    is_business_critical: bool = Field(
        default=False,
        description="Whether this email requires human_review routing",
    )
    is_phishing: bool = Field(
        default=False,
        description="Whether this email is a phishing attempt",
    )
    linked_incident: bool = Field(
        default=False,
        description="Whether this email is part of a cross-email dependency cluster",
    )