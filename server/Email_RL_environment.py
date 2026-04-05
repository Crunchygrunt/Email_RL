# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# """
# Email Rl Environment Implementation.

# A simple test environment that echoes back messages sent to it.
# Perfect for testing HTTP server infrastructure.
# """

# from uuid import uuid4

# from openenv.core.env_server.interfaces import Environment
# from openenv.core.env_server.types import State

# try:
#     from ..models import EmailRlAction, EmailRlObservation
# except ImportError:
#     from models import EmailRlAction, EmailRlObservation


# class EmailRlEnvironment(Environment):
#     """
#     A simple echo environment that echoes back messages.

#     This environment is designed for testing the HTTP server infrastructure.
#     It maintains minimal state and simply echoes back whatever message it receives.

#     Example:
#         >>> env = EmailRlEnvironment()
#         >>> obs = env.reset()
#         >>> print(obs.echoed_message)  # "Email Rl environment ready!"
#         >>>
#         >>> obs = env.step(EmailRlAction(message="Hello"))
#         >>> print(obs.echoed_message)  # "Hello"
#         >>> print(obs.message_length)  # 5
#     """

#     # Enable concurrent WebSocket sessions.
#     # Set to True if your environment isolates state between instances.
#     # When True, multiple WebSocket clients can connect simultaneously, each
#     # getting their own environment instance (when using factory mode in app.py).
#     SUPPORTS_CONCURRENT_SESSIONS: bool = True

#     def __init__(self):
#         """Initialize the Email_RL environment."""
#         self._state = State(episode_id=str(uuid4()), step_count=0)
#         self._reset_count = 0

#     def reset(self) -> EmailRlObservation:
#         """
#         Reset the environment.

#         Returns:
#             EmailRlObservation with a ready message
#         """
#         self._state = State(episode_id=str(uuid4()), step_count=0)
#         self._reset_count += 1

#         return EmailRlObservation(
#             echoed_message="Email Rl environment ready!",
#             message_length=0,
#             done=False,
#             reward=0.0,
#         )

#     def step(self, action: EmailRlAction) -> EmailRlObservation:  # type: ignore[override]
#         """
#         Execute a step in the environment by echoing the message.

#         Args:
#             action: EmailRlAction containing the message to echo

#         Returns:
#             EmailRlObservation with the echoed message and its length
#         """
#         self._state.step_count += 1

#         message = action.message
#         length = len(message)

#         # Simple reward: longer messages get higher rewards
#         reward = length * 0.1

#         return EmailRlObservation(
#             echoed_message=message,
#             message_length=length,
#             done=False,
#             reward=reward,
#             metadata={"original_message": message, "step": self._state.step_count},
#         )

#     @property
#     def state(self) -> State:
#         """
#         Get the current environment state.

#         Returns:
#             Current State with episode_id and step_count
#         """
#         return self._state

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Email Triage RL Environment Implementation.

A real-world email classification environment where the agent triages
synthetic business emails by assigning a priority, category, and route.

Reward Design (from email-triage notebook)
------------------------------------------
Base score per email:
    +1.0  correct priority   (most important signal)
    +0.5  correct category
    +0.3  correct route
    +0.1  format bonus  (priority correct AND ≥ 1 other field parsed)
    +0.2  perfect bonus (all three correct)
    → max base score = 2.1 per email

Shaped reward applied in step():
    base_score × urgency_multiplier + streak_bonus - overload_penalty

    urgency_multiplier: urgent=2.0, high=1.5, medium=1.0, low=0.8
    streak_bonus      : +0.3 when current_streak ≥ 3 consecutive perfect
    overload_penalty  : -0.5 when agent mislabels an urgent email as low/medium

Business-critical emails:
    A subset of emails require human sign-off regardless of category —
    legal disputes, large contract negotiations, GDPR/compliance violations,
    insurance claims, and policy changes.  These are tagged with
    is_business_critical=True in obs.metadata and have true_route='human_review'.
    The critical-escalation grader in inference.py scores the agent on whether
    it correctly routes these to 'human_review' while not over-escalating
    normal emails.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        CATEGORIES,
        PRIORITIES,
        ROUTE_MAP,
        ROUTES,
        URGENCY_BONUS,
        EmailTriageAction,
        EmailTriageObservation,
    )
except ImportError:
    from models import (
        CATEGORIES,
        PRIORITIES,
        ROUTE_MAP,
        ROUTES,
        URGENCY_BONUS,
        EmailTriageAction,
        EmailTriageObservation,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic email dataset
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (subject_template, body_template, priority, category)
# {amount}, {name}, {id}, {day}, {product}, {plan} are filled at generation time.

_EMAIL_TEMPLATES: List[Tuple[str, str, str, str]] = [
    # ── spam ──────────────────────────────────────────────────────────────
    (
        "You've been selected for a FREE {product}!",
        "Congratulations! You have been chosen to receive a complimentary {product} worth ${amount}. "
        "Click here immediately to claim your prize before it expires!",
        "low", "spam",
    ),
    (
        "URGENT: Claim your ${amount} reward NOW",
        "Your exclusive reward of ${amount} is waiting. This offer expires in 24 hours. "
        "No purchase necessary. Reply STOP to unsubscribe.",
        "low", "spam",
    ),
    (
        "You won a ${amount} {product}! Claim now!!!",
        "Dear winner, our automated lottery selected your email address. "
        "Send your bank details to release your ${amount} prize. Act fast!",
        "low", "spam",
    ),
    (
        "Re: Important account notification",
        "Your PayPаl account has been limited. Click the link below to verify "
        "your identity and restore full access: http://paypa1-secure.xyz/verify",
        "low", "spam",
    ),
    # ── newsletter ────────────────────────────────────────────────────────
    (
        "Your {product} Weekly Digest — {day}",
        "Here's what's new this week: top articles, product updates, and community highlights. "
        "Unsubscribe at any time using the link below.",
        "low", "newsletter",
    ),
    (
        "📰 Monthly roundup: trends in {product}",
        "Hi there, catch up on this month's best content from our editorial team. "
        "Features: industry news, how-to guides, and expert interviews.",
        "low", "newsletter",
    ),
    (
        "We've updated our {product} features — here's what's new",
        "Release notes for v{id}: improved performance, new dashboard widgets, "
        "and several bug fixes. Read the full changelog on our blog.",
        "low", "newsletter",
    ),
    # ── support ───────────────────────────────────────────────────────────
    (
        "Re: Ticket #{id} — {product} not working",
        "Hi Support, I opened ticket #{id} three days ago about {product} failing to load. "
        "I haven't heard back. Could you please provide an update? This is blocking my work.",
        "medium", "support",
    ),
    (
        "Follow-up: still experiencing login issues",
        "I submitted a bug report last week (ref #{id}). The login page still throws "
        "error 403 for my account. I've tried clearing cache and cookies with no luck.",
        "high", "support",
    ),
    (
        "Production outage — {product} down for all users",
        "CRITICAL: Our {product} integration has been completely unavailable for the past hour. "
        "All users are affected. We are losing revenue. Please escalate immediately.",
        "urgent", "support",
    ),
    (
        "Question about {product} configuration",
        "Hello, I'm trying to configure {product} for our setup. "
        "Could you point me to the right documentation or provide a quick example?",
        "low", "support",
    ),
    (
        "Data export stuck — {product}",
        "The data export job I triggered 2 hours ago is still showing 'processing'. "
        "Job ID: {id}. Is there a timeout or failure I should know about?",
        "medium", "support",
    ),
    # ── sales ─────────────────────────────────────────────────────────────
    (
        "Interested in {plan} plan — can we demo {day}?",
        "Hi, I saw your pricing page and I'm interested in the {plan} plan for a team of ~20. "
        "Would you have 30 minutes for a demo on {day}? Our budget is around ${amount}/month.",
        "medium", "sales",
    ),
    (
        "RFP for {product} — deadline {day}",
        "We're issuing an RFP for {product} solutions and would like to include your company. "
        "Deadline is {day}. Please confirm your interest and I'll send the full document.",
        "high", "sales",
    ),
    (
        "Following up on our call — {product} proposal",
        "Thanks for the great call last week! Attached is our formal proposal for {product}. "
        "We're ready to proceed pending legal review. Can we sign by {day}?",
        "high", "sales",
    ),
    (
        "Renewal quote needed — {plan} subscription",
        "Our annual {plan} subscription renews on {day}. Could you send an updated quote? "
        "We'd also like to discuss upgrading to ${amount}/month plan.",
        "medium", "sales",
    ),
    # ── internal ──────────────────────────────────────────────────────────
    (
        "Team standup notes — {day}",
        "Hi everyone, here are today's standup notes. Blockers: {name} is waiting on PR review. "
        "Upcoming: sprint review on {day}. Action items in the doc linked below.",
        "low", "internal",
    ),
    (
        "Reminder: Performance review cycle starts {day}",
        "This is a reminder that the Q{id} performance review cycle begins {day}. "
        "Please complete your self-assessment in Workday by end of week.",
        "medium", "internal",
    ),
    (
        "URGENT: Production deploy approval needed before {day}",
        "Hi {name}, the hotfix for issue #{id} is ready. We need your sign-off to deploy "
        "before {day} to avoid weekend downtime. Please review and approve ASAP.",
        "urgent", "internal",
    ),
    (
        "Action required: update your SSO credentials by {day}",
        "IT Security reminder: all employees must rotate their SSO credentials by {day}. "
        "Failure to comply will result in access being suspended automatically.",
        "high", "internal",
    ),
    (
        "Lunch & Learn: {product} deep-dive on {day}",
        "Join us for an informal Lunch & Learn about {product} on {day} at noon. "
        "Pizza provided. No preparation needed — just come curious!",
        "low", "internal",
    ),
    # ── billing ───────────────────────────────────────────────────────────
    (
        "Invoice #{id} — ${amount} due {day}",
        "Please find attached invoice #{id} for ${amount} covering {product} services "
        "for the period ending {day}. Payment is due within 30 days.",
        "medium", "billing",
    ),
    (
        "Receipt ${amount} — please update records",
        "Your payment of ${amount} for order #{id} has been received. "
        "Please update your accounting records accordingly. Receipt attached.",
        "medium", "billing",
    ),
    (
        "OVERDUE: Invoice #{id} — ${amount} — immediate payment required",
        "Invoice #{id} for ${amount} is now {id} days overdue. "
        "Failure to pay within 72 hours will result in service suspension.",
        "urgent", "billing",
    ),
    (
        "Billing discrepancy on account — please review",
        "We noticed a discrepancy on your account statement for {day}. "
        "The charge of ${amount} does not match our records. Please review and confirm.",
        "high", "billing",
    ),
    (
        "Annual subscription renewal — ${amount}",
        "Your annual {plan} subscription will auto-renew on {day} for ${amount}. "
        "To make changes or cancel, visit your account settings before that date.",
        "low", "billing",
    ),
    # ── security ──────────────────────────────────────────────────────────
    (
        "ALERT: Unusual login from new location — account #{id}",
        "We detected a login to account #{id} from an unrecognised IP address at {day}. "
        "If this was not you, reset your password immediately and contact security.",
        "urgent", "security",
    ),
    (
        "Security vulnerability reported in {product}",
        "A critical CVE has been identified in {product} v{id}. "
        "Patch available. Please update all instances before {day} to prevent exploitation.",
        "urgent", "security",
    ),
    (
        "Suspicious activity detected — possible data exfiltration",
        "Our SIEM flagged unusual outbound traffic from server {id} at {day}. "
        "Possible data exfiltration in progress. Immediate investigation required.",
        "urgent", "security",
    ),
    (
        "Routine security audit — {product} access review",
        "As part of the quarterly access review, please confirm the list of users "
        "who should retain access to {product}. Deadline: {day}.",
        "medium", "security",
    ),
    (
        "Phishing attempt reported by {name}",
        "{name} has reported a phishing email that impersonates our brand. "
        "The sample has been forwarded to security@company.com. Please investigate and update filters.",
        "high", "security",
    ),
]

# ── Business-critical email templates ─────────────────────────────────────
# These emails require human sign-off regardless of their category.
# true_route is always 'human_review'.
# Each entry: (subject_template, body_template, priority, category)

_CRITICAL_EMAIL_TEMPLATES: List[Tuple[str, str, str, str]] = [
    # ── legal ──────────────────────────────────────────────────────────────
    (
        "Legal notice: breach of contract — reference #{id}",
        "Dear Sir/Madam, our client contends that your company has materially breached "
        "clause 7.3 of contract #{id}. Unless remedied within 14 days, we will commence "
        "litigation seeking damages of ${amount}. Please forward to your legal counsel immediately.",
        "urgent", "support",
    ),
    (
        "Cease and desist — intellectual property infringement",
        "This letter serves as formal notice that your {product} product infringes on "
        "our registered trademark #{id}. You are required to cease all use immediately. "
        "Failure to comply will result in legal proceedings without further notice.",
        "urgent", "internal",
    ),
    (
        "Class action lawsuit — {product} data breach notification",
        "Our firm represents {amount} individuals affected by the {product} data breach "
        "disclosed on {day}. We are filing a class action and require preservation of all "
        "relevant records. A litigation hold notice is attached.",
        "urgent", "security",
    ),
    # ── compliance / GDPR ──────────────────────────────────────────────────
    (
        "GDPR right to erasure request — customer #{id}",
        "Under Article 17 of the GDPR, I formally request erasure of all personal data "
        "your company holds on me (customer #{id}). You have 30 days to comply or face "
        "regulatory action. Please confirm receipt and provide a deletion timeline.",
        "urgent", "security",
    ),
    (
        "Regulatory audit — compliance documentation required by {day}",
        "The Financial Conduct Authority has initiated a routine compliance audit. "
        "All documentation for {product} transactions between {day} and present must be "
        "submitted by {day}. Non-compliance may result in fines up to ${amount}.",
        "urgent", "internal",
    ),
    # ── large contract / pricing ────────────────────────────────────────────
    (
        "Enterprise contract negotiation — ${amount} annual deal",
        "Following our executive discussion, we are prepared to commit to a ${amount} "
        "annual contract for {product} subject to the following non-standard terms: "
        "custom SLA, dedicated account manager, and source code escrow. "
        "Board approval required on your side before {day}.",
        "urgent", "sales",
    ),
    (
        "Pricing policy change request — affects {amount} customers",
        "The sales and finance leads have proposed a 20% price increase for the {plan} "
        "tier effective {day}. This affects approximately {amount} existing customers. "
        "Executive sign-off required before we can communicate externally.",
        "high", "billing",
    ),
    (
        "Contract renewal — non-standard terms requested — ${amount}",
        "Our legal team has reviewed the renewal proposal for ${amount} and flagged "
        "clauses 4.2 (liability cap) and 9.1 (data sovereignty) as unacceptable. "
        "Escalation to VP of Legal and CEO required before {day} deadline.",
        "urgent", "sales",
    ),
    # ── insurance / claims ─────────────────────────────────────────────────
    (
        "Insurance claim #{id} — ${amount} property damage",
        "We are filing an insurance claim (ref #{id}) for ${amount} in property damage "
        "resulting from the server room flooding on {day}. A claims adjuster must be "
        "assigned and an inspection scheduled within 5 business days.",
        "urgent", "billing",
    ),
    (
        "Workers compensation claim — {name} — incident on {day}",
        "{name} has filed a workers compensation claim following a workplace injury on {day}. "
        "HR, legal, and your insurance carrier must be notified immediately. "
        "Documentation is attached. Please do not discuss the incident with {name} directly.",
        "urgent", "internal",
    ),
    # ── policy change ──────────────────────────────────────────────────────
    (
        "Proposed change to employee data retention policy",
        "Following advice from external counsel, we are proposing to reduce employee "
        "data retention from 7 years to 3 years to align with GDPR guidelines. "
        "This requires board approval and affects all HR systems. Review meeting set for {day}.",
        "high", "internal",
    ),
    (
        "Major vendor policy change — impacts {amount} integrations",
        "{product} has announced a breaking change to their API terms of service effective {day}. "
        "This affects {amount} of our customer integrations. Legal must review the new terms "
        "and engineering must assess the migration cost before we can respond.",
        "high", "support",
    ),
]

# Fill-in pools for template variables
_NAMES    = ["Alex", "Jordan", "Sam", "Morgan", "Taylor", "Casey", "Riley", "Drew"]
_PRODUCTS = ["Dashboard", "API Gateway", "Analytics Suite", "CRM", "DataPipeline", "Authenticator"]
_PLANS    = ["Starter", "Professional", "Enterprise", "Team", "Business"]
_DAYS     = ["Monday", "Tuesday", "Wednesday", "Friday", "next Friday", "March 31", "April 15"]
_DOMAINS  = ["acme.com", "techcorp.io", "startup.co", "enterprise.net", "company.org"]
_SENDERS  = [
    "noreply@mailer.io", "newsletter@updates.co", "billing@payments.net",
    "security@alerts.com", "support@helpdesk.io", "sales@leads.co",
]


def _generate_email(
    template_idx: Optional[int] = None,
    critical: bool = False,
) -> Dict[str, Any]:
    """
    Instantiate one email template with randomised fill-in values.

    Args:
        template_idx: Index into the appropriate template list.
                      None = random selection.
        critical:     If True, draw from _CRITICAL_EMAIL_TEMPLATES and set
                      route='human_review' and is_business_critical=True.

    Returns a dict with keys:
        email_id, subject, sender, body, priority, category,
        route, is_business_critical
    """
    template_list = _CRITICAL_EMAIL_TEMPLATES if critical else _EMAIL_TEMPLATES

    if template_idx is None:
        template_idx = random.randrange(len(template_list))
    template_idx = template_idx % len(template_list)

    subject_tmpl, body_tmpl, priority, category = template_list[template_idx]

    fills = {
        "name":    random.choice(_NAMES),
        "product": random.choice(_PRODUCTS),
        "plan":    random.choice(_PLANS),
        "day":     random.choice(_DAYS),
        "amount":  str(random.randint(50, 9999)),
        "id":      str(random.randint(100, 9999)),
    }

    subject = subject_tmpl.format(**fills)
    body    = body_tmpl.format(**fills)
    sender_name   = random.choice(_NAMES)
    sender_domain = random.choice(_DOMAINS)
    sender = f"{sender_name.lower()}@{sender_domain}"

    route = "human_review" if critical else ROUTE_MAP[category]

    return {
        "email_id":            str(uuid4()),
        "subject":             subject,
        "sender":              sender,
        "body":                body,
        "priority":            priority,
        "category":            category,
        "route":               route,
        "is_business_critical": critical,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GradeResult & TriageGrader (ported directly from notebook Cell 3)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GradeResult:
    """
    Raw correctness verdict for one triage action.

    Contains NO urgency scaling, streak bonuses, or overload penalties —
    those are reward-shaping concerns handled by the environment.
    This object is independently testable without the environment.
    """

    priority_ok: bool
    category_ok: bool
    route_ok:    bool

    @property
    def n_correct(self) -> int:
        return sum([self.priority_ok, self.category_ok, self.route_ok])

    @property
    def is_perfect(self) -> bool:
        return self.n_correct == 3

    @property
    def base_score(self) -> float:
        """
        Weighted additive correctness score — the canonical formula.

        Weights: priority=1.0, category=0.5, route=0.3
        Format bonus (+0.1): priority correct AND ≥1 other field correct
        Perfect bonus (+0.2): all three correct
        """
        score = (
            1.0 * self.priority_ok
            + 0.5 * self.category_ok
            + 0.3 * self.route_ok
        )
        if self.priority_ok:
            if self.category_ok or self.route_ok:
                score += 0.1
        if self.is_perfect:
            score += 0.2
        return round(score, 4)


class TriageGrader:
    """
    Grades one triage action against ground-truth email labels.

    Standalone and independently testable:

        grader = TriageGrader()
        result = grader.grade(
            action={'priority': 'urgent', 'category': 'security', 'route': 'security_team'},
            email ={'priority': 'urgent', 'category': 'security', 'route': 'security_team'},
        )
        print(result.base_score)   # → 2.1
    """

    def grade(
        self,
        action: Dict[str, str],
        email: Dict[str, str],
    ) -> GradeResult:
        """
        Compare agent action against email ground truth.

        Args:
            action: dict with keys 'priority', 'category', 'route'
            email:  dict with keys 'priority', 'category', 'route'

        Returns:
            GradeResult with per-field correctness flags and base_score.
        """
        pred_priority = str(action.get("priority", "")).strip().lower()
        pred_category = str(action.get("category", "")).strip().lower()
        pred_route    = str(action.get("route", "")).strip().lower()

        true_priority = str(email.get("priority", "")).strip().lower()
        true_category = str(email.get("category", "")).strip().lower()
        true_route    = str(email.get("route", "")).strip().lower()

        # Validate predictions against allowed vocabularies
        priority_ok = (pred_priority == true_priority) and (pred_priority in PRIORITIES)
        category_ok = (pred_category == true_category) and (pred_category in CATEGORIES)
        route_ok    = (pred_route    == true_route)    and (pred_route    in ROUTES)

        return GradeResult(
            priority_ok=priority_ok,
            category_ok=category_ok,
            route_ok=route_ok,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class EmailTriageEnvironment(Environment):
    """
    Email Triage RL Environment.

    Each episode consists of EPISODE_LENGTH emails drawn from the synthetic
    dataset.  The agent is shown one email per step and must return a triage
    decision (priority / category / route).

    Reward shaping (per step):
        shaped_reward = base_score × urgency_multiplier
                      + streak_bonus
                      - overload_penalty

        urgency_multiplier — scales reward by true email urgency so the model
                             learns to prioritise urgent/high emails correctly.
        streak_bonus       — +0.3 when current_streak >= STREAK_THRESHOLD,
                             rewarding sustained accuracy.
        overload_penalty   — -0.5 when the true priority is urgent/high but
                             the agent classifies it as low/medium (missed
                             urgent email costs extra).

    Episode ends after EPISODE_LENGTH steps; done=True is returned on the
    final step.

    Supports concurrent WebSocket sessions (each call to __init__ creates
    independent state).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    EPISODE_LENGTH:    int   = 10    # emails per episode
    STREAK_THRESHOLD:  int   = 3     # consecutive perfects needed for bonus
    STREAK_BONUS:      float = 0.3
    OVERLOAD_PENALTY:  float = 0.5

    def __init__(self) -> None:
        self._state        = State(episode_id=str(uuid4()), step_count=0)
        self._grader       = TriageGrader()
        self._email_queue: List[Dict[str, Any]] = []
        self._current_idx: int = 0
        self._streak:      int = 0
        self._last_grade:  Optional[GradeResult] = None
        # Set to True when step() is called without a prior reset() on this
        # instance (stateless HTTP mode). In this mode each step() is a
        # self-contained single-email episode and returns done=True.
        self._stateless_http_mode: bool = False

    # ── OpenEnv interface ─────────────────────────────────────────────────

    def reset(self) -> EmailTriageObservation:
        """
        Reset environment: generate a fresh email queue and return the first email.
        """
        self._state      = State(episode_id=str(uuid4()), step_count=0)
        self._streak     = 0
        self._last_grade = None
        self._current_idx = 0

        # Sample EPISODE_LENGTH emails (spread across categories for variety)
        self._email_queue = self._sample_episode()

        first_email = self._email_queue[0]
        return self._make_observation(first_email, reward=0.0, done=False)

    def step(self, action: EmailTriageAction) -> EmailTriageObservation:  # type: ignore[override]
        """
        Grade the agent's triage decision and advance to the next email.

        In stateless HTTP mode (OpenEnv creates a fresh env instance per
        request, so _email_queue is empty when step() is first called):
          - auto-resets to populate a fresh email queue
          - returns done=True after the first email so the client episode
            terminates cleanly
          - embeds the just-graded email's ground truth under 'graded_true_*'
            keys in metadata so inference.py graders always have the right labels

        Args:
            action: EmailTriageAction with priority / category / route fields.

        Returns:
            EmailTriageObservation with the next email (or terminal obs if done).
        """
        # ── Stateless HTTP guard ──────────────────────────────────────────
        if not self._email_queue:
            self.reset()
            self._stateless_http_mode = True

        current_email = self._email_queue[self._current_idx]
        self._state.step_count += 1

        # ── Grade current action ──────────────────────────────────────────
        grade = self._grader.grade(
            action={"priority": action.priority,
                    "category": action.category,
                    "route":    action.route},
            email=current_email,
        )
        self._last_grade = grade

        # ── Update streak ─────────────────────────────────────────────────
        if grade.is_perfect:
            self._streak += 1
        else:
            self._streak = 0

        # ── Shaped reward ─────────────────────────────────────────────────
        true_priority  = current_email["priority"]
        urgency_mult   = URGENCY_BONUS.get(true_priority, 1.0)
        shaped_reward  = grade.base_score * urgency_mult

        # Streak bonus
        if self._streak >= self.STREAK_THRESHOLD:
            shaped_reward += self.STREAK_BONUS

        # Overload penalty: missed urgent/high email
        if true_priority in ("urgent", "high") and action.priority in ("low", "medium"):
            shaped_reward -= self.OVERLOAD_PENALTY

        shaped_reward = round(shaped_reward, 4)

        # ── Advance to next email ─────────────────────────────────────────
        self._current_idx += 1
        # In stateless HTTP mode each step is a complete single-email episode
        done = self._stateless_http_mode or (self._current_idx >= len(self._email_queue))

        if done:
            next_email = current_email
        else:
            next_email = self._email_queue[self._current_idx]

        return self._make_observation(
            next_email,
            reward=shaped_reward,
            done=done,
            graded_email=current_email,   # ground truth for the email just graded
        )

    @property
    def state(self) -> State:
        return self._state

    # ── Helpers ───────────────────────────────────────────────────────────

    def _sample_episode(self) -> List[Dict[str, Any]]:
        """
        Build a balanced episode:
          - At least one email from each of the 7 standard categories
          - Exactly 2 business-critical emails (guarantees critical-escalation
            grader always has signal to evaluate within every episode)
          - Total padded to EPISODE_LENGTH with random standard emails
          - Result shuffled so critical emails appear in unpredictable positions
        """
        by_category: Dict[str, List[int]] = {cat: [] for cat in CATEGORIES}
        for idx, (_, _, _, cat) in enumerate(_EMAIL_TEMPLATES):
            by_category[cat].append(idx)

        selected: List[Dict[str, Any]] = []

        # One standard email from each category (7 emails)
        for cat in CATEGORIES:
            idx = random.choice(by_category[cat])
            selected.append(_generate_email(template_idx=idx, critical=False))

        # Exactly 2 business-critical emails
        critical_indices = random.sample(
            range(len(_CRITICAL_EMAIL_TEMPLATES)), k=2
        )
        for idx in critical_indices:
            selected.append(_generate_email(template_idx=idx, critical=True))

        # Pad to EPISODE_LENGTH with random standard emails (1 more needed for 10)
        while len(selected) < self.EPISODE_LENGTH:
            selected.append(_generate_email(critical=False))

        # Shuffle so critical emails don't always appear at the end
        random.shuffle(selected)
        return selected[: self.EPISODE_LENGTH]

    def _make_observation(
        self,
        email: Dict[str, Any],
        reward: float,
        done: bool,
        graded_email: Optional[Dict[str, Any]] = None,
    ) -> EmailTriageObservation:
        """
        Construct an EmailTriageObservation from an email dict.

        Args:
            email:        The NEXT email to show the agent (or current if done).
            reward:       Shaped reward for the action just taken.
            done:         Whether the episode has ended.
            graded_email: The email that was JUST graded (current_email in step).
                          Its ground truth is embedded under 'graded_true_*' keys
                          so inference.py graders always score the right labels,
                          even in stateless HTTP mode where reset and step run on
                          different env instances.
        """
        emails_remaining = max(0, len(self._email_queue) - self._current_idx - 1)

        # Ground truth for the email just graded (for client-side graders)
        graded_meta: Dict[str, Any] = {}
        if graded_email:
            graded_meta = {
                "graded_true_priority":      graded_email["priority"],
                "graded_true_category":      graded_email["category"],
                "graded_true_route":         graded_email["route"],
                "graded_is_business_critical": graded_email.get("is_business_critical", False),
            }

        return EmailTriageObservation(
            # Current email
            email_id      = email["email_id"],
            email_subject = email["subject"],
            email_sender  = email["sender"],
            email_body    = email["body"],
            # Feedback from previous action
            last_priority_correct = self._last_grade.priority_ok if self._last_grade else None,
            last_category_correct = self._last_grade.category_ok if self._last_grade else None,
            last_route_correct    = self._last_grade.route_ok    if self._last_grade else None,
            # Episode info
            emails_remaining = emails_remaining,
            current_streak   = self._streak,
            # Standard Observation fields
            done   = done,
            reward = reward,
            metadata = {
                "step":                self._state.step_count,
                "episode_id":          self._state.episode_id,
                "streak":              self._streak,
                # Ground truth for NEXT email (agent context)
                "true_priority":       email["priority"],
                "true_category":       email["category"],
                "true_route":          email["route"],
                "is_business_critical": email.get("is_business_critical", False),
                # Ground truth for JUST-GRADED email (for client-side graders)
                **graded_meta,
            },
        )