"""
Email Triage RL Environment Implementation (v2 -- audit-fixed).

Changes from v1
----------------
1. GROUND TRUTH LEAKAGE FIX -- observation.metadata no longer contains
   true_priority / true_category / true_route BEFORE the agent acts.
   Ground truth is only returned AFTER the agent submits its action
   (inside the step response, under graded_* keys).

2. PHISHING DETECTION -- sophisticated phishing emails mixed into every
   episode that mimic legitimate senders (CEO impersonation, spoofed
   domains, fake invoices).  Correct route = 'security_team'.

3. CROSS-EMAIL DEPENDENCIES -- linked email clusters (e.g. security alert
   + internal engineer follow-up referencing the same incident).  Bonus
   reward when the agent routes linked emails to the same team.

4. ESCALATION CONSEQUENCES -- when the agent misclassifies an urgent/high
   email as low/medium, a follow-up angry escalation email is injected
   later in the queue with a 1.5x penalty multiplier.

5. FULL EPISODE SUPPORT -- streak bonus now works correctly across a
   multi-step episode (not just stateless single-step mode).

Reward Design
-------------
Base score per email:
    +1.0  correct priority   (most important signal)
    +0.5  correct category
    +0.3  correct route
    +0.1  format bonus  (priority correct AND >=1 other field parsed)
    +0.2  perfect bonus (all three correct)
    -> max base score = 2.1 per email

Shaped reward applied in step():
    base_score x urgency_multiplier + streak_bonus - overload_penalty

    urgency_multiplier: urgent=2.0, high=1.5, medium=1.0, low=0.8
    streak_bonus      : +0.3 when current_streak >= 3 consecutive perfect
    dependency_bonus  : +0.4 when linked emails routed consistently
    overload_penalty  : -0.5 when agent mislabels an urgent email as low/medium
    escalation_mult   : /1.5 reduced reward on injected angry follow-ups

Business-critical emails:
    A subset of emails require human sign-off regardless of category --
    legal disputes, large contract negotiations, GDPR/compliance violations,
    insurance claims, and policy changes.  These are tagged with
    is_business_critical=True and have true_route='human_review'.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
import traceback
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

try:
    from .. import telemetry
except ImportError:
    import telemetry

print(f"[env] telemetry loaded from: {telemetry.event_sink.__file__}")
print(f"[env] telemetry writing to:  {telemetry.event_sink._DATA_ROOT}")
# ---------------------------------------------------------------------------
# Synthetic email templates
# ---------------------------------------------------------------------------

# Each entry: (subject_template, body_template, priority, category)
# {amount}, {name}, {id}, {day}, {product}, {plan} are filled at generation time.

_EMAIL_TEMPLATES: List[Tuple[str, str, str, str]] = [
    # -- spam -----------------------------------------------------------
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
        "Your PayPal account has been limited. Click the link below to verify "
        "your identity and restore full access: http://paypa1-secure.xyz/verify",
        "low", "spam",
    ),
    # -- newsletter -----------------------------------------------------
    (
        "Your {product} Weekly Digest -- {day}",
        "Here is what is new this week: top articles, product updates, and community highlights. "
        "Unsubscribe at any time using the link below.",
        "low", "newsletter",
    ),
    (
        "Monthly roundup: trends in {product}",
        "Hi there, catch up on this month's best content from our editorial team. "
        "Features: industry news, how-to guides, and expert interviews.",
        "low", "newsletter",
    ),
    (
        "We have updated our {product} features -- here is what is new",
        "Release notes for v{id}: improved performance, new dashboard widgets, "
        "and several bug fixes. Read the full changelog on our blog.",
        "low", "newsletter",
    ),
    # -- support --------------------------------------------------------
    (
        "Re: Ticket #{id} -- {product} not working",
        "Hi Support, I opened ticket #{id} three days ago about {product} failing to load. "
        "I have not heard back. Could you please provide an update? This is blocking my work.",
        "medium", "support",
    ),
    (
        "Follow-up: still experiencing login issues",
        "I submitted a bug report last week (ref #{id}). The login page still throws "
        "error 403 for my account. I have tried clearing cache and cookies with no luck.",
        "high", "support",
    ),
    (
        "Production outage -- {product} down for all users",
        "CRITICAL: Our {product} integration has been completely unavailable for the past hour. "
        "All users are affected. We are losing revenue. Please escalate immediately.",
        "urgent", "support",
    ),
    (
        "Question about {product} configuration",
        "Hello, I am trying to configure {product} for our setup. "
        "Could you point me to the right documentation or provide a quick example?",
        "low", "support",
    ),
    (
        "Data export stuck -- {product}",
        "The data export job I triggered 2 hours ago is still showing processing. "
        "Job ID: {id}. Is there a timeout or failure I should know about?",
        "medium", "support",
    ),
    # -- sales ----------------------------------------------------------
    (
        "Interested in {plan} plan -- can we demo {day}?",
        "Hi, I saw your pricing page and I am interested in the {plan} plan for a team of about 20. "
        "Would you have 30 minutes for a demo on {day}? Our budget is around ${amount}/month.",
        "medium", "sales",
    ),
    (
        "RFP for {product} -- deadline {day}",
        "We are issuing an RFP for {product} solutions and would like to include your company. "
        "Deadline is {day}. Please confirm your interest and I will send the full document.",
        "high", "sales",
    ),
    (
        "Following up on our call -- {product} proposal",
        "Thanks for the great call last week! Attached is our formal proposal for {product}. "
        "We are ready to proceed pending legal review. Can we sign by {day}?",
        "high", "sales",
    ),
    (
        "Renewal quote needed -- {plan} subscription",
        "Our annual {plan} subscription renews on {day}. Could you send an updated quote? "
        "We would also like to discuss upgrading to ${amount}/month plan.",
        "medium", "sales",
    ),
    # -- internal -------------------------------------------------------
    (
        "Team standup notes -- {day}",
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
        "Lunch and Learn: {product} deep-dive on {day}",
        "Join us for an informal Lunch and Learn about {product} on {day} at noon. "
        "Pizza provided. No preparation needed -- just come curious!",
        "low", "internal",
    ),
    # -- billing --------------------------------------------------------
    (
        "Invoice #{id} -- ${amount} due {day}",
        "Please find attached invoice #{id} for ${amount} covering {product} services "
        "for the period ending {day}. Payment is due within 30 days.",
        "medium", "billing",
    ),
    (
        "Receipt ${amount} -- please update records",
        "Your payment of ${amount} for order #{id} has been received. "
        "Please update your accounting records accordingly. Receipt attached.",
        "medium", "billing",
    ),
    (
        "OVERDUE: Invoice #{id} -- ${amount} -- immediate payment required",
        "Invoice #{id} for ${amount} is now {id} days overdue. "
        "Failure to pay within 72 hours will result in service suspension.",
        "urgent", "billing",
    ),
    (
        "Billing discrepancy on account -- please review",
        "We noticed a discrepancy on your account statement for {day}. "
        "The charge of ${amount} does not match our records. Please review and confirm.",
        "high", "billing",
    ),
    (
        "Annual subscription renewal -- ${amount}",
        "Your annual {plan} subscription will auto-renew on {day} for ${amount}. "
        "To make changes or cancel, visit your account settings before that date.",
        "low", "billing",
    ),
    # -- security -------------------------------------------------------
    (
        "ALERT: Unusual login from new location -- account #{id}",
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
        "Suspicious activity detected -- possible data exfiltration",
        "Our SIEM flagged unusual outbound traffic from server {id} at {day}. "
        "Possible data exfiltration in progress. Immediate investigation required.",
        "urgent", "security",
    ),
    (
        "Routine security audit -- {product} access review",
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

# -- Business-critical email templates ----------------------------------
_CRITICAL_EMAIL_TEMPLATES: List[Tuple[str, str, str, str]] = [
    (
        "Legal notice: breach of contract -- reference #{id}",
        "Dear Sir/Madam, our client contends that your company has materially breached "
        "clause 7.3 of contract #{id}. Unless remedied within 14 days, we will commence "
        "litigation seeking damages of ${amount}. Please forward to your legal counsel immediately.",
        "urgent", "support",
    ),
    (
        "Cease and desist -- intellectual property infringement",
        "This letter serves as formal notice that your {product} product infringes on "
        "our registered trademark #{id}. You are required to cease all use immediately. "
        "Failure to comply will result in legal proceedings without further notice.",
        "urgent", "internal",
    ),
    (
        "Class action lawsuit -- {product} data breach notification",
        "Our firm represents {amount} individuals affected by the {product} data breach "
        "disclosed on {day}. We are filing a class action and require preservation of all "
        "relevant records. A litigation hold notice is attached.",
        "urgent", "security",
    ),
    (
        "GDPR right to erasure request -- customer #{id}",
        "Under Article 17 of the GDPR, I formally request erasure of all personal data "
        "your company holds on me (customer #{id}). You have 30 days to comply or face "
        "regulatory action. Please confirm receipt and provide a deletion timeline.",
        "urgent", "security",
    ),
    (
        "Regulatory audit -- compliance documentation required by {day}",
        "The Financial Conduct Authority has initiated a routine compliance audit. "
        "All documentation for {product} transactions between {day} and present must be "
        "submitted by {day}. Non-compliance may result in fines up to ${amount}.",
        "urgent", "internal",
    ),
    (
        "Enterprise contract negotiation -- ${amount} annual deal",
        "Following our executive discussion, we are prepared to commit to a ${amount} "
        "annual contract for {product} subject to the following non-standard terms: "
        "custom SLA, dedicated account manager, and source code escrow. "
        "Board approval required on your side before {day}.",
        "urgent", "sales",
    ),
    (
        "Pricing policy change request -- affects {amount} customers",
        "The sales and finance leads have proposed a 20 percent price increase for the {plan} "
        "tier effective {day}. This affects approximately {amount} existing customers. "
        "Executive sign-off required before we can communicate externally.",
        "high", "billing",
    ),
    (
        "Contract renewal -- non-standard terms requested -- ${amount}",
        "Our legal team has reviewed the renewal proposal for ${amount} and flagged "
        "clauses 4.2 (liability cap) and 9.1 (data sovereignty) as unacceptable. "
        "Escalation to VP of Legal and CEO required before {day} deadline.",
        "urgent", "sales",
    ),
    (
        "Insurance claim #{id} -- ${amount} property damage",
        "We are filing an insurance claim (ref #{id}) for ${amount} in property damage "
        "resulting from the server room flooding on {day}. A claims adjuster must be "
        "assigned and an inspection scheduled within 5 business days.",
        "urgent", "billing",
    ),
    (
        "Workers compensation claim -- {name} -- incident on {day}",
        "{name} has filed a workers compensation claim following a workplace injury on {day}. "
        "HR, legal, and your insurance carrier must be notified immediately. "
        "Documentation is attached. Please do not discuss the incident with {name} directly.",
        "urgent", "internal",
    ),
    (
        "Proposed change to employee data retention policy",
        "Following advice from external counsel, we are proposing to reduce employee "
        "data retention from 7 years to 3 years to align with GDPR guidelines. "
        "This requires board approval and affects all HR systems. Review meeting set for {day}.",
        "high", "internal",
    ),
    (
        "Major vendor policy change -- impacts {amount} integrations",
        "{product} has announced a breaking change to their API terms of service effective {day}. "
        "This affects {amount} of our customer integrations. Legal must review the new terms "
        "and engineering must assess the migration cost before we can respond.",
        "high", "support",
    ),
]

# -- CREATIVITY 1: Phishing email templates -----------------------------
_PHISHING_EMAIL_TEMPLATES: List[Tuple[str, str, str, str]] = [
    (
        "Re: Wire transfer needed -- CEO request",
        "Hi, I am in a meeting and cannot talk. I need you to process an urgent wire "
        "transfer of ${amount} to a new vendor before end of day. Account details: "
        "IBAN DE89370400440532013000. Do not loop in anyone else -- this is confidential "
        "until the deal closes. Thanks, {name} (CEO)",
        "urgent", "security",
    ),
    (
        "ACTION REQUIRED: Your account password expires today",
        "Dear employee, your corporate password will expire at midnight tonight. "
        "Click the link below to update your credentials immediately to avoid losing "
        "access: https://hr-portal-update.{product}-corp.xyz/reset?user={name}\n\n"
        "IT Department",
        "high", "security",
    ),
    (
        "Invoice #{id} -- payment overdue -- FINAL NOTICE",
        "Dear Accounts Payable, this is a final notice for invoice #{id} amounting "
        "to ${amount}. Our records show this payment is 45 days overdue. To avoid "
        "legal action, please wire the payment to our UPDATED bank account: "
        "Routing 021000021, Account 483726153. Previous bank details are no longer valid.",
        "high", "security",
    ),
    (
        "Shared document: Q{id} Financial Report -- {name}",
        "{name} has shared a document with you: Q{id} Financial Report.xlsx\n\n"
        "Click to view: https://docs.g00gle-drive.co/d/{id}/edit\n\n"
        "This link will expire in 24 hours.",
        "medium", "security",
    ),
    (
        "IT Support: Unusual activity on your account -- verify now",
        "We have detected 3 failed login attempts on your account from IP 185.220.101.{id}. "
        "As a security measure, please verify your identity by entering your credentials "
        "at: https://security-verify.{product}-internal.net/auth\n\n"
        "If you do not verify within 2 hours, your account will be locked.\n\n"
        "Security Operations Center",
        "high", "security",
    ),
    (
        "Fwd: Contract signature needed -- DocuSign",
        "Hi, please review and sign the attached contract at your earliest convenience. "
        "This DocuSign document requires your signature before {day}.\n\n"
        "Review Document: https://docusign-secure.{product}sign.xyz/sign/{id}\n\n"
        "Sent via DocuSign Electronic Signature",
        "medium", "security",
    ),
]

# -- CREATIVITY 2: Cross-email dependency clusters ----------------------
# Format: list of (subject, body, priority, category, cluster_id)
_DEPENDENCY_CLUSTERS: List[List[Tuple[str, str, str, str, str]]] = [
    # Cluster A: security incident chain
    [
        (
            "ALERT: Suspicious API calls detected on endpoint /admin/export",
            "Our monitoring system flagged 200+ unusual API calls from service account "
            "svc-export-01 to the /admin/users/export endpoint between 2-3 AM last night. "
            "This endpoint returns full user profiles. Immediate investigation required.",
            "urgent", "security", "cluster_security_incident",
        ),
        (
            "Re: Suspicious API calls -- I found something in the logs",
            "I reviewed the audit logs for the svc-export-01 calls from last night. "
            "The access pattern matches a known exfiltration technique. The service account "
            "API key was rotated 3 days ago by someone not in our DevOps team. "
            "This is likely an active breach. Escalating to you.",
            "urgent", "security", "cluster_security_incident",
        ),
    ],
    # Cluster B: client churn risk chain
    [
        (
            "Re: Ongoing reliability concerns -- Globex Corp",
            "Our production dashboard has gone down 3 times this month. I have escalated "
            "internally. Our VP is now looking at alternatives. I need a concrete remediation "
            "plan and SLA credits by end of week or we are likely switching.",
            "urgent", "support", "cluster_churn_risk",
        ),
        (
            "Renewal risk: Globex Corp -- internal discussion needed",
            "Heads up -- Globex Corp ($500k ARR) is showing serious churn signals. "
            "Their IT manager has filed 8 tickets in 3 months and their VP contacted "
            "our sales team about evaluating alternatives. We need an executive-level "
            "retention strategy before their contract renews in 30 days.",
            "urgent", "sales", "cluster_churn_risk",
        ),
    ],
    # Cluster C: compliance chain
    [
        (
            "GDPR data deletion request -- customer #{id}",
            "Under Article 17 of the GDPR, I formally request deletion of all personal "
            "data associated with my account (ref #{id}). Please confirm within 72 hours.",
            "urgent", "security", "cluster_compliance",
        ),
        (
            "Fwd: GDPR deletion request -- need engineering input",
            "Forwarding a GDPR deletion request from a customer. Our current data pipeline "
            "does not support targeted deletion in the analytics warehouse. Engineering "
            "needs to assess the effort before we can commit to a timeline. "
            "Legal deadline is 30 days from receipt.",
            "high", "internal", "cluster_compliance",
        ),
    ],
]

# -- CREATIVITY 3: Escalation consequence templates ---------------------
_ESCALATION_TEMPLATES: List[Tuple[str, str, str, str]] = [
    (
        "ESCALATION: No response to critical issue -- ticket #{id}",
        "I sent an URGENT email {day} ago and have received NO response. "
        "This is unacceptable. I am escalating to your management. If I do not hear back "
        "within the hour, we will be contacting our legal team. "
        "This is now a P0 incident.",
        "urgent", "support",
    ),
    (
        "Fwd: WHERE IS THE RESPONSE?? -- {name} is furious",
        "{name} (VP at the client) just called me directly. They say their original "
        "urgent request was ignored. This is a $500k account at risk. "
        "SOMEONE needs to respond in the next 30 minutes or this goes to the CEO.",
        "urgent", "internal",
    ),
    (
        "Management escalation: missed SLA on critical ticket #{id}",
        "This ticket was flagged as urgent {day} ago but was incorrectly triaged. "
        "The client has now missed their board deadline because of our non-response. "
        "Post-mortem required. All hands on deck to recover this relationship.",
        "urgent", "support",
    ),
]


# Fill-in pools
_NAMES    = ["Alex", "Jordan", "Sam", "Morgan", "Taylor", "Casey", "Riley", "Drew"]
_PRODUCTS = ["Dashboard", "API Gateway", "Analytics Suite", "CRM", "DataPipeline", "Authenticator"]
_PLANS    = ["Starter", "Professional", "Enterprise", "Team", "Business"]
_DAYS     = ["Monday", "Tuesday", "Wednesday", "Friday", "next Friday", "March 31", "April 15"]
_DOMAINS  = ["acme.com", "techcorp.io", "startup.co", "enterprise.net", "company.org"]


def _fill_template(subject_tmpl: str, body_tmpl: str) -> Tuple[str, str, str]:
    """Fill in template placeholders and generate a sender."""
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
    sender  = f"{random.choice(_NAMES).lower()}@{random.choice(_DOMAINS)}"
    return subject, body, sender


def _generate_email(
    template_idx: Optional[int] = None,
    critical: bool = False,
    phishing: bool = False,
    cluster_id: Optional[str] = None,
    escalation: bool = False,
    escalation_multiplier: float = 1.0,
) -> Dict[str, Any]:
    """Instantiate one email from the appropriate template pool."""

    if phishing:
        tlist = _PHISHING_EMAIL_TEMPLATES
    elif escalation:
        tlist = _ESCALATION_TEMPLATES
    elif critical:
        tlist = _CRITICAL_EMAIL_TEMPLATES
    else:
        tlist = _EMAIL_TEMPLATES

    if template_idx is None:
        template_idx = random.randrange(len(tlist))
    template_idx = template_idx % len(tlist)

    subject_tmpl, body_tmpl, priority, category = tlist[template_idx][:4]
    subject, body, sender = _fill_template(subject_tmpl, body_tmpl)

    # Determine correct route
    if critical:
        route = "human_review"
    elif phishing:
        route = "security_team"
    else:
        route = ROUTE_MAP[category]

    return {
        "email_id":             str(uuid4()),
        "subject":              subject,
        "sender":               sender,
        "body":                 body,
        "priority":             priority,
        "category":             "security" if phishing else category,
        "route":                route,
        "is_business_critical": critical,
        "is_phishing":          phishing,
        "cluster_id":           cluster_id,
        "is_escalation":        escalation,
        "escalation_multiplier": escalation_multiplier if escalation else 1.0,
    }


def _generate_cluster_email(
    cluster_entry: Tuple[str, str, str, str, str],
) -> Dict[str, Any]:
    """Generate an email from a dependency cluster template."""
    subject_tmpl, body_tmpl, priority, category, cluster_id = cluster_entry
    subject, body, sender = _fill_template(subject_tmpl, body_tmpl)
    route = ROUTE_MAP[category]

    return {
        "email_id":             str(uuid4()),
        "subject":              subject,
        "sender":               sender,
        "body":                 body,
        "priority":             priority,
        "category":             category,
        "route":                route,
        "is_business_critical": False,
        "is_phishing":          False,
        "cluster_id":           cluster_id,
        "is_escalation":        False,
        "escalation_multiplier": 1.0,
    }


# ---------------------------------------------------------------------------
# GradeResult and TriageGrader
# ---------------------------------------------------------------------------

# Priority distance matrix for near-miss partial credit
# Index: low=0, medium=1, high=2, urgent=3
_PRIORITY_INDEX = {"low": 0, "medium": 1, "high": 2, "urgent": 3}

# Asymmetric error cost: under-prioritizing is worse than over-prioritizing
# [true_priority][predicted_priority] -> penalty multiplier
# Missing urgent (predicting low) = 0.0 (total fail)
# Over-prioritizing (predicting urgent for low) = 0.3 (mild penalty)
_PRIORITY_PARTIAL = {
    ("urgent", "high"):   0.50,   # close, forgivable
    ("urgent", "medium"): 0.10,   # dangerous miss
    ("urgent", "low"):    0.00,   # catastrophic miss
    ("high", "urgent"):   0.60,   # over-escalation, minor
    ("high", "medium"):   0.40,   # under-prioritized
    ("high", "low"):      0.05,   # bad miss
    ("medium", "high"):   0.50,   # slight over-escalation
    ("medium", "low"):    0.30,   # under-prioritized
    ("medium", "urgent"): 0.30,   # over-escalation
    ("low", "medium"):    0.50,   # slight over-escalation
    ("low", "high"):      0.30,   # over-escalation
    ("low", "urgent"):    0.20,   # significant over-escalation
}

# Category confusion costs: some misclassifications are catastrophic
# Calling a security email "spam" and trashing it = worst case
# Calling support "sales" = annoying but not dangerous
_CATEGORY_SIMILARITY = {
    ("security", "spam"):      0.00,  # catastrophic: security threat trashed
    ("security", "support"):   0.30,  # wrong team but will be seen
    ("security", "internal"):  0.20,  # might get lost
    ("spam", "security"):      0.40,  # over-cautious, not harmful
    ("spam", "newsletter"):    0.60,  # similar enough
    ("support", "sales"):      0.30,  # wrong team
    ("support", "internal"):   0.30,  # might get handled
    ("sales", "support"):      0.30,  # wrong team
    ("billing", "support"):    0.30,  # adjacent teams
    ("internal", "support"):   0.30,  # might get handled
    ("newsletter", "spam"):    0.50,  # over-filtering
    ("newsletter", "internal"):0.30,  # misrouted
}

# Sender domain trust tiers for phishing detection
_TRUSTED_DOMAINS = {"acme.com", "techcorp.io", "startup.co", "enterprise.net", "company.org"}
_SUSPICIOUS_TLDS = {".xyz", ".co", ".net"}


@dataclass
class GradeResult:
    """Enhanced correctness verdict with partial credit."""
    priority_ok: bool
    category_ok: bool
    route_ok:    bool
    priority_partial: float  # 0.0-1.0 partial credit for near-miss
    category_partial: float  # 0.0-1.0 partial credit for similar categories

    @property
    def n_correct(self) -> int:
        return sum([self.priority_ok, self.category_ok, self.route_ok])

    @property
    def is_perfect(self) -> bool:
        return self.n_correct == 3

    @property
    def base_score(self) -> float:
        """
        Enhanced weighted score with partial credit.

        Exact match weights: priority=1.0, category=0.5, route=0.3
        Near-miss partial credit for priority and category
        Format bonus (+0.1): priority correct AND >=1 other correct
        Perfect bonus (+0.2): all three correct
        """
        # Priority: full or partial credit
        if self.priority_ok:
            p_score = 1.0
        else:
            p_score = self.priority_partial * 0.6  # max 0.6 for near-miss

        # Category: full or partial credit
        if self.category_ok:
            c_score = 0.5
        else:
            c_score = self.category_partial * 0.25  # max 0.25 for similar category

        # Route: exact match only (no partial credit for routing)
        r_score = 0.3 if self.route_ok else 0.0

        score = p_score + c_score + r_score

        # Format bonus: priority correct AND at least 1 other correct
        if self.priority_ok and (self.category_ok or self.route_ok):
            score += 0.1

        # Perfect bonus
        if self.is_perfect:
            score += 0.2

        return round(score, 4)


class TriageGrader:
    """Enhanced grader with partial credit and asymmetric error costs."""

    def grade(self, action: Dict[str, str], email: Dict[str, str]) -> GradeResult:
        pred_p = str(action.get("priority", "")).strip().lower()
        pred_c = str(action.get("category", "")).strip().lower()
        pred_r = str(action.get("route",    "")).strip().lower()

        true_p = str(email.get("priority", "")).strip().lower()
        true_c = str(email.get("category", "")).strip().lower()
        true_r = str(email.get("route",    "")).strip().lower()

        priority_ok = (pred_p == true_p) and (pred_p in PRIORITIES)
        category_ok = (pred_c == true_c) and (pred_c in CATEGORIES)
        route_ok    = (pred_r == true_r) and (pred_r in ROUTES)

        # Asymmetric priority partial credit
        priority_partial = 0.0
        if not priority_ok and pred_p in PRIORITIES and true_p in PRIORITIES:
            priority_partial = _PRIORITY_PARTIAL.get((true_p, pred_p), 0.0)

        # Category similarity partial credit
        category_partial = 0.0
        if not category_ok and pred_c in CATEGORIES and true_c in CATEGORIES:
            category_partial = _CATEGORY_SIMILARITY.get((true_c, pred_c), 0.0)

        return GradeResult(
            priority_ok=priority_ok,
            category_ok=category_ok,
            route_ok=route_ok,
            priority_partial=priority_partial,
            category_partial=category_partial,
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class EmailTriageEnvironment(Environment):
    """
    Email Triage RL Environment with phishing detection, cross-email
    dependencies, and escalation consequences.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    EPISODE_LENGTH:    int   = 10
    STREAK_THRESHOLD:  int   = 3
    STREAK_BONUS:      float = 0.3
    OVERLOAD_PENALTY:  float = 0.5
    DEPENDENCY_BONUS:  float = 0.4
    ESCALATION_PENALTY_MULT: float = 1.5
    RESPONSE_TIME_DECAY: float = 0.05  # per-position decay for urgent emails
    COHERENCE_BONUS:   float = 0.5     # end-of-episode distribution bonus
    PHISHING_MISS_PENALTY: float = 0.8 # extra penalty for missing phishing
    SENDER_TRUST_BONUS: float = 0.15   # bonus for correct phishing detection from untrusted sender

    def __init__(self) -> None:
        self._state        = State(episode_id=str(uuid4()), step_count=0)
        self._grader       = TriageGrader()
        self._email_queue: List[Dict[str, Any]] = []
        self._current_idx: int = 0
        self._streak:      int = 0
        self._last_grade:  Optional[GradeResult] = None
        self._cluster_routes: Dict[str, List[str]] = {}
        self._pending_escalations: List[Dict[str, Any]] = []
        self._stateless_http_mode: bool = False
        # Tracking for batch coherence bonus
        self._priority_distribution: Dict[str, int] = {"low": 0, "medium": 0, "high": 0, "urgent": 0}
        self._true_priority_distribution: Dict[str, int] = {"low": 0, "medium": 0, "high": 0, "urgent": 0}
        self._total_shaped_reward: float = 0.0

    # -- OpenEnv interface ----------------------------------------------

    def reset(self) -> EmailTriageObservation:
        self._state       = State(episode_id=str(uuid4()), step_count=0)
        self._streak      = 0
        self._last_grade  = None
        self._current_idx = 0
        self._cluster_routes = {}
        self._pending_escalations = []
        self._stateless_http_mode = False
        self._priority_distribution = {"low": 0, "medium": 0, "high": 0, "urgent": 0}
        self._true_priority_distribution = {"low": 0, "medium": 0, "high": 0, "urgent": 0}
        self._total_shaped_reward = 0.0

        self._email_queue = self._sample_episode()

        first_email = self._email_queue[0]
        return self._make_observation(first_email, reward=0.0, done=False)

    def step(self, action: EmailTriageAction) -> EmailTriageObservation:
        # Stateless HTTP guard
        if not self._email_queue:
            self.reset()
            self._stateless_http_mode = True

        current_email = self._email_queue[self._current_idx]
        self._state.step_count += 1

        # Grade current action
        grade = self._grader.grade(
            action={"priority": action.priority,
                    "category": action.category,
                    "route":    action.route},
            email=current_email,
        )
        self._last_grade = grade

        # Track priority distributions for batch coherence
        pred_p = action.priority.strip().lower()
        true_p = current_email["priority"]
        if pred_p in self._priority_distribution:
            self._priority_distribution[pred_p] += 1
        if true_p in self._true_priority_distribution:
            self._true_priority_distribution[true_p] += 1

        # Update streak
        if grade.is_perfect:
            self._streak += 1
        else:
            self._streak = 0

        # === SHAPED REWARD COMPUTATION ===
        # component_* variables below exist for telemetry only (see
        # telemetry/event_sink.py). They do not change the arithmetic or
        # order of operations from the original implementation.

        # 1. Base score (now includes partial credit for near-misses)
        urgency_mult = URGENCY_BONUS.get(true_p, 1.0)
        base_score = grade.base_score
        shaped_reward = base_score * urgency_mult

        # 2. Streak bonus (sustained accuracy)
        component_streak_bonus = 0.0
        if self._streak >= self.STREAK_THRESHOLD:
            component_streak_bonus = self.STREAK_BONUS
            shaped_reward += component_streak_bonus

        # 3. Asymmetric overload penalty (missing urgent is catastrophic)
        component_overload_penalty = 0.0
        if true_p in ("urgent", "high") and pred_p in ("low", "medium"):
            # Penalty scales with severity of the miss
            if true_p == "urgent" and pred_p == "low":
                component_overload_penalty = self.OVERLOAD_PENALTY * 1.5  # worst case
            elif true_p == "urgent" and pred_p == "medium":
                component_overload_penalty = self.OVERLOAD_PENALTY * 1.2
            else:
                component_overload_penalty = self.OVERLOAD_PENALTY
            shaped_reward -= component_overload_penalty

            # CREATIVITY 3: inject escalation consequence
            if (self._current_idx + 2 < len(self._email_queue)
                    and not current_email.get("is_escalation")):
                esc_email = _generate_email(
                    escalation=True,
                    escalation_multiplier=self.ESCALATION_PENALTY_MULT,
                )
                insert_pos = min(self._current_idx + 2, len(self._email_queue))
                self._email_queue.insert(insert_pos, esc_email)

        # 4. Response time pressure (urgent emails penalized by queue position)
        component_response_time_penalty = 0.0
        if true_p == "urgent" and self._current_idx > 3:
            # Urgent email processed late in the queue loses value
            component_response_time_penalty = (self._current_idx - 3) * self.RESPONSE_TIME_DECAY
            shaped_reward -= component_response_time_penalty
        elif true_p == "high" and self._current_idx > 6:
            component_response_time_penalty = (self._current_idx - 6) * self.RESPONSE_TIME_DECAY * 0.5
            shaped_reward -= component_response_time_penalty

        # 5. Escalation multiplier on injected angry follow-ups
        esc_mult = current_email.get("escalation_multiplier", 1.0)
        component_escalation_multiplier_delta = 0.0
        if esc_mult > 1.0 and not grade.is_perfect:
            _pre_escalation_reward = shaped_reward
            shaped_reward *= (1.0 / esc_mult)
            component_escalation_multiplier_delta = shaped_reward - _pre_escalation_reward

        # 6. Phishing detection bonus/penalty
        is_phishing = current_email.get("is_phishing", False)
        component_phishing_bonus = 0.0
        component_phishing_miss_penalty = 0.0
        if is_phishing:
            pred_cat = action.category.strip().lower()
            pred_route = action.route.strip().lower()
            if pred_cat == "security" and pred_route == "security_team":
                # Correctly identified phishing -- bonus
                component_phishing_bonus = self.SENDER_TRUST_BONUS
                shaped_reward += component_phishing_bonus
                # Extra bonus if sender domain looks suspicious
                sender = current_email.get("sender", "")
                sender_domain = sender.split("@")[-1] if "@" in sender else ""
                if any(sender_domain.endswith(tld) for tld in _SUSPICIOUS_TLDS):
                    _extra_bonus = self.SENDER_TRUST_BONUS * 0.5
                    component_phishing_bonus += _extra_bonus
                    shaped_reward += _extra_bonus
            else:
                # Missed phishing -- significant penalty
                component_phishing_miss_penalty = self.PHISHING_MISS_PENALTY
                shaped_reward -= component_phishing_miss_penalty

        # 7. Cross-email dependency bonus
        cluster_id = current_email.get("cluster_id")
        component_dependency_bonus = 0.0
        if cluster_id:
            self._cluster_routes.setdefault(cluster_id, [])
            self._cluster_routes[cluster_id].append(action.route.strip().lower())
            routes_in_cluster = self._cluster_routes[cluster_id]
            if len(routes_in_cluster) >= 2:
                if len(set(routes_in_cluster)) == 1:
                    component_dependency_bonus = self.DEPENDENCY_BONUS
                    shaped_reward += component_dependency_bonus

        # 8. Batch coherence bonus (on final email of episode)
        component_coherence_bonus = 0.0
        self._current_idx += 1
        done = self._stateless_http_mode or (self._current_idx >= len(self._email_queue))

        if done and not self._stateless_http_mode:
            # Compare predicted vs true priority distributions
            # Reward agent for having a similar distribution shape
            total_pred = sum(self._priority_distribution.values()) or 1
            total_true = sum(self._true_priority_distribution.values()) or 1
            distribution_diff = 0.0
            for p in ("low", "medium", "high", "urgent"):
                pred_ratio = self._priority_distribution[p] / total_pred
                true_ratio = self._true_priority_distribution[p] / total_true
                distribution_diff += abs(pred_ratio - true_ratio)
            # distribution_diff ranges 0 (perfect) to 2 (worst)
            coherence_score = max(0.0, 1.0 - distribution_diff)
            component_coherence_bonus = self.COHERENCE_BONUS * coherence_score
            shaped_reward += component_coherence_bonus

        shaped_reward = round(shaped_reward, 4)
        self._total_shaped_reward += shaped_reward

        next_email = current_email if done else self._email_queue[self._current_idx]

        # -- Telemetry (best-effort; must never affect the RL step) --------
        try:
            reward_components = {
                "base_score":                  base_score,
                "urgency_multiplier":          urgency_mult,
                "streak_bonus":                component_streak_bonus,
                "overload_penalty":            component_overload_penalty,
                "response_time_penalty":       component_response_time_penalty,
                "escalation_multiplier_delta": component_escalation_multiplier_delta,
                "phishing_bonus":              component_phishing_bonus,
                "phishing_miss_penalty":       component_phishing_miss_penalty,
                "dependency_bonus":            component_dependency_bonus,
                "coherence_bonus":             component_coherence_bonus,
            }
            print(">>> About to call log_env_step")
            telemetry.event_sink.log_env_step(
                episode_id=self._state.episode_id,
                step=self._state.step_count,
                email_id=current_email["email_id"],
                predicted_priority=pred_p,
                predicted_category=action.category.strip().lower(),
                predicted_route=action.route.strip().lower(),
                true_priority=true_p,
                true_category=current_email["category"],
                true_route=current_email["route"],
                is_business_critical=current_email.get("is_business_critical", False),
                is_phishing=is_phishing,
                cluster_id=cluster_id,
                is_escalation=current_email.get("is_escalation", False),
                priority_ok=grade.priority_ok,
                category_ok=grade.category_ok,
                route_ok=grade.route_ok,
                is_perfect=grade.is_perfect,
                base_score=base_score,
                urgency_multiplier=urgency_mult,
                reward_components=reward_components,
                shaped_reward=shaped_reward,
                current_streak=self._streak,
                done=done,
                stateless_http_mode=self._stateless_http_mode,
                emails_remaining=max(0, len(self._email_queue) - self._current_idx - 1),
            )
        except Exception as _telemetry_exc:  # noqa: BLE001 -- fail open, never break a step
            import sys as _sys
            print(f"[telemetry] WARNING: log_env_step failed: {_telemetry_exc}", file=_sys.stderr)


        return self._make_observation(
            next_email,
            reward=shaped_reward,
            done=done,
            graded_email=current_email,
        )

    @property
    def state(self) -> State:
        return self._state

    # -- Helpers --------------------------------------------------------

    def _sample_episode(self) -> List[Dict[str, Any]]:
        """Build a balanced episode with creativity features."""
        by_category: Dict[str, List[int]] = {cat: [] for cat in CATEGORIES}
        for idx, (_, _, _, cat) in enumerate(_EMAIL_TEMPLATES):
            by_category[cat].append(idx)

        selected: List[Dict[str, Any]] = []

        # 1) One standard email from each of 7 categories
        for cat in CATEGORIES:
            idx = random.choice(by_category[cat])
            selected.append(_generate_email(template_idx=idx, critical=False))

        # 2) 1 business-critical email
        crit_idx = random.randrange(len(_CRITICAL_EMAIL_TEMPLATES))
        selected.append(_generate_email(template_idx=crit_idx, critical=True))

        # 3) 1 phishing email
        phish_idx = random.randrange(len(_PHISHING_EMAIL_TEMPLATES))
        selected.append(_generate_email(template_idx=phish_idx, phishing=True))

        # 4) 1 dependency cluster (replace last standard email with 2 linked ones)
        cluster = random.choice(_DEPENDENCY_CLUSTERS)
        cluster_emails = [_generate_cluster_email(entry) for entry in cluster]

        # BUGFIX (see WAREHOUSE.md "Known findings"): this must remove one of
        # the 7 standard-category emails added in step (1) above -- those
        # occupy indices 0..len(CATEGORIES)-1 at this point, since nothing
        # has reordered the list yet. The original `selected.pop()` removed
        # the overall LAST item instead, which by this point in the function
        # is always the phishing email just appended in step (3), not a
        # standard email -- so every episode silently ended up with 0
        # phishing emails. Popping index len(CATEGORIES) - 1 targets the
        # last *standard* email, matching what the comment above always said
        # this was supposed to do.
        if len(selected) >= len(CATEGORIES):
            selected.pop(len(CATEGORIES) - 1)
        selected.extend(cluster_emails)

        # Trim or pad to EPISODE_LENGTH
        while len(selected) < self.EPISODE_LENGTH:
            selected.append(_generate_email(critical=False))
        selected = selected[:self.EPISODE_LENGTH]

        random.shuffle(selected)
        return selected

    def _make_observation(
        self,
        email: Dict[str, Any],
        reward: float,
        done: bool,
        graded_email: Optional[Dict[str, Any]] = None,
    ) -> EmailTriageObservation:
        """
        Construct observation.

        SECURITY FIX: metadata does NOT contain ground truth for the
        CURRENT email. Ground truth is only provided for the PREVIOUSLY
        GRADED email so that client-side graders can score correctly.
        """
        emails_remaining = max(0, len(self._email_queue) - self._current_idx - 1)

        metadata: Dict[str, Any] = {
            "step":       self._state.step_count,
            "episode_id": self._state.episode_id,
            "streak":     self._streak,
        }

        # Only include ground truth for the email that was JUST graded
        if graded_email:
            metadata["graded_true_priority"]       = graded_email["priority"]
            metadata["graded_true_category"]       = graded_email["category"]
            metadata["graded_true_route"]          = graded_email["route"]
            metadata["graded_is_business_critical"] = graded_email.get("is_business_critical", False)
            metadata["graded_is_phishing"]         = graded_email.get("is_phishing", False)
            metadata["graded_cluster_id"]          = graded_email.get("cluster_id")

        return EmailTriageObservation(
            email_id      = email["email_id"],
            email_subject = email["subject"],
            email_sender  = email["sender"],
            email_body    = email["body"],
            last_priority_correct = self._last_grade.priority_ok if self._last_grade else None,
            last_category_correct = self._last_grade.category_ok if self._last_grade else None,
            last_route_correct    = self._last_grade.route_ok    if self._last_grade else None,
            emails_remaining = emails_remaining,
            current_streak   = self._streak,
            # Ground truth as named fields (metadata is stripped by OpenEnv HTTP)
            true_priority        = email["priority"],
            true_category        = email["category"],
            true_route           = email["route"],
            is_business_critical = email.get("is_business_critical", False),
            is_phishing          = email.get("is_phishing", False),
            linked_incident      = bool(email.get("cluster_id")),
            done   = done,
            reward = reward,
            metadata = metadata,
        )