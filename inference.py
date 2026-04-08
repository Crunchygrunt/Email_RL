"""
inference.py -- Email Triage RL Environment
==========================================
MANDATORY ENVIRONMENT VARIABLES
    API_BASE_URL   The API endpoint for the LLM (must have default).
    MODEL_NAME     The model identifier to use for inference (must have default).
    HF_TOKEN       Your Hugging Face / API key (mandatory, no default).

STDOUT FORMAT -- strictly followed, no deviation:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

TASKS (4 tasks, each with its own grader, all rewards in [0.0, 1.0]):
    spam-detection           (easy)   -- binary spam vs legitimate
    priority-classification  (medium) -- exact urgency level match
    full-triage              (hard)   -- weighted score across all 3 fields
    critical-escalation      (hard)   -- business-critical -> human_review detection
"""
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


import asyncio
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import httpx
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from models import EmailTriageAction, EmailTriageObservation
except ModuleNotFoundError:
    from Email_RL.models import EmailTriageAction, EmailTriageObservation


# ---------------------------------------------------------------------------
# HTTP client for the environment
# ---------------------------------------------------------------------------

@dataclass
class _StepResult:
    observation: EmailTriageObservation
    reward:      float
    done:        bool


class EmailTriageEnv:
    """HTTP client for the Email Triage environment server."""

    def __init__(self, base_url: str) -> None:
        self._base_url   = base_url.rstrip("/")
        self._client     = httpx.AsyncClient(timeout=60.0)
        self._session_id = None

    async def reset(self) -> _StepResult:
        resp = await self._client.post(f"{self._base_url}/reset")
        resp.raise_for_status()
        payload = resp.json()
        self._session_id = payload.get("session_id") or payload.get("episode_id")
        return self._parse(payload)

    async def step(self, action: EmailTriageAction) -> _StepResult:
        action_data = {
            "priority": action.priority,
            "category": action.category,
            "route":    action.route,
        }
        payload: dict = {"action": action_data}
        if self._session_id:
            payload["session_id"] = self._session_id

        resp = await self._client.post(f"{self._base_url}/step", json=payload)

        # Fallback: flat fields (stateless HTTP mode)
        if resp.status_code == 422:
            payload = dict(action_data)
            if self._session_id:
                payload["session_id"] = self._session_id
            resp = await self._client.post(f"{self._base_url}/step", json=payload)

        # Fallback: wrapped, no session
        if resp.status_code == 422:
            resp = await self._client.post(
                f"{self._base_url}/step", json={"action": action_data}
            )

        resp.raise_for_status()
        return self._parse(resp.json())

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "EmailTriageEnv":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    def _parse(self, payload: dict) -> _StepResult:
        obs_data = payload.get("observation", {})
        observation = EmailTriageObservation(
            email_id              = obs_data.get("email_id", ""),
            email_subject         = obs_data.get("email_subject", ""),
            email_sender          = obs_data.get("email_sender", ""),
            email_body            = obs_data.get("email_body", ""),
            last_priority_correct = obs_data.get("last_priority_correct"),
            last_category_correct = obs_data.get("last_category_correct"),
            last_route_correct    = obs_data.get("last_route_correct"),
            emails_remaining      = obs_data.get("emails_remaining", 0),
            current_streak        = obs_data.get("current_streak", 0),
            # Ground truth fields (named fields, not metadata)
            true_priority         = obs_data.get("true_priority"),
            true_category         = obs_data.get("true_category"),
            true_route            = obs_data.get("true_route"),
            is_business_critical  = obs_data.get("is_business_critical", False),
            is_phishing           = obs_data.get("is_phishing", False),
            linked_incident       = obs_data.get("linked_incident", False),
            done                  = payload.get("done", False),
            reward                = payload.get("reward"),
            metadata              = obs_data.get("metadata", {}),
        )
        return _StepResult(
            observation=observation,
            reward=payload.get("reward") or 0.0,
            done=payload.get("done", False),
        )


# ---------------------------------------------------------------------------
# Environment variables -- per hackathon guidelines
# ---------------------------------------------------------------------------
# API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")

# MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
MODEL_NAME   = os.getenv("MODEL_NAME",   "qwen/qwen-2.5-72b-instruct")

HF_TOKEN     = os.getenv("HF_TOKEN")
SERVER_URL   = os.getenv("EMAIL_RL_SERVER_URL", "http://localhost:8000")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK   = "Email_RL"
MAX_STEPS   = 10
TEMPERATURE = 0.3
MAX_TOKENS  = 500  # increased for action_plan and threat_report JSON

# -- Domain constants ---------------------------------------------------
VALID_PRIORITIES = {"low", "medium", "high", "urgent"}
VALID_CATEGORIES = {"spam", "newsletter", "support", "sales",
                    "internal", "billing", "security"}
VALID_ROUTES     = {"inbox", "archive", "support_team", "sales_team",
                    "security_team", "billing_team", "trash", "human_review"}


# ---------------------------------------------------------------------------
# Client-side graders -- each returns a float in [0.0, 1.0]
# ---------------------------------------------------------------------------

def _extract_graded_truth(obs_data: Dict) -> Dict:
    """
    Extract ground truth from the observation.
    Reads from named fields on the observation (true_priority, etc.)
    since the OpenEnv HTTP framework strips the metadata dict.
    Falls back to metadata keys for WebSocket mode compatibility.
    """
    metadata = obs_data.get("metadata") or {}
    return {
        "true_priority":        obs_data.get("true_priority",
                                metadata.get("true_priority", "")),
        "true_category":        obs_data.get("true_category",
                                metadata.get("true_category", "")),
        "true_route":           obs_data.get("true_route",
                                metadata.get("true_route", "")),
        "is_business_critical": obs_data.get("is_business_critical",
                                metadata.get("is_business_critical", False)),
    }


def _grade_spam_detection(action: EmailTriageAction, obs_data: Dict) -> float:
    """Binary: did the agent correctly identify spam vs legitimate?"""
    truth = _extract_graded_truth(obs_data)
    true_is_spam = truth["true_category"] == "spam"
    pred_is_spam = action.category.strip().lower() == "spam"
    return 1.0 if true_is_spam == pred_is_spam else 0.0


def _grade_priority_classification(action: EmailTriageAction, obs_data: Dict) -> float:
    """Binary: exact priority level match."""
    truth = _extract_graded_truth(obs_data)
    true_p = truth["true_priority"].strip().lower()
    pred_p = action.priority.strip().lower()
    return 1.0 if pred_p == true_p else 0.0


def _grade_full_triage(action: EmailTriageAction, obs_data: Dict) -> float:
    """Normalized weighted score across all 3 fields. Returns [0.0, 1.0]."""
    truth = _extract_graded_truth(obs_data)
    true_p = truth["true_priority"].strip().lower()
    true_c = truth["true_category"].strip().lower()
    true_r = truth["true_route"].strip().lower()

    pred_p = action.priority.strip().lower()
    pred_c = action.category.strip().lower()
    pred_r = action.route.strip().lower()

    p_ok = (pred_p == true_p) and (pred_p in VALID_PRIORITIES)
    c_ok = (pred_c == true_c) and (pred_c in VALID_CATEGORIES)
    r_ok = (pred_r == true_r) and (pred_r in VALID_ROUTES)

    score = 1.0 * p_ok + 0.5 * c_ok + 0.3 * r_ok
    if p_ok and (c_ok or r_ok):
        score += 0.1
    if p_ok and c_ok and r_ok:
        score += 0.2
    return round(min(score / 2.1, 1.0), 4)


def _grade_critical_escalation(action: EmailTriageAction, obs_data: Dict) -> float:
    """Binary: business-critical -> human_review detection."""
    truth = _extract_graded_truth(obs_data)
    is_critical  = bool(truth["is_business_critical"])
    routed_human = action.route.strip().lower() == "human_review"

    if is_critical and routed_human:
        return 1.0
    if not is_critical and not routed_human:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# NEW GRADER: Action Orchestrator
# Scores the quality of an action plan based on email context
# ---------------------------------------------------------------------------

# Expected systems per category/priority
_EXPECTED_SYSTEMS = {
    "urgent": {"pagerduty", "slack"},
    "high": {"jira", "slack"},
    "medium": {"jira"},
    "low": set(),
}

_EXPECTED_SYSTEMS_BY_CATEGORY = {
    "security": {"security_scan", "slack"},
    "billing": {"accounting", "email"},
    "support": {"jira", "email"},
    "sales": {"crm", "email", "calendar"},
    "internal": {"slack"},
    "spam": set(),
    "newsletter": set(),
}

_EXPECTED_STAKEHOLDERS = {
    "urgent": {"cto", "vp engineering", "on-call engineer", "account manager"},
    "high": {"team lead", "account manager"},
    "medium": {"team lead"},
    "low": set(),
}


def _parse_json_field(text: Optional[str]) -> Optional[Dict]:
    """Safely parse a JSON string, return None on failure."""
    if not text or not text.strip():
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def _grade_action_orchestrator(action: EmailTriageAction, obs_data: Dict) -> float:
    """
    Grades the action plan quality.

    Scoring (total 1.0):
      0.20 -- valid JSON with 'actions' list
      0.25 -- actions mention appropriate systems for the email priority/category
      0.20 -- sla_deadline present and reasonable
      0.20 -- stakeholders_to_notify present and relevant
      0.15 -- response_draft present with appropriate tone
    """
    truth = _extract_graded_truth(obs_data)
    true_priority = truth.get("true_priority", "medium")
    true_category = truth.get("true_category", "support")

    plan = _parse_json_field(action.action_plan)
    if plan is None:
        return 0.0

    score = 0.0

    # 1. Valid structure with actions list (0.20)
    actions = plan.get("actions")
    if isinstance(actions, list) and len(actions) > 0:
        score += 0.20

        # 2. Appropriate systems mentioned (0.25)
        mentioned_systems = set()
        for a in actions:
            if isinstance(a, dict):
                sys_name = str(a.get("system", "")).lower()
                if sys_name:
                    mentioned_systems.add(sys_name)

        expected_by_priority = _EXPECTED_SYSTEMS.get(true_priority, set())
        expected_by_category = _EXPECTED_SYSTEMS_BY_CATEGORY.get(true_category, set())
        all_expected = expected_by_priority | expected_by_category

        if all_expected:
            overlap = mentioned_systems & all_expected
            system_ratio = len(overlap) / len(all_expected) if all_expected else 0
            score += 0.25 * min(system_ratio, 1.0)
        else:
            # Low priority, no specific systems expected -- give credit for any plan
            score += 0.15 if len(mentioned_systems) > 0 else 0.0

    # 3. SLA deadline present and reasonable (0.20)
    sla = plan.get("sla_deadline") or plan.get("sla") or plan.get("deadline")
    if sla:
        sla_str = str(sla).lower()
        if true_priority == "urgent" and any(w in sla_str for w in ["1 hour", "30 min", "immediate", "asap"]):
            score += 0.20
        elif true_priority == "high" and any(w in sla_str for w in ["4 hour", "today", "same day", "end of day"]):
            score += 0.20
        elif true_priority in ("medium", "low"):
            score += 0.15  # any SLA is reasonable for lower priority
        else:
            score += 0.05  # present but not well-calibrated

    # 4. Stakeholders present and relevant (0.20)
    stakeholders = plan.get("stakeholders_to_notify") or plan.get("stakeholders") or plan.get("notify")
    if isinstance(stakeholders, list) and len(stakeholders) > 0:
        stakeholder_lower = {str(s).lower() for s in stakeholders}
        expected_stakeholders = _EXPECTED_STAKEHOLDERS.get(true_priority, set())
        if expected_stakeholders:
            overlap = sum(1 for es in expected_stakeholders if any(es in sl for sl in stakeholder_lower))
            ratio = overlap / len(expected_stakeholders)
            score += 0.20 * min(ratio, 1.0)
        else:
            score += 0.10  # low priority, any stakeholder list is fine

    # 5. Response draft present (0.15)
    has_response = False
    if isinstance(actions, list):
        for a in actions:
            if isinstance(a, dict):
                if a.get("action") in ("draft_reply", "reply", "respond", "email"):
                    has_response = True
                if "response" in str(a.get("system", "")).lower():
                    has_response = True
                if a.get("message") or a.get("draft") or a.get("body"):
                    has_response = True
    if plan.get("response_draft") or plan.get("draft") or plan.get("reply"):
        has_response = True
    if has_response:
        score += 0.15

    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# NEW GRADER: Threat Assessment
# Scores security threat detection quality
# ---------------------------------------------------------------------------

_PHISHING_INDICATORS = {
    "spoofed_domain", "suspicious_url", "urgency_pressure", "credential_request",
    "unusual_request", "secrecy_pressure", "ceo_impersonation", "executive_impersonation",
    "changed_bank_details", "fake_login", "typosquatting", "impersonation",
}

_ATTACK_VECTORS = {
    "ceo_impersonation", "business_email_compromise", "credential_phishing",
    "invoice_fraud", "spear_phishing", "fake_document", "fake_it_support",
}


def _grade_threat_assessment(action: EmailTriageAction, obs_data: Dict) -> float:
    """
    Grades security threat assessment quality.

    Scoring (total 1.0):
      0.30 -- correct threat/no-threat classification
      0.20 -- threat_type or attack_vector identified (if phishing)
      0.20 -- indicators list quality (if phishing)
      0.15 -- recommended_actions present and sensible
      0.15 -- risk_score calibrated to actual threat level
    """
    truth = _extract_graded_truth(obs_data)
    is_phishing = bool(obs_data.get("is_phishing", False))
    true_category = truth.get("true_category", "")

    report = _parse_json_field(action.threat_report)
    if report is None:
        # No report -- partial credit if agent at least got category right
        if is_phishing and action.category.strip().lower() == "security":
            return 0.15
        if not is_phishing and action.category.strip().lower() != "security":
            return 0.15
        return 0.0

    score = 0.0

    # 1. Correct threat classification (0.30)
    is_threat = report.get("is_threat", report.get("threat_detected", False))
    threat_type = str(report.get("threat_type", "")).lower()
    risk_score_val = report.get("risk_score", 0)

    if is_phishing:
        # Should detect as threat
        if is_threat or threat_type not in ("", "none", "legitimate") or (isinstance(risk_score_val, (int, float)) and risk_score_val > 5):
            score += 0.30
    else:
        # Should NOT flag as threat
        if not is_threat and threat_type in ("", "none", "legitimate", "low_risk"):
            score += 0.30
        elif isinstance(risk_score_val, (int, float)) and risk_score_val <= 3:
            score += 0.20

    # For non-phishing emails, remaining criteria are less relevant
    if not is_phishing:
        # Give partial credit for having a structured report
        if isinstance(report, dict) and len(report) >= 3:
            score += 0.10
        return round(min(score, 1.0), 4)

    # 2. Attack vector identified (0.20) -- only for phishing
    if threat_type:
        known_vectors = _ATTACK_VECTORS | {"phishing", "social_engineering", "bec", "fraud"}
        if any(v in threat_type for v in known_vectors):
            score += 0.20
        else:
            score += 0.05  # some attempt

    # 3. Indicators list quality (0.20) -- only for phishing
    indicators = report.get("indicators", [])
    if isinstance(indicators, list) and len(indicators) > 0:
        indicator_lower = {str(i).lower().replace(" ", "_") for i in indicators}
        known_hits = sum(1 for ki in _PHISHING_INDICATORS if any(ki in il for il in indicator_lower))
        if known_hits >= 3:
            score += 0.20
        elif known_hits >= 2:
            score += 0.15
        elif known_hits >= 1:
            score += 0.10
        else:
            score += 0.05  # has indicators but not from known set

    # 4. Recommended actions (0.15)
    rec_actions = report.get("recommended_actions", [])
    if isinstance(rec_actions, list) and len(rec_actions) > 0:
        action_str = " ".join(str(a).lower() for a in rec_actions)
        good_actions = ["quarantine", "notify", "security", "block", "investigate", "verify", "report"]
        hits = sum(1 for ga in good_actions if ga in action_str)
        if hits >= 2:
            score += 0.15
        elif hits >= 1:
            score += 0.10
        else:
            score += 0.05

    # 5. Risk score calibration (0.15)
    if isinstance(risk_score_val, (int, float)):
        if 7.0 <= risk_score_val <= 10.0:
            score += 0.15  # high risk for phishing = correct
        elif 5.0 <= risk_score_val < 7.0:
            score += 0.10
        elif risk_score_val > 0:
            score += 0.05

    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    name:              str
    difficulty:        str
    system_prompt:     str
    grader:            Callable[[EmailTriageAction, Dict], float]
    success_threshold: float


TASKS: List[TaskConfig] = [
    TaskConfig(
        name="spam-detection",
        difficulty="easy",
        success_threshold=0.6,
        grader=_grade_spam_detection,
        system_prompt=textwrap.dedent("""
            You are an email spam filter for a B2B software company.
            Your ONLY job is to decide whether each email is SPAM or LEGITIMATE.

            SPAM: unsolicited promotions, prize scams, phishing, fake invoices.
            LEGITIMATE: genuine support, billing, internal comms, sales, newsletters, security alerts.

            BEWARE OF PHISHING: Some emails impersonate executives (CEO wire transfer
            requests), fake IT password resets, or spoofed invoices with changed bank
            details. These are NOT legitimate -- classify as "security" and route to
            "security_team".

            You must output all three fields, but focus on getting category right.

            PRIORITY : low | medium | high | urgent
            CATEGORY : spam | newsletter | support | sales | internal | billing | security
            ROUTE    : inbox | archive | support_team | sales_team |
                       security_team | billing_team | trash | human_review

            Reply ONLY with these three XML tags:
                <priority>VALUE</priority>
                <category>VALUE</category>
                <route>VALUE</route>
        """).strip(),
    ),

    TaskConfig(
        name="priority-classification",
        difficulty="medium",
        success_threshold=0.5,
        grader=_grade_priority_classification,
        system_prompt=textwrap.dedent("""
            You are an email urgency classifier for a B2B software company.
            Assign the correct PRIORITY LEVEL to each email.

            PRIORITY LEVELS:
                low    -- no time pressure (spam, newsletters, FYI, optional reads)
                medium -- handle within 1-2 days (routine support, billing, standard sales, reminders)
                high   -- handle today (escalated support, large sales, overdue payments,
                         access reviews, security audits, sophisticated phishing attempts)
                urgent -- act immediately (production outages, data breaches, critical CVEs,
                         severely overdue invoices, suspicious logins, CEO fraud attempts)

            BEWARE: Phishing emails that impersonate executives or fake urgent IT requests
            should be classified as "high" or "urgent" priority since they represent
            active security threats requiring immediate attention.

            PRIORITY : low | medium | high | urgent
            CATEGORY : spam | newsletter | support | sales | internal | billing | security
            ROUTE    : inbox | archive | support_team | sales_team |
                       security_team | billing_team | trash | human_review

            Reply ONLY with these three XML tags:
                <priority>VALUE</priority>
                <category>VALUE</category>
                <route>VALUE</route>
        """).strip(),
    ),

    TaskConfig(
        name="full-triage",
        difficulty="hard",
        success_threshold=0.4,
        grader=_grade_full_triage,
        system_prompt=textwrap.dedent("""
            You are an expert email triage assistant for a B2B software company.
            Classify each email across all three dimensions: priority, category, and route.

            PRIORITY : low | medium | high | urgent
            CATEGORY : spam | newsletter | support | sales | internal | billing | security
            ROUTE    : inbox | archive | support_team | sales_team |
                       security_team | billing_team | trash | human_review

            ROUTING GUIDE:
                spam       -> trash           newsletter -> archive
                support    -> support_team    sales      -> sales_team
                internal   -> inbox           billing    -> billing_team
                security   -> security_team
                BUSINESS CRITICAL (legal/compliance/large contracts/claims) -> human_review

            PHISHING DETECTION:
                Watch for these red flags and route to security_team:
                - CEO/executive impersonation requesting wire transfers
                - Fake IT password reset links with suspicious URLs
                - Invoices with "updated" bank details
                - Spoofed domains (g00gle, paypa1, docusign-secure.xyz)
                - Urgency pressure + request for credentials or money

            Reply ONLY with these three XML tags:
                <priority>VALUE</priority>
                <category>VALUE</category>
                <route>VALUE</route>
        """).strip(),
    ),

    TaskConfig(
        name="critical-escalation",
        difficulty="hard",
        success_threshold=0.6,
        grader=_grade_critical_escalation,
        system_prompt=textwrap.dedent("""
            You are an expert email triage assistant for a B2B software company.
            Your primary job is to identify emails requiring HUMAN SIGN-OFF
            and route them to 'human_review'.

            ROUTE TO human_review when the email involves:
                - Legal disputes, lawsuits, cease-and-desist letters
                - Large contract negotiations ($10k+ or enterprise deals)
                - GDPR / regulatory compliance violations or audits
                - Insurance claims or workers compensation
                - Company-wide policy changes requiring executive approval
                - Any email where an automated decision could create legal liability

            Route to the STANDARD queue for everything else:
                spam       -> trash           newsletter -> archive
                support    -> support_team    sales      -> sales_team
                internal   -> inbox           billing    -> billing_team
                security   -> security_team

            IMPORTANT: Do NOT over-escalate. Phishing emails should go to
            security_team, NOT human_review. Only genuine legal/compliance/
            contract matters need human_review.

            PRIORITY : low | medium | high | urgent
            CATEGORY : spam | newsletter | support | sales | internal | billing | security
            ROUTE    : inbox | archive | support_team | sales_team |
                       security_team | billing_team | trash | human_review

            Reply ONLY with these three XML tags:
                <priority>VALUE</priority>
                <category>VALUE</category>
                <route>VALUE</route>
        """).strip(),
    ),

    TaskConfig(
        name="action-orchestrator",
        difficulty="hard",
        success_threshold=0.3,
        grader=_grade_action_orchestrator,
        system_prompt=textwrap.dedent("""
            You are an AI Chief of Staff for a B2B software company.
            For each email, you must BOTH classify it AND generate a complete action plan.

            CLASSIFICATION (same as before):
                PRIORITY : low | medium | high | urgent
                CATEGORY : spam | newsletter | support | sales | internal | billing | security
                ROUTE    : inbox | archive | support_team | sales_team |
                           security_team | billing_team | trash | human_review

            ACTION PLAN -- a JSON object describing what to do about this email:
                "actions": list of actions, each with:
                    "system": which tool/platform (pagerduty, slack, jira, email, calendar, crm, accounting, security_scan)
                    "action": what to do (create_incident, send_message, create_ticket, draft_reply, schedule_meeting, etc.)
                    "details": brief description
                "sla_deadline": how quickly must this be handled ("1 hour", "4 hours", "end of day", "this week")
                "stakeholders_to_notify": list of roles to involve (["CTO", "VP Engineering", "Account Manager", "Team Lead", "On-Call Engineer"])
                "response_draft": brief 1-2 sentence reply to the sender (if reply needed)

            EXAMPLES:
                Urgent production outage -> pagerduty P1, slack #incidents, jira ticket, SLA 1 hour, notify CTO + on-call
                Sales RFP with deadline -> crm update, calendar demo, email reply, SLA end of day, notify team lead
                Spam email -> no actions needed, SLA: none

            Reply with these XML tags:
                <priority>VALUE</priority>
                <category>VALUE</category>
                <route>VALUE</route>
                <action_plan>{"actions": [...], "sla_deadline": "...", "stakeholders_to_notify": [...], "response_draft": "..."}</action_plan>
        """).strip(),
    ),

    TaskConfig(
        name="threat-assessment",
        difficulty="hard",
        success_threshold=0.3,
        grader=_grade_threat_assessment,
        system_prompt=textwrap.dedent("""
            You are a cybersecurity analyst for a B2B software company.
            For each email, you must classify it AND produce a threat assessment report.

            CLASSIFICATION (same as before):
                PRIORITY : low | medium | high | urgent
                CATEGORY : spam | newsletter | support | sales | internal | billing | security
                ROUTE    : inbox | archive | support_team | sales_team |
                           security_team | billing_team | trash | human_review

            THREAT ASSESSMENT -- a JSON object evaluating security risk:
                "is_threat": true or false
                "threat_type": type of attack if threat detected
                    Options: "business_email_compromise", "credential_phishing", "invoice_fraud",
                             "spear_phishing", "ceo_impersonation", "fake_it_support",
                             "fake_document", "social_engineering", "none"
                "confidence": 0.0 to 1.0 (how sure are you)
                "indicators": list of red flags found
                    Options: "spoofed_domain", "suspicious_url", "urgency_pressure",
                             "credential_request", "unusual_request", "secrecy_pressure",
                             "ceo_impersonation", "changed_bank_details", "typosquatting",
                             "fake_login", "impersonation"
                "recommended_actions": what to do
                    Options: "quarantine", "notify_security", "block_sender", "investigate",
                             "verify_with_sender", "report_phishing", "no_action"
                "risk_score": 0.0 to 10.0 (0 = safe, 10 = critical threat)

            RED FLAGS TO WATCH FOR:
                - CEO/executive requesting wire transfers or secrecy
                - Suspicious URLs (g00gle, paypa1, -secure.xyz domains)
                - Changed bank details on invoices
                - Fake password reset or account verification links
                - Urgency + credential requests combined

            FOR LEGITIMATE EMAILS: set is_threat=false, threat_type="none",
                indicators=[], risk_score between 0.0-2.0

            Reply with these XML tags:
                <priority>VALUE</priority>
                <category>VALUE</category>
                <route>VALUE</route>
                <threat_report>{"is_threat": ..., "threat_type": "...", "confidence": ..., "indicators": [...], "recommended_actions": [...], "risk_score": ...}</threat_report>
        """).strip(),
    ),
]


# ---------------------------------------------------------------------------
# XML parser
# ---------------------------------------------------------------------------

_PRIORITY_RE    = re.compile(r"<priority>\s*([^<]+?)\s*</priority>", re.IGNORECASE)
_CATEGORY_RE    = re.compile(r"<category>\s*([^<]+?)\s*</category>", re.IGNORECASE)
_ROUTE_RE       = re.compile(r"<route>\s*([^<]+?)\s*</route>",       re.IGNORECASE)
_ACTION_PLAN_RE = re.compile(r"<action_plan>\s*(.*?)\s*</action_plan>", re.IGNORECASE | re.DOTALL)
_THREAT_RE      = re.compile(r"<threat_report>\s*(.*?)\s*</threat_report>", re.IGNORECASE | re.DOTALL)


def _parse_action(text: str) -> EmailTriageAction:
    """Extract all fields from XML output. Safe defaults on failure."""
    p = _PRIORITY_RE.search(text)
    c = _CATEGORY_RE.search(text)
    r = _ROUTE_RE.search(text)
    ap = _ACTION_PLAN_RE.search(text)
    tr = _THREAT_RE.search(text)

    priority = p.group(1).strip().lower() if p else "low"
    category = c.group(1).strip().lower() if c else "spam"
    route    = r.group(1).strip().lower() if r else "trash"

    if priority not in VALID_PRIORITIES:
        priority = "low"
    if category not in VALID_CATEGORIES:
        category = "spam"
    if route not in VALID_ROUTES:
        route = "trash"

    action_plan    = ap.group(1).strip() if ap else None
    threat_report  = tr.group(1).strip() if tr else None

    return EmailTriageAction(
        priority=priority,
        category=category,
        route=route,
        action_plan=action_plan,
        threat_report=threat_report,
    )


# ---------------------------------------------------------------------------
# Stdout logging -- exact format per hackathon guidelines
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    # Clamp reward strictly between 0 and 1
    reward = max(0.01, min(0.99, reward))
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    score = sum(rewards) / len(rewards) if rewards else 0.01
    # Clamp score strictly between 0 and 1 (not exactly 0.0 or 1.0)
    score = max(0.01, min(0.99, score))
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM call with error reporting
# ---------------------------------------------------------------------------

def _call_llm(
    client: OpenAI,
    system_prompt: str,
    obs: EmailTriageObservation,
    step: int,
    history: List[str],
) -> str:
    """Build user prompt and call the LLM. Returns raw text or empty on error."""
    feedback = ""
    if obs.last_priority_correct is not None:
        parts = [
            f"priority={'OK' if obs.last_priority_correct else 'WRONG'}",
            f"category={'OK' if obs.last_category_correct else 'WRONG'}",
            f"route={'OK'    if obs.last_route_correct    else 'WRONG'}",
        ]
        feedback = f"\nPrevious action feedback: {', '.join(parts)} | streak={obs.current_streak}"

    history_block = ""
    if history:
        history_block = "\nRecent decisions:\n" + "\n".join(f"  {h}" for h in history[-3:])

    linked_hint = ""
    if obs.metadata and obs.metadata.get("linked_incident"):
        linked_hint = "\nWARNING: This email may be related to another incident in this batch."

    user_prompt = textwrap.dedent(f"""
        Step {step} | Emails remaining after this: {obs.emails_remaining}
        {feedback}
        {history_block}
        {linked_hint}

        --- EMAIL ---
        From   : {obs.email_sender}
        Subject: {obs.email_subject}
        Body   :
        {obs.email_body}
        --- END EMAIL ---

        Classify this email. Reply ONLY with the three XML tags.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        result = (completion.choices[0].message.content or "").strip()
        return result
    except Exception as exc:
        return ""


# ---------------------------------------------------------------------------
# Episode runner -- STATELESS HTTP MODE (reset + step per email)
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, task: TaskConfig) -> None:
    """
    Run one task episode in stateless HTTP mode.
    Does reset() + step() per email, repeated MAX_STEPS times.
    """
    env = EmailTriageEnv(base_url=SERVER_URL)

    history:     List[str]   = []
    rewards:     List[float] = []
    steps_taken: int         = 0
    success:     bool        = False

    log_start(task=task.name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            error: Optional[str] = None

            # -- Reset to get a fresh email ---------------------------------
            reset_result = await env.reset()
            obs = reset_result.observation

            # -- Cache ground truth from reset() observation ----------------
            reset_obs_data = {
                "true_priority":        obs.true_priority if hasattr(obs, 'true_priority') else None,
                "true_category":        obs.true_category if hasattr(obs, 'true_category') else None,
                "true_route":           obs.true_route if hasattr(obs, 'true_route') else None,
                "is_business_critical": obs.is_business_critical if hasattr(obs, 'is_business_critical') else False,
                "metadata":             obs.metadata,
            }

            # -- LLM decision -----------------------------------------------
            raw_text = _call_llm(client, task.system_prompt, obs, step, history)
            action   = _parse_action(raw_text)
            action_str = (
                f"priority={action.priority},"
                f"category={action.category},"
                f"route={action.route}"
            )

            # -- Step to submit action --------------------------------------
            step_result = await env.step(action)
            done        = step_result.done

            # -- Grade using RESET observation (correct email's ground truth)
            task_reward = task.grader(action, reset_obs_data)
            # Clamp strictly between 0 and 1
            task_reward = max(0.01, min(0.99, task_reward))

            rewards.append(task_reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=task_reward,
                done=(step == MAX_STEPS),
                error=error,
            )

            history.append(f"Step {step}: {action_str} -> reward={task_reward:.2f}")

        # Episode score = mean per-step reward
        score   = sum(rewards) / len(rewards) if rewards else 0.01
        score   = round(min(max(score, 0.0), 1.0), 4)
        success = score >= task.success_threshold

    except Exception:
        pass

    finally:
        try:
            await env.close()
        except Exception:
            pass

        log_end(success=success, steps=steps_taken, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Run all 4 tasks in sequence.
    Each task: [START] -> [STEP] x 10 -> [END].
    Estimated: 4 tasks x 10 steps x ~2s/call = ~80s (well within 20min).
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    for task in TASKS:
        await run_task(client, task)


if __name__ == "__main__":
    asyncio.run(main())