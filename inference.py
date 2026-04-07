"""
inference.py -- Email Triage RL Environment
===========================================

MANDATORY ENVIRONMENT VARIABLES
API_BASE_URL   The API endpoint for the LLM.
MODEL_NAME     The model identifier to use for inference.
HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT -- strictly followed, no deviation:
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

TASKS (4 tasks, each with its own grader, all rewards in [0.0, 1.0]):
spam-detection           (easy)   -- binary spam vs legitimate
priority-classification  (medium) -- exact urgency level match
full-triage              (hard)   -- weighted score across all 3 fields
critical-escalation      (hard)   -- business-critical to human_review detection
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
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

from dataclasses import dataclass

@dataclass
class _StepResult:
    observation: EmailTriageObservation
    reward: float
    done: bool

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
            done                  = payload.get("done", False),
            reward                = payload.get("reward"),
            metadata              = obs_data.get("metadata", {}),
        )
        return _StepResult(
            observation=observation,
            reward=payload.get("reward") or 0.0,
            done=payload.get("done", False),
        )


# -- Environment variables ----------------------------------------------
API_KEY      = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY") 
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
SERVER_URL   = os.getenv("EMAIL_RL_SERVER_URL", "http://localhost:8000")

BENCHMARK   = "Email_RL"
MAX_STEPS   = 10
TEMPERATURE = 0.3
MAX_TOKENS  = 120

# -- Domain constants ---------------------------------------------------
VALID_PRIORITIES = {"low", "medium", "high", "urgent"}
VALID_CATEGORIES = {"spam", "newsletter", "support", "sales",
                    "internal", "billing", "security"}
VALID_ROUTES     = {"inbox", "archive", "support_team", "sales_team",
                    "security_team", "billing_team", "trash", "human_review"}


# ---------------------------------------------------------------------------
# Client-side graders -- each returns a float in [0.0, 1.0]
# Ground truth is read from obs.metadata GRADED keys (post-action, no leakage)
# ---------------------------------------------------------------------------

def _extract_graded_truth(metadata: Dict) -> Dict:
    """
    Extract ground truth for the JUST-GRADED email from metadata.
    Uses graded_true_* keys which are only available AFTER the agent acts.
    Falls back to true_* keys for backward compatibility.
    """
    return {
        "true_priority":        metadata.get("graded_true_priority",
                                metadata.get("true_priority", "")),
        "true_category":        metadata.get("graded_true_category",
                                metadata.get("true_category", "")),
        "true_route":           metadata.get("graded_true_route",
                                metadata.get("true_route", "")),
        "is_business_critical": metadata.get("graded_is_business_critical",
                                metadata.get("is_business_critical", False)),
    }


def _grade_spam_detection(action: EmailTriageAction, metadata: Dict) -> float:
    """Binary: did the agent correctly identify spam vs legitimate?"""
    truth = _extract_graded_truth(metadata)
    true_is_spam = truth["true_category"] == "spam"
    pred_is_spam = action.category.strip().lower() == "spam"
    return 1.0 if true_is_spam == pred_is_spam else 0.0


def _grade_priority_classification(action: EmailTriageAction, metadata: Dict) -> float:
    """Binary: exact priority level match."""
    truth = _extract_graded_truth(metadata)
    true_p = truth["true_priority"].strip().lower()
    pred_p = action.priority.strip().lower()
    return 1.0 if pred_p == true_p else 0.0


def _grade_full_triage(action: EmailTriageAction, metadata: Dict) -> float:
    """Normalized weighted score across all 3 fields. Returns [0.0, 1.0]."""
    truth = _extract_graded_truth(metadata)
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


def _grade_critical_escalation(action: EmailTriageAction, metadata: Dict) -> float:
    """Binary: business-critical -> human_review detection."""
    truth = _extract_graded_truth(metadata)
    is_critical  = bool(truth["is_business_critical"])
    routed_human = action.route.strip().lower() == "human_review"

    if is_critical and routed_human:
        return 1.0
    if not is_critical and not routed_human:
        return 1.0
    return 0.0


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

            LINKED INCIDENTS:
                Some emails may reference the same ongoing incident. Try to route
                related emails to the same team for consistent handling.

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
]


# ---------------------------------------------------------------------------
# XML parser
# ---------------------------------------------------------------------------

_PRIORITY_RE = re.compile(r"<priority>\s*([^<]+?)\s*</priority>", re.IGNORECASE)
_CATEGORY_RE = re.compile(r"<category>\s*([^<]+?)\s*</category>", re.IGNORECASE)
_ROUTE_RE    = re.compile(r"<route>\s*([^<]+?)\s*</route>",       re.IGNORECASE)


def _parse_action(text: str) -> EmailTriageAction:
    """Extract (priority, category, route) from XML output. Safe defaults on failure."""
    p = _PRIORITY_RE.search(text)
    c = _CATEGORY_RE.search(text)
    r = _ROUTE_RE.search(text)

    priority = p.group(1).strip().lower() if p else "low"
    category = c.group(1).strip().lower() if c else "spam"
    route    = r.group(1).strip().lower() if r else "trash"

    if priority not in VALID_PRIORITIES:
        priority = "low"
    if category not in VALID_CATEGORIES:
        category = "spam"
    if route not in VALID_ROUTES:
        route = "trash"

    return EmailTriageAction(priority=priority, category=category, route=route)


# ---------------------------------------------------------------------------
# Stdout logging -- exact format, no deviation
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(
    client: OpenAI,
    system_prompt: str,
    obs: EmailTriageObservation,
    step: int,
    history: List[str],
) -> str:
    """Build user prompt and call the LLM."""
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

    # Hint about linked incidents (from metadata, not an answer)
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
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        return ""


# ---------------------------------------------------------------------------
# Episode runner -- FULL EPISODES (reset once, step N times)
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, task: TaskConfig) -> None:
    """
    Run one full episode for a single task.
    Runs a proper multi-step episode (reset once, step N times).
    """
    env = EmailTriageEnv(base_url=SERVER_URL)

    history:     List[str]   = []
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task.name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset once to start the episode
        result = await env.reset()
        obs    = result.observation
        done   = result.done

        step = 0
        while not done and step < MAX_STEPS + 5:  # +5 buffer for injected escalations
            step += 1

            # LLM decision
            raw_text = _call_llm(client, task.system_prompt, obs, step, history)
            action   = _parse_action(raw_text)
            action_str = (
                f"priority={action.priority},"
                f"category={action.category},"
                f"route={action.route}"
            )

            # Step the environment
            result = await env.step(action)
            new_obs = result.observation
            done    = result.done
            error:  Optional[str] = None

            # Grade using post-action metadata
            step_metadata = new_obs.metadata or {}
            task_reward = task.grader(action, step_metadata)

            rewards.append(task_reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=task_reward,
                done=done,
                error=error,
            )

            history.append(f"Step {step}: {action_str} -> reward={task_reward:.2f}")
            obs = new_obs

        # Episode score = mean per-step reward
        score   = sum(rewards) / len(rewards) if rewards else 0.0
        score   = round(min(max(score, 0.0), 1.0), 4)
        success = score >= task.success_threshold

    except Exception as exc:
        log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)
        return

    finally:
        try:
            await env.close()
        except Exception:
            pass

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Run all 4 tasks in sequence.
    Each task produces [START] -> [STEP] x N -> [END].
    Estimated runtime: 4 tasks x ~12 steps x ~2s/call = ~100s (well within 20min).
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in TASKS:
        await run_task(client, task)


if __name__ == "__main__":
    asyncio.run(main())