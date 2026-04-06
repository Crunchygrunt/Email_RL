
"""
inference.py — Email Triage RL Environment
==========================================
MANDATORY ENVIRONMENT VARIABLES
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Defaults (must reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

STDOUT FORMAT — strictly followed, no deviation:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

TASKS (3 tasks, each with its own grader, all rewards in [0.0, 1.0]):
    spam-detection          Binary: did the agent correctly identify spam?
    priority-classification Binary: did the agent assign the correct urgency level?
    full-triage             Normalized: weighted correctness across all three fields

All per-step rewards are normalized to [0.0, 1.0] by the client-side graders.
Ground truth is read from obs.metadata (embedded by the server in every observation).
"""
from dotenv import load_dotenv
load_dotenv()
import asyncio
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import httpx
from dataclasses import dataclass as _dataclass
from openai import OpenAI

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from models import EmailTriageAction, EmailTriageObservation
except ModuleNotFoundError:
    from Email_RL.models import EmailTriageAction, EmailTriageObservation


@_dataclass
class _StepResult:
    observation: EmailTriageObservation
    reward:      float
    done:        bool


class EmailTriageEnv:
    """
    HTTP client for the Email Triage environment.
    Works against both a local server and a HuggingFace Space.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url    = base_url.rstrip("/")
        self._client      = httpx.AsyncClient(timeout=30.0)
        self._session_id  = None

    async def reset(self) -> _StepResult:
        resp = await self._client.post(f"{self._base_url}/reset")
        resp.raise_for_status()
        payload = resp.json()
        self._session_id = payload.get("session_id") or payload.get("episode_id")
        return self._parse(payload)

    async def step(self, action: EmailTriageAction) -> _StepResult:
        action_data = {"priority": action.priority,
                       "category": action.category,
                       "route":    action.route}
        # Format 1: action wrapped (what the server expects when session exists)
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
            email_id             = obs_data.get("email_id", ""),
            email_subject        = obs_data.get("email_subject", ""),
            email_sender         = obs_data.get("email_sender", ""),
            email_body           = obs_data.get("email_body", ""),
            last_priority_correct= obs_data.get("last_priority_correct"),
            last_category_correct= obs_data.get("last_category_correct"),
            last_route_correct   = obs_data.get("last_route_correct"),
            emails_remaining     = obs_data.get("emails_remaining", 0),
            current_streak       = obs_data.get("current_streak", 0),
            done                 = payload.get("done", False),
            reward               = payload.get("reward"),
            metadata             = obs_data.get("metadata", {}),
        )
        return _StepResult(
            observation = observation,
            reward      = payload.get("reward") or 0.0,
            done        = payload.get("done", False),
        )

# ── Environment variables (mandatory) ─────────────────────────────────────
IMAGE_NAME   = os.getenv("IMAGE_NAME")                              # Docker image (optional)
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-1.5B-Instruct")
SERVER_URL   = os.getenv("EMAIL_RL_SERVER_URL", "http://localhost:8000")

BENCHMARK  = "Email_RL"
MAX_STEPS  = 10     # matches EmailTriageEnvironment.EPISODE_LENGTH
TEMPERATURE = 0.3   # lower = more deterministic classification
MAX_TOKENS  = 80    # XML response is short

# ── Domain constants (must stay in sync with models.py) ───────────────────
VALID_PRIORITIES = {"low", "medium", "high", "urgent"}
VALID_CATEGORIES = {"spam", "newsletter", "support", "sales",
                    "internal", "billing", "security"}
VALID_ROUTES     = {"inbox", "archive", "support_team", "sales_team",
                    "security_team", "billing_team", "trash", "human_review"}

# ─────────────────────────────────────────────────────────────────────────────
# Client-side graders — each returns a float in [0.0, 1.0]
# Ground truth is read from obs.metadata which the server embeds in every
# observation: {"true_priority": ..., "true_category": ..., "true_route": ...}
# ─────────────────────────────────────────────────────────────────────────────

def _grade_spam_detection(action: EmailTriageAction, metadata: Dict) -> float:
    """
    Binary spam detection grader.

    Score = 1.0 when the agent correctly identifies spam vs. legitimate.
    Score = 0.0 otherwise.

    The agent still outputs all three fields (priority/category/route) but
    only the binary spam/not-spam signal from `category` is scored here.
    """
    true_is_spam = metadata.get("true_category", "") == "spam"
    pred_is_spam = action.category.strip().lower() == "spam"
    return 1.0 if true_is_spam == pred_is_spam else 0.0


def _grade_priority_classification(action: EmailTriageAction, metadata: Dict) -> float:
    """
    Binary priority classification grader.

    Score = 1.0 when the agent assigns the exact correct urgency level.
    Score = 0.0 otherwise.

    Strict match required — 'medium' when ground truth is 'high' scores 0.0.
    This teaches the model to distinguish urgency levels precisely.
    """
    true_priority = metadata.get("true_priority", "").strip().lower()
    pred_priority = action.priority.strip().lower()
    return 1.0 if pred_priority == true_priority else 0.0


def _grade_full_triage(action: EmailTriageAction, metadata: Dict) -> float:
    """
    Normalized full triage grader — scores all three fields.

    Uses the same weighted formula as the server (from TriageGrader.base_score)
    but normalises the result to [0.0, 1.0] by dividing by the maximum
    possible base score of 2.1.

    Weights:
        priority  1.0  (most important signal)
        category  0.5
        route     0.3
        format bonus   +0.1  (priority correct + ≥1 other correct)
        perfect bonus  +0.2  (all three correct)
        max possible = 2.1

    Returns a float in [0.0, 1.0].
    """
    true_priority = metadata.get("true_priority", "").strip().lower()
    true_category = metadata.get("true_category", "").strip().lower()
    true_route    = metadata.get("true_route",    "").strip().lower()

    pred_priority = action.priority.strip().lower()
    pred_category = action.category.strip().lower()
    pred_route    = action.route.strip().lower()

    priority_ok = (pred_priority == true_priority) and (pred_priority in VALID_PRIORITIES)
    category_ok = (pred_category == true_category) and (pred_category in VALID_CATEGORIES)
    route_ok    = (pred_route    == true_route)    and (pred_route    in VALID_ROUTES)

    score = 1.0 * priority_ok + 0.5 * category_ok + 0.3 * route_ok
    if priority_ok and (category_ok or route_ok):
        score += 0.1
    if priority_ok and category_ok and route_ok:
        score += 0.2

    _MAX_BASE_SCORE = 2.1
    return round(min(score / _MAX_BASE_SCORE, 1.0), 4)


def _grade_critical_escalation(action: EmailTriageAction, metadata: Dict) -> float:
    """
    Binary critical escalation grader.

    Scores whether the agent correctly identifies emails that require human
    sign-off (legal disputes, large contract negotiations, GDPR/compliance
    violations, insurance claims, policy changes) and routes them to
    'human_review', while NOT over-escalating normal emails.

    Score = 1.0 when:
        - Email IS business-critical AND agent routed to 'human_review'
        - Email is NOT business-critical AND agent did NOT route to 'human_review'
    Score = 0.0 when:
        - Email IS business-critical AND agent routed elsewhere  (missed escalation)
        - Email is NOT business-critical AND agent routed to 'human_review' (over-escalation)

    Both failure modes are penalised equally — the agent must learn the
    boundary between routine and critical, not simply always escalate.
    """
    is_critical  = bool(metadata.get("is_business_critical", False))
    routed_human = action.route.strip().lower() == "human_review"

    if is_critical and routed_human:
        return 1.0   # correct escalation
    if not is_critical and not routed_human:
        return 1.0   # correct non-escalation
    return 0.0       # missed escalation or over-escalation


# ─────────────────────────────────────────────────────────────────────────────
# Task definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskConfig:
    name:              str
    system_prompt:     str
    grader:            Callable[[EmailTriageAction, Dict], float]
    success_threshold: float   # minimum mean reward to declare success


TASKS: List[TaskConfig] = [
    TaskConfig(
        name="spam-detection",
        success_threshold=0.6,
        grader=_grade_spam_detection,
        system_prompt=textwrap.dedent("""
            You are an email spam filter for a B2B software company.
            Your ONLY job is to decide whether each email is SPAM or LEGITIMATE.

            SPAM emails: unsolicited promotions, prize notifications, phishing,
                         scam messages, fake lottery winners.
            LEGITIMATE emails: genuine support requests, billing, internal
                               communication, sales enquiries, newsletters
                               from known senders, security alerts.

            You must still output all three fields, but focus on getting
            `category` right — use "spam" for spam, or any other valid category
            for legitimate mail.

            PRIORITY : low | medium | high | urgent
            CATEGORY : spam | newsletter | support | sales | internal | billing | security
            ROUTE    : inbox | archive | support_team | sales_team |
                       security_team | billing_team | trash

            Reward: 1.0 if you correctly detect spam vs. legitimate, 0.0 otherwise.

            Reply ONLY with these three XML tags:
                <priority>VALUE</priority>
                <category>VALUE</category>
                <route>VALUE</route>
        """).strip(),
    ),

    TaskConfig(
        name="priority-classification",
        success_threshold=0.5,
        grader=_grade_priority_classification,
        system_prompt=textwrap.dedent("""
            You are an email urgency classifier for a B2B software company.
            Your primary job is to assign the correct PRIORITY LEVEL to each email.

            PRIORITY LEVELS:
                low    — no time pressure (spam, newsletters, FYI, optional reads)
                medium — handle within 1-2 business days (routine support, billing queries,
                         standard sales leads, internal reminders)
                high   — handle today (escalated support, large sales opportunities,
                         overdue payments, access reviews, security audits)
                urgent — act immediately (production outages, data breaches, critical CVEs,
                         severely overdue invoices, suspicious logins)

            You must still output all three fields, but `priority` is what matters most.

            PRIORITY : low | medium | high | urgent
            CATEGORY : spam | newsletter | support | sales | internal | billing | security
            ROUTE    : inbox | archive | support_team | sales_team |
                       security_team | billing_team | trash

            Reward: 1.0 if you assign the exact correct priority, 0.0 otherwise.

            Reply ONLY with these three XML tags:
                <priority>VALUE</priority>
                <category>VALUE</category>
                <route>VALUE</route>
        """).strip(),
    ),

    TaskConfig(
        name="full-triage",
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
                spam       → trash           newsletter → archive
                support    → support_team    sales      → sales_team
                internal   → inbox           billing    → billing_team
                security   → security_team
                BUSINESS CRITICAL (legal/compliance/large contracts/claims) → human_review

            REWARD STRUCTURE (normalized to [0.0, 1.0]):
                Correct priority  → +1.0    Correct category → +0.5
                Correct route     → +0.3    All correct      → +0.2 bonus
                Format bonus (+0.1): priority correct AND ≥1 other field correct
                Maximum possible score = 2.1, divided to normalize to [0,1].

            Reply ONLY with these three XML tags:
                <priority>VALUE</priority>
                <category>VALUE</category>
                <route>VALUE</route>
        """).strip(),
    ),

    TaskConfig(
        name="critical-escalation",
        success_threshold=0.6,
        grader=_grade_critical_escalation,
        system_prompt=textwrap.dedent("""
            You are an expert email triage assistant for a B2B software company.
            Your primary job in this task is to identify emails that require
            HUMAN SIGN-OFF and route them to 'human_review'.

            ROUTE TO human_review when the email involves:
                - Legal disputes, lawsuits, cease-and-desist letters
                - Large contract negotiations (typically $10k+ or enterprise deals)
                - GDPR / regulatory compliance violations or audit requests
                - Insurance claims or workers compensation
                - Company-wide policy changes requiring executive or board approval
                - Any email where an automated decision could create legal liability

            Route to the STANDARD queue for everything else:
                spam       → trash           newsletter → archive
                support    → support_team    sales      → sales_team
                internal   → inbox           billing    → billing_team
                security   → security_team

            PRIORITY : low | medium | high | urgent
            CATEGORY : spam | newsletter | support | sales | internal | billing | security
            ROUTE    : inbox | archive | support_team | sales_team |
                       security_team | billing_team | trash | human_review

            REWARD STRUCTURE:
                Business-critical email routed to human_review  → 1.0
                Normal email NOT routed to human_review         → 1.0
                Business-critical email routed elsewhere        → 0.0  (missed escalation)
                Normal email routed to human_review             → 0.0  (over-escalation)

            Reply ONLY with these three XML tags:
                <priority>VALUE</priority>
                <category>VALUE</category>
                <route>VALUE</route>
        """).strip(),
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# XML parser
# ─────────────────────────────────────────────────────────────────────────────

_PRIORITY_RE = re.compile(r"<priority>\s*([^<]+?)\s*</priority>", re.IGNORECASE)
_CATEGORY_RE = re.compile(r"<category>\s*([^<]+?)\s*</category>", re.IGNORECASE)
_ROUTE_RE    = re.compile(r"<route>\s*([^<]+?)\s*</route>",       re.IGNORECASE)


def _parse_action(text: str) -> EmailTriageAction:
    """
    Extract (priority, category, route) from the model's XML output.
    Falls back to safe defaults on parse failure.
    """
    p = _PRIORITY_RE.search(text)
    c = _CATEGORY_RE.search(text)
    r = _ROUTE_RE.search(text)

    priority = p.group(1).strip().lower() if p else "low"
    category = c.group(1).strip().lower() if c else "spam"
    route    = r.group(1).strip().lower() if r else "trash"

    # Validate against allowed vocabularies, fall back on invalid
    if priority not in VALID_PRIORITIES:
        priority = "low"
    if category not in VALID_CATEGORIES:
        category = "spam"
    if route not in VALID_ROUTES:
        route = "trash"

    return EmailTriageAction(priority=priority, category=category, route=route)


# ─────────────────────────────────────────────────────────────────────────────
# Stdout logging helpers — must match format exactly
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(
    client: OpenAI,
    system_prompt: str,
    obs,
    step: int,
    history: List[str],
) -> str:
    """Build the user prompt and call the LLM. Returns raw text response."""
    feedback = ""
    if obs.last_priority_correct is not None:
        parts = []
        parts.append(f"priority={'✓' if obs.last_priority_correct else '✗'}")
        parts.append(f"category={'✓' if obs.last_category_correct else '✗'}")
        parts.append(f"route={'✓'    if obs.last_route_correct    else '✗'}")
        feedback = f"\nPrevious action: {', '.join(parts)} | streak={obs.current_streak}"

    history_block = ""
    if history:
        history_block = "\nRecent decisions:\n" + "\n".join(f"  {h}" for h in history[-3:])

    user_prompt = textwrap.dedent(f"""
        Step {step} of {MAX_STEPS} | Emails remaining after this: {obs.emails_remaining}
        {feedback}
        {history_block}

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


# ─────────────────────────────────────────────────────────────────────────────
# Single task episode runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_task(
    client: OpenAI,
    task: TaskConfig,
) -> None:
    """
    Run one full episode for a single task.

    Emits exactly:
        [START] ...
        [STEP]  ... × MAX_STEPS
        [END]   ...

    Reward computation:
        1. env.reset() returns obs with metadata containing ground truth for email_0.
        2. Before each step, cache obs.metadata (ground truth for the current email).
        3. Take the step → get new obs (for next email).
        4. Compute task-specific reward from cached metadata + action taken.
           This reward is always in [0.0, 1.0].
        5. Log the task-specific reward (not the server's shaped reward).
    """
    env = EmailTriageEnv(base_url=SERVER_URL)

    history:     List[str]   = []
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task.name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):

            # ── Reset to get a fresh email ──────────────────────────────
            # In stateless HTTP mode the server creates a new env instance
            # per request, so we reset before every step to get a new email.
            result = await env.reset()
            obs    = result.observation

            # ── LLM decision ────────────────────────────────────────────
            raw_text = _call_llm(client, task.system_prompt, obs, step, history)
            action   = _parse_action(raw_text)
            action_str = (
                f"priority={action.priority},"
                f"category={action.category},"
                f"route={action.route}"
            )

            # ── Environment step ─────────────────────────────────────────
            result = await env.step(action)
            obs    = result.observation
            done   = result.done
            error: Optional[str] = None

            # ── Ground truth for the email just graded ───────────────────
            step_metadata = obs.metadata or {}
            graded_metadata = {
                "true_priority":        step_metadata.get("graded_true_priority",
                                        step_metadata.get("true_priority", "")),
                "true_category":        step_metadata.get("graded_true_category",
                                        step_metadata.get("true_category", "")),
                "true_route":           step_metadata.get("graded_true_route",
                                        step_metadata.get("true_route", "")),
                "is_business_critical": step_metadata.get("graded_is_business_critical",
                                        step_metadata.get("is_business_critical", False)),
            }

            # ── Task-specific reward (always in [0.0, 1.0]) ──────────────
            task_reward = task.grader(action, graded_metadata)

            rewards.append(task_reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=task_reward, done=(step == MAX_STEPS), error=error)

            history.append(
                f"Step {step}: {action_str} → reward={task_reward:.2f}"
            )

        # Episode score = mean per-step reward, already in [0.0, 1.0]
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


# ─────────────────────────────────────────────────────────────────────────────
# Main — runs all 3 tasks sequentially
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    """
    Run all 4 tasks in sequence.

    Each task produces its own [START]→[STEP]×N→[END] block.
    Total runtime: 4 tasks × 10 steps × ~2s per LLM call ≈ 80s
    Well within the 20-minute limit on vcpu=2 / 8GB RAM.
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in TASKS:
        await run_task(client, task)


if __name__ == "__main__":
    asyncio.run(main())