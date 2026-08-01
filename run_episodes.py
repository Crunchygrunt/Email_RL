# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
run_episodes.py -- correct, multi-step episode runner.

Why this file exists (see WAREHOUSE.md "Known findings" for the full
writeup): inference.py was provided for the hackathon's automated grading
and is left completely untouched as a record of what was actually
submitted. Its hand-rolled HTTP client talks to the server's plain
/reset and /step routes. openenv-core serves those in "simulation" mode,
where EVERY call -- reset or step -- gets a brand-new, throwaway
EmailTriageEnvironment instance. There is no continuity even WITHIN a
single reset->step pair, let alone across a 10-email episode: the agent
is shown one email via /reset, and the very next /step call grades its
action against a different, freshly-and-independently-sampled email on
a different instance.

client.py's EmailTriageEnv(EnvClient) already does this correctly -- it
connects over WebSocket to /ws, which keeps ONE persistent, session-bound
environment instance alive for the life of the connection. This script is
the thin, correct driver for that client. It imports inference.py and
reuses its TASKS list, graders, system prompts, LLM call, and action
parser VERBATIM, so none of the grading logic is duplicated or drifts out
of sync -- this file only replaces the transport layer and the
per-episode control flow (reset once per episode, step through every
email in it, instead of reset-per-email).

Usage (same environment variables as inference.py):
    export API_BASE_URL=...
    export MODEL_NAME=...
    export HF_TOKEN=...
    export EMAIL_RL_SERVER_URL=http://localhost:8000
    python run_episodes.py [--episodes 3] [--tasks spam-detection,full-triage]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from typing import List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)   # flat imports: inference, telemetry, models
sys.path.insert(0, _PARENT_DIR)   # package imports: Email_RL.client

# Reuse inference.py's TASKS, graders, prompts, LLM call, and action
# parser as-is -- this script only swaps the client + control flow.
import inference  # noqa: E402

# gpt-oss-120b (and other hidden-reasoning models) can spend the whole
# completion budget on invisible reasoning tokens before writing any
# visible answer, leaving `content` empty even on a genuine HTTP 200 --
# see WAREHOUSE.md / the empty-response warning in _parse_action_lenient
# below. inference.py is frozen, so we bump the module-level constant
# _call_llm reads at call time instead of editing the file. Raising this
# doesn't cost more tokens than the model actually uses -- it just stops
# truncating generation before it reaches the answer.
inference.MAX_TOKENS = 2000

# client.py's own `from .models import ...` is a bare relative import with
# no flat-script fallback (unlike inference.py/models.py/__init__.py), so
# it can only be imported in proper package context -- i.e. as
# Email_RL.client, not a flat `import client`. Try flat first anyway in
# case this script is ever run from inside an already-installed package
# layout where that resolves differently; fall back to the package path,
# which is what actually works from a plain `python run_episodes.py`.
try:
    from client import EmailTriageEnv
except ImportError:
    from Email_RL.client import EmailTriageEnv

try:
    import telemetry
except ModuleNotFoundError:
    from Email_RL import telemetry

import websockets.exceptions
from openai import OpenAI

# --- Rate-limit / flaky-connection resilience -------------------------
#
# Both added after a real run hit two separate failures back to back:
# (1) Groq's free-tier RPM/TPM budget got tripped ~160 requests into a
#     240-request run, and _call_llm's blanket except-Exception-return-''
#     made that indistinguishable from a genuine parse failure until we
#     started retrying instead of accepting the first empty response.
# (2) The local WebSocket connection itself got dropped mid-run
#     (websockets.exceptions.ConnectionClosedError), which previously
#     took the entire multi-hour, multi-task run down with it instead of
#     just losing the one in-flight episode.
_LLM_RETRY_DELAYS = (5, 15, 30)     # seconds; retried on an EMPTY llm response only
_REQUEST_PACING_DELAY = 1.0         # seconds; small proactive gap between LLM calls
_MAX_EPISODE_CONNECTION_RETRIES = 3
_EPISODE_RETRY_DELAY = 10           # seconds, before reconnecting with a fresh session


async def _call_llm_with_retry(llm_client: OpenAI, system_prompt: str, obs, step: int, history: List[str]) -> str:
    """
    Wraps inference._call_llm with retry-on-empty-response.

    _call_llm swallows every exception and returns "" -- so an empty
    response could mean a genuine formatting failure OR a rate limit OR
    a dropped connection to the LLM provider. We can't tell which from
    the return value alone, but rate limits are the most likely cause on
    a free tier mid-run, and they're worth waiting out rather than
    immediately burning a graded step on a forced default. Only truly
    persistent emptiness (after all retries) gets treated as real by
    _parse_action_lenient.
    """
    raw_text = inference._call_llm(llm_client, system_prompt, obs, step, history)
    for attempt, delay in enumerate(_LLM_RETRY_DELAYS, start=1):
        if raw_text and raw_text.strip():
            return raw_text
        print(
            f"[retry] empty LLM response (likely a rate limit) -- "
            f"attempt {attempt}/{len(_LLM_RETRY_DELAYS)}, waiting {delay}s before retrying...",
            file=sys.stderr,
        )
        await asyncio.sleep(delay)
        raw_text = inference._call_llm(llm_client, system_prompt, obs, step, history)
    return raw_text


# ---------------------------------------------------------------------------
# Layered, model-agnostic action parser
#
# inference.py's own _parse_action only recognizes strict
# <priority>...</priority>-style XML and silently defaults to
# priority="low", category="spam", route="trash" on ANY mismatch -- no
# logging, no distinction between "model answered in a different format"
# and "the LLM call itself failed and _call_llm swallowed the exception,
# returning ''". That silence is exactly what made an exhausted API key
# and a model that doesn't reliably emit XML produce the identical
# symptom (see WAREHOUSE.md).
#
# This parser is layered, tried top to bottom per field, first match
# wins:
#   1. Strict XML          -- identical regexes to inference.py, so a
#                              fully XML-compliant model parses exactly
#                              the same either way.
#   2. "Key: value" lines   -- covers "Priority: high" / "**Priority:**
#                              high" style answers.
#   3. Loose whole-word     -- last resort: scan the raw text for any
#      search                 valid token as a whole word.
# action_plan / threat_report are only ever read from the strict XML tag
# (their payload is JSON; loosely scanning prose for a JSON blob isn't
# safe), so a model that skips that tag gets a real, visible zero from
# the grader instead of an invented value.
#
# Every fallback -- and a genuinely empty raw_text, which is the
# signature of a swallowed _call_llm exception -- prints a warning to
# stderr, and the caller gets back enough diagnostic info to log it to
# telemetry too, so this failure mode is visible in the JSONL/warehouse,
# not just a live terminal.
# ---------------------------------------------------------------------------

from inference import (  # noqa: E402
    EmailTriageAction,
    VALID_PRIORITIES, VALID_CATEGORIES, VALID_ROUTES,
    _PRIORITY_RE, _CATEGORY_RE, _ROUTE_RE, _ACTION_PLAN_RE, _THREAT_RE,
)

_KV_TEMPLATE = r"(?:\*\*)?\s*{field}\s*(?:\*\*)?\s*[:\-]\s*(?:\*\*)?\s*([A-Za-z_ ]+)"
_KV_PRIORITY_RE = re.compile(_KV_TEMPLATE.format(field="priority"), re.IGNORECASE)
_KV_CATEGORY_RE = re.compile(_KV_TEMPLATE.format(field="category"), re.IGNORECASE)
_KV_ROUTE_RE    = re.compile(_KV_TEMPLATE.format(field="route"),    re.IGNORECASE)


def _loose_word_search(text: str, valid_values) -> Optional[str]:
    """Last resort: earliest valid token that appears as a whole word."""
    lower = text.lower()
    best_pos, best_val = None, None
    for val in valid_values:
        m = re.search(rf"\b{re.escape(val)}\b", lower)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos, best_val = m.start(), val
    return best_val


def _parse_field(text: str, xml_re, kv_re, valid_values, default: str):
    """Try XML -> Key:value -> loose search -> default. Returns (value, layer)."""
    m = xml_re.search(text)
    if m and m.group(1).strip().lower() in valid_values:
        return m.group(1).strip().lower(), "xml"

    m = kv_re.search(text)
    if m and m.group(1).strip().lower() in valid_values:
        return m.group(1).strip().lower(), "kv"

    val = _loose_word_search(text, valid_values)
    if val:
        return val, "loose"

    return default, "default"


def _parse_action_lenient(raw_text: str):
    """
    Returns (EmailTriageAction, diagnostics) where diagnostics is a dict:
        parse_ok      -- False if ANY field had to fall back to a hard
                          default (i.e. this row should NOT be trusted as
                          a real model classification)
        empty_response -- True if raw_text was empty/whitespace, the
                          signature of a swallowed _call_llm exception
        raw_snippet   -- first 200 chars of raw_text, for telemetry
    """
    if not raw_text or not raw_text.strip():
        print(
            "[parser] WARNING: empty LLM response. This is usually a "
            "swallowed exception in _call_llm (rate limit, auth, network) "
            "or a reasoning model that spent its whole token budget on "
            "hidden reasoning. Run `python diagnose_llm.py` before "
            "trusting any output from this run.",
            file=sys.stderr,
        )
        action = EmailTriageAction(priority="low", category="spam", route="trash",
                                    action_plan=None, threat_report=None)
        return action, {"parse_ok": False, "empty_response": True, "raw_snippet": ""}

    priority, p_layer = _parse_field(raw_text, _PRIORITY_RE, _KV_PRIORITY_RE, VALID_PRIORITIES, "low")
    category, c_layer = _parse_field(raw_text, _CATEGORY_RE, _KV_CATEGORY_RE, VALID_CATEGORIES, "spam")
    route,    r_layer = _parse_field(raw_text, _ROUTE_RE,    _KV_ROUTE_RE,    VALID_ROUTES,    "trash")

    parse_ok = "default" not in (p_layer, c_layer, r_layer)
    snippet = raw_text.strip().replace("\n", " ")[:200]

    if not parse_ok:
        print(
            f"[parser] WARNING: fell back to a default value "
            f"(priority via {p_layer}, category via {c_layer}, route via {r_layer}). "
            f"Raw text snippet: {snippet!r}",
            file=sys.stderr,
        )
    elif p_layer != "xml" or c_layer != "xml" or r_layer != "xml":
        print(
            f"[parser] NOTE: non-XML format recovered "
            f"(priority via {p_layer}, category via {c_layer}, route via {r_layer}).",
            file=sys.stderr,
        )

    ap = _ACTION_PLAN_RE.search(raw_text)
    tr = _THREAT_RE.search(raw_text)
    action = EmailTriageAction(
        priority=priority,
        category=category,
        route=route,
        action_plan=ap.group(1).strip() if ap else None,
        threat_report=tr.group(1).strip() if tr else None,
    )
    return action, {"parse_ok": parse_ok, "empty_response": False, "raw_snippet": snippet}


async def run_episode(llm_client: OpenAI, task: "inference.TaskConfig", episode_num: int) -> None:
    """
    Run ONE full multi-email episode for `task` over a single persistent
    WebSocket session: reset() once, then step() through every email the
    environment hands back -- with real continuity, so streak/dependency/
    coherence bonuses can actually fire, and the ground truth used for
    grading is guaranteed to be for the same email the agent was shown.
    """
    async with EmailTriageEnv(base_url=inference.SERVER_URL) as env:
        result = await env.reset()
        obs = result.observation
        state = await env.state()
        episode_id = getattr(state, "episode_id", None)

        history: List[str] = []
        rewards: List[float] = []
        step = 0

        while True:
            step += 1

            # Ground truth for the email we're ABOUT to grade -- same shape
            # inference.py's run_task() uses (see _extract_graded_truth),
            # plus is_phishing, which _grade_threat_assessment reads
            # directly and run_task()'s own reset_obs_data never included.
            reset_obs_data = {
                "true_priority":        obs.true_priority,
                "true_category":        obs.true_category,
                "true_route":           obs.true_route,
                "is_business_critical": obs.is_business_critical,
                "is_phishing":          obs.is_phishing,
                "metadata":             obs.metadata,
            }
            email_id = obs.email_id

            _llm_start = time.monotonic()
            raw_text = await _call_llm_with_retry(llm_client, task.system_prompt, obs, step, history)
            llm_latency_ms = (time.monotonic() - _llm_start) * 1000.0
            await asyncio.sleep(_REQUEST_PACING_DELAY)  # small proactive gap, cheaper than tripping the limit
            action, parse_diag = _parse_action_lenient(raw_text)
            action_str = f"priority={action.priority},category={action.category},route={action.route}"
            if not parse_diag["parse_ok"]:
                action_str += " [FALLBACK]"

            step_result = await env.step(action)
            obs = step_result.observation
            done = step_result.done

            task_reward = task.grader(action, reset_obs_data)
            task_reward = max(0.01, min(0.99, task_reward))
            rewards.append(task_reward)

            print(
                f"[{task.name} ep{episode_num} step{step}] {action_str} "
                f"-> env_reward={step_result.reward} task_reward={task_reward:.2f} done={done}"
            )

            try:
                telemetry.event_sink.log_client_step(
                    model_name=inference.MODEL_NAME,
                    task=task.name,
                    step=step,
                    email_id=email_id,
                    predicted_priority=action.priority,
                    predicted_category=action.category,
                    predicted_route=action.route,
                    action_plan=action.action_plan,
                    threat_report=action.threat_report,
                    task_reward=task_reward,
                    done=done,
                    llm_latency_ms=llm_latency_ms,
                    session_id=episode_id,
                    error=(
                        "empty_llm_response" if parse_diag["empty_response"]
                        else "parser_fallback" if not parse_diag["parse_ok"]
                        else None
                    ),
                    parse_ok=parse_diag["parse_ok"],
                    raw_response_snippet=parse_diag["raw_snippet"],
                )
            except Exception as exc:  # noqa: BLE001 -- fail open, never break a run
                print(f"[telemetry] WARNING: log_client_step failed: {exc}", file=sys.stderr)

            history.append(f"Step {step}: {action_str} -> reward={task_reward:.2f}")

            if done or step >= inference.MAX_STEPS:
                break

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        print(
            f"=== {task.name} episode {episode_num}: {len(rewards)} steps, "
            f"avg_task_reward={avg_reward:.3f} ===\n"
        )


async def main(n_episodes: int, task_filter: Optional[List[str]]) -> None:
    llm_client = OpenAI(base_url=inference.API_BASE_URL, api_key=inference.HF_TOKEN)
    tasks = inference.TASKS
    if task_filter:
        wanted = set(task_filter)
        tasks = [t for t in tasks if t.name in wanted]
        missing = wanted - {t.name for t in tasks}
        if missing:
            print(f"WARNING: unknown task name(s), skipping: {sorted(missing)}", file=sys.stderr)

    failed_episodes: List[tuple] = []

    for task in tasks:
        for ep in range(1, n_episodes + 1):
            for attempt in range(1, _MAX_EPISODE_CONNECTION_RETRIES + 1):
                try:
                    await run_episode(llm_client, task, ep)
                    break
                except (websockets.exceptions.ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
                    print(
                        f"[connection] WARNING: {task.name} ep{ep} lost its WebSocket "
                        f"connection (attempt {attempt}/{_MAX_EPISODE_CONNECTION_RETRIES}): "
                        f"{exc!r}",
                        file=sys.stderr,
                    )
                    if attempt == _MAX_EPISODE_CONNECTION_RETRIES:
                        print(
                            f"[connection] ERROR: {task.name} ep{ep} failed after "
                            f"{_MAX_EPISODE_CONNECTION_RETRIES} attempts -- skipping it "
                            f"and continuing with the rest of the run.",
                            file=sys.stderr,
                        )
                        failed_episodes.append((task.name, ep))
                    else:
                        print(
                            f"[connection] Retrying with a fresh connection in "
                            f"{_EPISODE_RETRY_DELAY}s (this episode restarts from step 1; "
                            f"steps already logged before the drop stay in the telemetry)...",
                            file=sys.stderr,
                        )
                        await asyncio.sleep(_EPISODE_RETRY_DELAY)

    if failed_episodes:
        print(
            f"\n=== {len(failed_episodes)} episode(s) could not be completed after "
            f"retries and were skipped: {failed_episodes} ===",
            file=sys.stderr,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full multi-email episodes over the WebSocket client.")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes to run per task (default: 1)")
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Comma-separated task names to run (default: all TASKS from inference.py)",
    )
    args = parser.parse_args()
    task_filter = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    asyncio.run(main(args.episodes, task_filter))
    
    