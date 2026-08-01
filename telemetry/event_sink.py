# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
telemetry/event_sink.py -- fail-open, dependency-free JSONL event logging.

Design constraints (see WAREHOUSE.md for the full rationale):

- Zero third-party dependencies. This module is imported directly by the
  core server (server/Email_RL_environment.py) and by inference.py, and
  must never require adding anything to requirements.txt or introduce an
  import that could break a graded run.
- Fail OPEN. Any logging failure (disk full, bad permissions, unexpected
  argument, whatever) is caught, a one-line warning is printed to stderr,
  and execution continues. A telemetry bug must never turn into a broken
  RL step or a broken eval run.
- One JSONL file per (stream, UTC date), append-only:
      data/raw/env_steps/dt=YYYY-MM-DD/events.jsonl
      data/raw/client_steps/dt=YYYY-MM-DD/events.jsonl
- One threading.Lock per stream. This assumes a single process. Under
  `uvicorn --workers > 1` each worker gets its own lock and its own file
  handle -- writes are not coordinated across processes. Fine for local /
  single-worker use; see WAREHOUSE.md limitations if you scale workers up.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Root of the raw landing zone. Overridable via env var so tests / the
# compaction script / custom deployments don't need to touch code.
# _DATA_ROOT = Path(os.environ.get("EMAIL_RL_TELEMETRY_ROOT", "data/raw"))
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"

_DATA_ROOT = Path(
    os.environ.get("EMAIL_RL_TELEMETRY_ROOT", str(DEFAULT_DATA_ROOT))
).expanduser().resolve()

_LOCKS: Dict[str, threading.Lock] = {
    "env_steps": threading.Lock(),
    "client_steps": threading.Lock(),
}

print("=" * 60)
print("Loaded event_sink from:", __file__)
print("=" * 60)

def _event_path(stream: str) -> Path:
    dt = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _DATA_ROOT / stream / f"dt={dt}" / "events.jsonl"


def _write_event(stream: str, event: Dict[str, Any]) -> None:
    """Append one JSON line to the stream's current-day file. Never raises."""
    lock = _LOCKS.setdefault(stream, threading.Lock())
    try:
        path = _event_path(stream)
        line = json.dumps(event, default=str)
        with lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 -- intentionally broad, fail-open by design
        print(f"[telemetry] WARNING: failed to log {stream} event: {exc}", file=sys.stderr)


def log_env_step(
    *,
    episode_id: str,
    step: int,
    email_id: str,
    predicted_priority: str,
    predicted_category: str,
    predicted_route: str,
    true_priority: str,
    true_category: str,
    true_route: str,
    is_business_critical: bool,
    is_phishing: bool,
    cluster_id: Optional[str],
    is_escalation: bool,
    priority_ok: bool,
    category_ok: bool,
    route_ok: bool,
    is_perfect: bool,
    base_score: float,
    urgency_multiplier: float,
    reward_components: Dict[str, float],
    shaped_reward: float,
    current_streak: int,
    done: bool,
    stateless_http_mode: bool,
    emails_remaining: int,
) -> None:
    """
    Log one EmailTriageEnvironment.step() call.

    Called server-side, inside step(), right before the observation is
    returned -- same process, before the HTTP boundary, so every reward
    component is available as a local variable and nothing needs to
    round-trip over the wire.
    """
    print(">>> log_env_step called")
    
    event = {
        "event_type": "env_step",
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "episode_id": episode_id,
        "step": step,
        "email_id": email_id,
        "predicted_priority": predicted_priority,
        "predicted_category": predicted_category,
        "predicted_route": predicted_route,
        "true_priority": true_priority,
        "true_category": true_category,
        "true_route": true_route,
        "is_business_critical": is_business_critical,
        "is_phishing": is_phishing,
        "cluster_id": cluster_id,
        "is_escalation": is_escalation,
        "priority_ok": priority_ok,
        "category_ok": category_ok,
        "route_ok": route_ok,
        "is_perfect": is_perfect,
        "base_score": base_score,
        "urgency_multiplier": urgency_multiplier,
        "reward_components": reward_components,
        "shaped_reward": shaped_reward,
        "current_streak": current_streak,
        "done": done,
        "stateless_http_mode": stateless_http_mode,
        "emails_remaining": emails_remaining,
    }
    print(">>> Writing env step")
    _write_event("env_steps", event)
    print(">>> Done")


def log_client_step(
    *,
    model_name: str,
    task: str,
    step: int,
    email_id: str,
    predicted_priority: str,
    predicted_category: str,
    predicted_route: str,
    action_plan: Optional[str],
    threat_report: Optional[str],
    task_reward: float,
    done: bool,
    llm_latency_ms: float,
    session_id: Optional[str] = None,
    error: Optional[str] = None,
    parse_ok: Optional[bool] = None,
    raw_response_snippet: Optional[str] = None,
) -> None:
    """
    Log one inference.py run_task() step.

    Called client-side, right around the existing `env.step(action)` call.
    `email_id` should be the id of the email the agent just responded to
    (i.e. from the reset() observation, since run_task() resets before
    every graded email -- see WAREHOUSE.md for why most episodes are
    currently 1 email long).

    `parse_ok` / `raw_response_snippet`: set by run_episodes.py's
    `_parse_action_lenient()`. `parse_ok=False` means at least one field
    hit its hard default (priority=low/category=spam/route=trash) rather
    than being genuinely parsed from the model's output -- these rows
    should be excluded from accuracy analysis, not counted as real
    predictions. `raw_response_snippet` (first ~200 chars of the raw LLM
    text) makes a future occurrence of this diagnosable straight from the
    JSONL/warehouse instead of requiring a live rerun with stderr open.
    """
    event = {
        "event_type": "client_step",
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "task": task,
        "step": step,
        "email_id": email_id,
        "predicted_priority": predicted_priority,
        "predicted_category": predicted_category,
        "predicted_route": predicted_route,
        "action_plan": action_plan,
        "threat_report": threat_report,
        "task_reward": task_reward,
        "done": done,
        "llm_latency_ms": llm_latency_ms,
        "session_id": session_id,
        "error": error,
        "parse_ok": parse_ok,
        "raw_response_snippet": raw_response_snippet,
    }
    _write_event("client_steps", event)