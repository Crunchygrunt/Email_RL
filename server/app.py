# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Email Triage RL Environment.

This module creates an HTTP server that exposes the EmailTriageEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    POST /reset    -- Reset the environment and receive the first email
    POST /step     -- Submit a triage action and receive the next email + reward
    GET  /state    -- Get current environment state (episode_id, step_count)
    GET  /schema   -- Get action / observation JSON schemas
    WS   /ws       -- WebSocket endpoint for persistent sessions

Usage:
    # Development (auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or via pyproject entry point:
    uv run --project . server
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with:\n    uv sync\n"
    ) from e

try:
    from ..models import EmailTriageAction, EmailTriageObservation
    from .Email_RL_environment import EmailTriageEnvironment
except ImportError:
    from models import EmailTriageAction, EmailTriageObservation
    from server.Email_RL_environment import EmailTriageEnvironment


app = create_app(
    EmailTriageEnvironment,
    EmailTriageAction,
    EmailTriageObservation,
    env_name="Email_RL",
    max_concurrent_envs=10,
)


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Email Triage RL Environment Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
