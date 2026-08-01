# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Telemetry: fail-open JSONL event logging for the data-engineering pivot.

Import `event_sink` and call `log_env_step(...)` / `log_client_step(...)`.
Zero third-party dependencies by design, so this package can be imported
from the core RL server (server/Email_RL_environment.py) without adding
anything to requirements.txt. See WAREHOUSE.md for the full pipeline.
"""

from . import event_sink

__all__ = ["event_sink"]
