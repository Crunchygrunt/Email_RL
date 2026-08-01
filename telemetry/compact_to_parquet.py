# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
telemetry/compact_to_parquet.py -- JSONL landing zone -> partitioned Parquet lake.

Usage:
    python telemetry/compact_to_parquet.py [--data-root data/raw] [--lake-root data/lake]

What it does:
    Reads every data/raw/{env_steps,client_steps}/dt=YYYY-MM-DD/events.jsonl
    file that hasn't already been compacted, serializes the nested
    `reward_components` dict (env_steps only) to a compact JSON *string*
    column -- not a pyarrow struct, since struct inference gets painful
    when keys are sometimes absent, and DuckDB's native json_extract_string()
    handles a JSON-string column natively -- and writes partitioned Parquet
    via pq.write_to_dataset(table, partition_cols=["dt"]).

Idempotency:
    Each JSONL source file, once successfully compacted (or found empty),
    is moved into a sibling `_compacted/` archive folder next to it.
    Re-running this script never re-reads, and therefore never
    double-counts, a file it already processed.

Dependencies: pyarrow only (see requirements-warehouse.txt). This is a
standalone batch job -- it is never imported by the core server, so it
does not touch the server's requirements.txt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List


import pyarrow as pa  
import pyarrow.parquet as pq  

STREAMS = ("env_steps", "client_steps")

# Fields that are nested dicts in the raw JSONL and need to become a
# compact JSON string column before they can go into Parquet.
_JSON_STRING_FIELDS = {
    "env_steps": ("reward_components",),
    "client_steps": (),
}

# Explicit per-stream Parquet schemas.
#
# Why: pa.Table.from_pylist() infers each column's type from the values
# present in THAT batch. A day's file where a field happens to be None in
# every row -- e.g. action_plan/threat_report on a spam-detection-only
# run, which never populates them -- gets typed as whatever pyarrow's
# all-null fallback is, which showed up here as INTEGER. A later day's
# file, once that field IS populated (action-orchestrator, say), gets
# typed VARCHAR -- and DuckDB's read_parquet(glob, hive_partitioning=true)
# then has to reconcile INTEGER vs VARCHAR for the same column across
# files. Pinning the schema here means every file gets the same type
# regardless of what happened to be non-null that day.
_ENV_STEPS_SCHEMA = pa.schema([
    ("event_type",           pa.string()),
    ("logged_at",            pa.string()),
    ("episode_id",           pa.string()),
    ("step",                 pa.int64()),
    ("email_id",             pa.string()),
    ("predicted_priority",   pa.string()),
    ("predicted_category",   pa.string()),
    ("predicted_route",      pa.string()),
    ("true_priority",        pa.string()),
    ("true_category",        pa.string()),
    ("true_route",           pa.string()),
    ("is_business_critical", pa.bool_()),
    ("is_phishing",          pa.bool_()),
    ("cluster_id",           pa.string()),  # also all-null on non-cluster days -- same bug, same fix
    ("is_escalation",        pa.bool_()),
    ("priority_ok",          pa.bool_()),
    ("category_ok",          pa.bool_()),
    ("route_ok",             pa.bool_()),
    ("is_perfect",           pa.bool_()),
    ("base_score",           pa.float64()),
    ("urgency_multiplier",   pa.float64()),
    ("reward_components",    pa.string()),  # JSON string -- see _prepare_rows
    ("shaped_reward",        pa.float64()),
    ("current_streak",       pa.int64()),
    ("done",                 pa.bool_()),
    ("stateless_http_mode",  pa.bool_()),
    ("emails_remaining",     pa.int64()),
    ("dt",                   pa.string()),
])

_CLIENT_STEPS_SCHEMA = pa.schema([
    ("event_type",           pa.string()),
    ("logged_at",            pa.string()),
    ("model_name",           pa.string()),
    ("task",                 pa.string()),
    ("step",                 pa.int64()),
    ("email_id",             pa.string()),
    ("predicted_priority",   pa.string()),
    ("predicted_category",   pa.string()),
    ("predicted_route",      pa.string()),
    ("action_plan",          pa.string()),  # the reported bug: pinned so it's never inferred as INTEGER
    ("threat_report",        pa.string()),  # same
    ("task_reward",          pa.float64()),
    ("done",                 pa.bool_()),
    ("llm_latency_ms",       pa.float64()),
    ("session_id",           pa.string()),
    ("error",                pa.string()),
    ("parse_ok",             pa.bool_()),
    ("raw_response_snippet", pa.string()),
    ("dt",                   pa.string()),
])

_SCHEMAS = {"env_steps": _ENV_STEPS_SCHEMA, "client_steps": _CLIENT_STEPS_SCHEMA}


def _normalize_rows(rows: List[Dict[str, Any]], schema: "pa.Schema", stream: str, path: Path) -> List[Dict[str, Any]]:
    """
    Align each row dict to exactly schema.names before handing it to
    pyarrow: fill any schema field absent from a row with None (handles
    JSONL written by an older event_sink.py that predates a field), and
    warn -- once per file, not per row -- about any keys present in the
    data but missing from the pinned schema (usually means event_sink.py
    grew a new field this schema hasn't been updated to match). Either
    way this never crashes the compaction job; it just tells you loudly
    what it dropped or defaulted.
    """
    known = set(schema.names)
    unexpected: set = set()
    normalized = []
    for row in rows:
        unexpected |= (set(row.keys()) - known)
        normalized.append({name: row.get(name) for name in schema.names})
    if unexpected:
        print(
            f"[{stream}] WARNING: {path} has field(s) not in the pinned "
            f"schema, dropped from Parquet: {sorted(unexpected)}. If these "
            f"are new, intentional telemetry fields, add them to the "
            f"_ENV_STEPS_SCHEMA / _CLIENT_STEPS_SCHEMA in this file.",
            file=sys.stderr,
        )
    return normalized


def _find_uncompacted_files(stream_dir: Path) -> List[Path]:
    """Every dt=*/events.jsonl file under stream_dir not already archived."""
    if not stream_dir.exists():
        return []
    return sorted(stream_dir.glob("dt=*/events.jsonl"))


def _load_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  WARNING: skipping malformed line {line_num} in {path}: {exc}", file=sys.stderr)
    return events


def _prepare_rows(events: List[Dict[str, Any]], dt: str, stream: str) -> List[Dict[str, Any]]:
    """Flatten nested-dict fields to compact JSON strings; attach dt partition column."""
    json_fields = _JSON_STRING_FIELDS.get(stream, ())
    rows = []
    for e in events:
        row = dict(e)
        for field in json_fields:
            if field in row and isinstance(row[field], dict):
                row[field] = json.dumps(row[field], sort_keys=True)
        row["dt"] = dt
        rows.append(row)
    return rows


def _compact_stream(data_root: Path, lake_root: Path, stream: str) -> int:
    stream_dir = data_root / stream
    files = _find_uncompacted_files(stream_dir)
    if not files:
        print(f"[{stream}] nothing to compact")
        return 0

    total_rows = 0
    for path in files:
        dt = path.parent.name.split("=", 1)[-1]  # "dt=2026-07-29" -> "2026-07-29"
        events = _load_events(path)

        if not events:
            print(f"[{stream}] {path} had 0 usable rows, archiving without writing")
        else:
            rows = _prepare_rows(events, dt, stream)
            schema = _SCHEMAS[stream]
            rows = _normalize_rows(rows, schema, stream, path)
            table = pa.Table.from_pylist(rows, schema=schema)
            pq.write_to_dataset(
                table,
                root_path=str(lake_root / stream),
                partition_cols=["dt"],
            )
            total_rows += len(rows)
            print(f"[{stream}] compacted {len(rows)} rows from {path} -> {lake_root / stream} (dt={dt})")

        # Idempotency: archive the source file so a re-run never re-reads
        # (and double-counts) it.
        archive_dir = path.parent / "_compacted"
        archive_dir.mkdir(exist_ok=True)
        shutil.move(str(path), str(archive_dir / path.name))

    return total_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact JSONL telemetry into partitioned Parquet.")
    parser.add_argument("--data-root", default="data/raw", help="JSONL landing zone root")
    parser.add_argument("--lake-root", default="data/lake", help="Parquet lake output root")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    lake_root = Path(args.lake_root)
    lake_root.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    for stream in STREAMS:
        grand_total += _compact_stream(data_root, lake_root, stream)

    print(f"\nDone. {grand_total} total rows compacted across {len(STREAMS)} streams.")


if __name__ == "__main__":
    main()
    