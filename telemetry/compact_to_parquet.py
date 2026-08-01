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
            table = pa.Table.from_pylist(rows)
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
