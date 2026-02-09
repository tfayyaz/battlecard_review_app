#!/usr/bin/env python3
"""
Backup Lakebase/Postgres tables to local JSONL files.

This script exports all tables from the `public` schema (or a selected subset),
creates a metadata manifest, and optionally archives everything into a `.tar.gz`.

Example:
  .venv/bin/python scripts/backup_lakebase.py --archive
"""

from __future__ import annotations

import argparse
import json
import tarfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import sys

from sqlalchemy import text

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import ENGINE


@dataclass
class TableExportMeta:
    table: str
    row_count: int
    file: str
    started_at_utc: str
    completed_at_utc: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_tables(schema: str = "public") -> list[str]:
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT table_name "
                "FROM information_schema.tables "
                "WHERE table_schema = :schema "
                "ORDER BY table_name"
            ),
            {"schema": schema},
        ).scalars().all()
    return list(rows)


def count_rows(table: str) -> int:
    with ENGINE.begin() as conn:
        return int(
            conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar() or 0
        )


def export_table_jsonl(table: str, out_file: Path, fetch_size: int = 1000) -> TableExportMeta:
    started = utc_now_iso()
    total_rows = 0

    with ENGINE.connect() as conn, out_file.open("w", encoding="utf-8") as f:
        result = conn.execution_options(stream_results=True).execute(
            text(f'SELECT * FROM "{table}"')
        )
        while True:
            batch = result.mappings().fetchmany(fetch_size)
            if not batch:
                break
            for row in batch:
                total_rows += 1
                f.write(json.dumps(dict(row), default=str, ensure_ascii=True))
                f.write("\n")

    completed = utc_now_iso()
    return TableExportMeta(
        table=table,
        row_count=total_rows,
        file=str(out_file),
        started_at_utc=started,
        completed_at_utc=completed,
    )


def build_archive(source_dir: Path, archive_path: Path):
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name)


def parse_args():
    parser = argparse.ArgumentParser(description="Backup Lakebase/Postgres tables to local JSONL.")
    parser.add_argument(
        "--output-root",
        default="runs/db_backups",
        help="Root output directory for backup runs",
    )
    parser.add_argument(
        "--schema",
        default="public",
        help="Schema to export (default: public)",
    )
    parser.add_argument(
        "--tables",
        default="",
        help="Comma-separated table names. Empty means all tables in schema.",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Create a tar.gz archive of the backup directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"lakebase_backup_{ts}"
    data_dir = out_dir / "tables_jsonl"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.tables.strip():
        tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    else:
        tables = list_tables(schema=args.schema)

    if not tables:
        raise SystemExit("No tables found to export.")

    export_results: list[TableExportMeta] = []
    pre_counts: dict[str, int] = {}

    for table in tables:
        pre_counts[table] = count_rows(table)
        out_file = data_dir / f"{table}.jsonl"
        meta = export_table_jsonl(table, out_file)
        export_results.append(meta)
        print(f"exported {table}: {meta.row_count} rows -> {out_file}")

    manifest = {
        "created_at_utc": utc_now_iso(),
        "schema": args.schema,
        "table_count": len(tables),
        "tables": [asdict(r) for r in export_results],
        "pre_counts": pre_counts,
        "output_dir": str(out_dir),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")

    archive_path = None
    if args.archive:
        archive_path = Path(args.output_root) / f"{out_dir.name}.tar.gz"
        build_archive(out_dir, archive_path)
        print(f"archive: {archive_path}")

    final = {
        "output_dir": str(out_dir),
        "manifest": str(manifest_path),
        "archive": str(archive_path) if archive_path else None,
    }
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
