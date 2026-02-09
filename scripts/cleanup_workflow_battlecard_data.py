#!/usr/bin/env python3
"""
Cleanup workflow/battlecard/review data while preserving product category tables.

Default behavior is dry-run. Use --execute to apply deletions.

This script intentionally does NOT delete:
  - product_category_catalog
  - product_mappings
  - prompt templates/versions
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

from sqlalchemy import text

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import ENGINE


@dataclass
class Scope:
    competitor_name: str | None
    product_area: str | None
    session_ids: list[str]
    generation_ids: list[int]


def parse_args():
    p = argparse.ArgumentParser(description="Cleanup workflow/battlecard data (preserve product categories).")
    p.add_argument("--competitor", default=None, help="Filter by workflow_sessions.competitor_name")
    p.add_argument("--product-area", default=None, help="Filter by workflow_sessions.product_area")
    p.add_argument("--session-ids", default="", help="Comma-separated workflow session UUIDs")
    p.add_argument("--all", action="store_true", help="Target all workflow/battlecard data")
    p.add_argument("--list-contexts", action="store_true", help="Print available competitor/product contexts")
    p.add_argument("--archive-first", action="store_true", help="Run backup_lakebase.py --archive before cleanup")
    p.add_argument("--execute", action="store_true", help="Apply deletion (default is dry-run)")
    return p.parse_args()


def list_contexts():
    q = text(
        """
        SELECT ws.competitor_name, ws.product_area,
               COUNT(DISTINCT ws.session_id) AS workflows,
               COUNT(DISTINCT ws.generation_id) AS generations,
               MIN(ws.created_at) AS first_seen,
               MAX(ws.created_at) AS last_seen
        FROM workflow_sessions ws
        GROUP BY ws.competitor_name, ws.product_area
        ORDER BY last_seen DESC
        """
    )
    with ENGINE.begin() as conn:
        rows = conn.execute(q).mappings().all()
    return [dict(r) for r in rows]


def resolve_scope(args) -> Scope:
    explicit_sessions = [s.strip() for s in args.session_ids.split(",") if s.strip()]
    filters = []
    params = {}

    if not args.all:
        if args.competitor:
            filters.append("ws.competitor_name = :comp")
            params["comp"] = args.competitor
        if args.product_area:
            filters.append("ws.product_area = :area")
            params["area"] = args.product_area
        if explicit_sessions:
            filters.append("ws.session_id::text = ANY(:session_ids)")
            params["session_ids"] = explicit_sessions

        if not filters:
            raise ValueError("Specify a scope via --competitor/--product-area/--session-ids, or use --all.")

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    q = text(
        f"""
        SELECT ws.session_id::text AS session_id, ws.generation_id
        FROM workflow_sessions ws
        {where_clause}
        ORDER BY ws.created_at
        """
    )
    with ENGINE.begin() as conn:
        rows = conn.execute(q, params).mappings().all()

    session_ids = [r["session_id"] for r in rows]
    generation_ids = sorted({int(r["generation_id"]) for r in rows if r["generation_id"] is not None})
    return Scope(
        competitor_name=args.competitor,
        product_area=args.product_area,
        session_ids=session_ids,
        generation_ids=generation_ids,
    )


def count_for_scope(scope: Scope) -> dict[str, int]:
    if not scope.session_ids and not scope.generation_ids:
        return {}

    counts = {}
    with ENGINE.begin() as conn:
        if scope.session_ids:
            counts["workflow_sessions"] = int(
                conn.execute(
                    text("SELECT COUNT(*) FROM workflow_sessions WHERE session_id::text = ANY(:sids)"),
                    {"sids": scope.session_ids},
                ).scalar() or 0
            )
            for table in ("workflow_steps", "workflow_artifacts", "agent_turns", "session_category_selections", "pass2_debug_logs"):
                counts[table] = int(
                    conn.execute(
                        text(f'SELECT COUNT(*) FROM "{table}" WHERE session_id::text = ANY(:sids)'),
                        {"sids": scope.session_ids},
                    ).scalar() or 0
                )
            counts["eval_run_results_by_session"] = int(
                conn.execute(
                    text("SELECT COUNT(*) FROM eval_run_results WHERE session_id::text = ANY(:sids)"),
                    {"sids": scope.session_ids},
                ).scalar() or 0
            )

        if scope.generation_ids:
            gids = scope.generation_ids
            counts["battlecard_generations"] = int(
                conn.execute(
                    text("SELECT COUNT(*) FROM battlecard_generations WHERE generation_id = ANY(:gids)"),
                    {"gids": gids},
                ).scalar() or 0
            )
            counts["key_differentiators"] = int(
                conn.execute(
                    text("SELECT COUNT(*) FROM key_differentiators WHERE generation_id = ANY(:gids)"),
                    {"gids": gids},
                ).scalar() or 0
            )
            counts["claims"] = int(
                conn.execute(
                    text("SELECT COUNT(*) FROM claims WHERE generation_id = ANY(:gids)"),
                    {"gids": gids},
                ).scalar() or 0
            )
            counts["claim_detail_items"] = int(
                conn.execute(
                    text(
                        "SELECT COUNT(*) FROM claim_detail_items di "
                        "JOIN claims c ON c.claim_id = di.claim_id "
                        "WHERE c.generation_id = ANY(:gids)"
                    ),
                    {"gids": gids},
                ).scalar() or 0
            )
            counts["evidence"] = int(
                conn.execute(
                    text(
                        "SELECT COUNT(*) FROM evidence e "
                        "JOIN claims c ON c.claim_id = e.claim_id "
                        "WHERE c.generation_id = ANY(:gids)"
                    ),
                    {"gids": gids},
                ).scalar() or 0
            )
            counts["fact_checks"] = int(
                conn.execute(
                    text(
                        "SELECT COUNT(*) FROM fact_checks fc "
                        "JOIN evidence e ON e.evidence_id = fc.evidence_id "
                        "JOIN claims c ON c.claim_id = e.claim_id "
                        "WHERE c.generation_id = ANY(:gids)"
                    ),
                    {"gids": gids},
                ).scalar() or 0
            )
            counts["human_reviews"] = int(
                conn.execute(
                    text("SELECT COUNT(*) FROM human_reviews WHERE generation_id = ANY(:gids)"),
                    {"gids": gids},
                ).scalar() or 0
            )
            counts["eval_datasets_by_generation"] = int(
                conn.execute(
                    text("SELECT COUNT(*) FROM eval_datasets WHERE source_generation_id = ANY(:gids)"),
                    {"gids": gids},
                ).scalar() or 0
            )
    return counts


def run_backup():
    cmd = [str(ROOT_DIR / ".venv" / "bin" / "python"), str(ROOT_DIR / "scripts" / "backup_lakebase.py"), "--archive"]
    subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))


def execute_cleanup(scope: Scope) -> dict[str, int]:
    if not scope.session_ids and not scope.generation_ids:
        return {}

    deleted: dict[str, int] = {}
    with ENGINE.begin() as conn:
        sids = scope.session_ids
        gids = scope.generation_ids

        if sids:
            # Session-scoped tables
            for table in ("pass2_debug_logs", "agent_turns", "workflow_artifacts", "workflow_steps", "session_category_selections"):
                rc = conn.execute(
                    text(f'DELETE FROM "{table}" WHERE session_id::text = ANY(:sids)'),
                    {"sids": sids},
                ).rowcount or 0
                deleted[table] = int(rc)

            # Optional eval linkage (keep dataset tables but remove run results tied to these sessions)
            rc = conn.execute(
                text("DELETE FROM eval_run_results WHERE session_id::text = ANY(:sids)"),
                {"sids": sids},
            ).rowcount or 0
            deleted["eval_run_results"] = int(rc)

            rc = conn.execute(
                text("DELETE FROM workflow_sessions WHERE session_id::text = ANY(:sids)"),
                {"sids": sids},
            ).rowcount or 0
            deleted["workflow_sessions"] = int(rc)

        if gids:
            # Break references that can block generation deletion
            conn.execute(
                text("UPDATE battlecard_generations SET previous_generation_id = NULL WHERE previous_generation_id = ANY(:gids)"),
                {"gids": gids},
            )
            conn.execute(
                text("UPDATE eval_datasets SET source_generation_id = NULL WHERE source_generation_id = ANY(:gids)"),
                {"gids": gids},
            )

            # Delete generation-scoped content in FK-safe order
            rc = conn.execute(
                text("DELETE FROM human_reviews WHERE generation_id = ANY(:gids)"),
                {"gids": gids},
            ).rowcount or 0
            deleted["human_reviews"] = int(rc)

            rc = conn.execute(
                text(
                    "DELETE FROM fact_checks fc WHERE fc.evidence_id IN ("
                    "SELECT e.evidence_id FROM evidence e "
                    "JOIN claims c ON c.claim_id = e.claim_id "
                    "WHERE c.generation_id = ANY(:gids))"
                ),
                {"gids": gids},
            ).rowcount or 0
            deleted["fact_checks"] = int(rc)

            rc = conn.execute(
                text(
                    "DELETE FROM evidence e USING claims c "
                    "WHERE e.claim_id = c.claim_id AND c.generation_id = ANY(:gids)"
                ),
                {"gids": gids},
            ).rowcount or 0
            deleted["evidence"] = int(rc)

            rc = conn.execute(
                text(
                    "DELETE FROM claim_detail_items di USING claims c "
                    "WHERE di.claim_id = c.claim_id AND c.generation_id = ANY(:gids)"
                ),
                {"gids": gids},
            ).rowcount or 0
            deleted["claim_detail_items"] = int(rc)

            rc = conn.execute(
                text("DELETE FROM claims WHERE generation_id = ANY(:gids)"),
                {"gids": gids},
            ).rowcount or 0
            deleted["claims"] = int(rc)

            rc = conn.execute(
                text("DELETE FROM key_differentiators WHERE generation_id = ANY(:gids)"),
                {"gids": gids},
            ).rowcount or 0
            deleted["key_differentiators"] = int(rc)

            rc = conn.execute(
                text("DELETE FROM battlecard_generations WHERE generation_id = ANY(:gids)"),
                {"gids": gids},
            ).rowcount or 0
            deleted["battlecard_generations"] = int(rc)

    return deleted


def main():
    args = parse_args()
    if args.list_contexts:
        contexts = list_contexts()
        print(json.dumps(contexts, default=str, indent=2))
        return

    scope = resolve_scope(args)
    summary = {
        "requested_scope": {
            "competitor": scope.competitor_name,
            "product_area": scope.product_area,
            "session_ids": len(scope.session_ids),
            "generation_ids": len(scope.generation_ids),
        },
        "counts": count_for_scope(scope),
        "mode": "execute" if args.execute else "dry_run",
        "at_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(summary, default=str, indent=2))

    if not args.execute:
        return

    if args.archive_first:
        run_backup()

    deleted = execute_cleanup(scope)
    print(json.dumps({"deleted": deleted}, indent=2))


if __name__ == "__main__":
    main()
