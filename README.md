# Battlecard Review App

## Archived Skill Snapshot

This repo now includes a local archive of the legacy Google Slides generation skill:

- `archive/skills/generate-battlecard_2026-01-14/`
- Source: `compete_automation/battlecards-app/1st_principles_go_products/.claude/skills/generate-battlecard/`

Why it was copied:
- To preserve the exact slide-generation implementation that produced the reference deck format.
- To provide a reproducible baseline for comparing this app's Pass 1/Pass 2 prompt versions against the older pipeline.
- To keep an internal fallback copy if the source repository evolves or removes files.

## Autoscaling Dev Utilities

- `app.local.yaml`: run-local entrypoint using `.venv/bin/python`.
- `scripts/migrate_provisioned_to_autoscaling.sh`: migrate `public` tables from a provisioned Lakebase database instance to an autoscaling endpoint using `pg_dump`/`pg_restore` plus row-count verification.
- `scripts/run_local_autoscaling_dev.sh`: run `databricks apps run-local` against a specific autoscaling endpoint by resolving and exporting `PGHOST`.
