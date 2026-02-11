# AGENTS.md

## Archived Skill Snapshot

The legacy Google Slides battlecard generation skill was copied into this repo as an archive:

- `archive/skills/generate-battlecard_2026-01-14/`
- Source: `compete_automation/battlecards-app/1st_principles_go_products/.claude/skills/generate-battlecard/`

Why this was archived locally:
- Preserve the exact scripts that produced the reference deck format (especially the L200 table layout).
- Keep a stable, versioned baseline for prompt/rendering comparisons with this app's Pass 1/Pass 2 workflow.
- Avoid future regressions caused by upstream changes in another repository.

## Run Local App

### Prerequisites
- Databricks CLI installed (`databricks --version`)
- `uv` installed (`uv --version`)
- Valid Databricks auth/profile (default used here: `fe-vm-pmt`)
- `.env` file present with `EXA_API_KEY` (required because `app.yaml` uses `valueFrom: exa_api_key`)

### Start Locally (recommended)
```bash
cd /Users/tahir.fayyaz/databricks-dev/battle-station-dev/battlecard-review-app
set -a
source .env
set +a
databricks apps run-local \
  --entry-point app.yaml \
  --profile fe-vm-pmt \
  --prepare-environment
```

### Start Locally with explicit ports
```bash
cd /Users/tahir.fayyaz/databricks-dev/battle-station-dev/battlecard-review-app
set -a
source .env
set +a
databricks apps run-local \
  --entry-point app.yaml \
  --profile fe-vm-pmt \
  --prepare-environment \
  --app-port 5055 \
  --port 5056
```

- App proxy URL will be `http://localhost:5056`
- Flask app process listens on `5055` in this example

### Common Issues
- Error: `EXA_API_KEY ... can't be resolved locally`
  - Fix: export `EXA_API_KEY` (or `source .env`) before `run-local`.
- Error: `ModuleNotFoundError` (for example `tiktoken`)
  - Fix: use `--prepare-environment` so Databricks CLI installs dependencies.
- Browser auth/token issues
  - Re-login: `databricks auth login --host https://fe-vm-pmt.cloud.databricks.com --profile fe-vm-pmt`

### Stop
- Press `Ctrl+C` in the terminal where `run-local` is running.

## Generate Battlecard Locally

### UI flow (recommended)
1. Start local app (see commands above).
2. Open `http://localhost:5056/workflow/new`.
3. Fill:
   - `Competitor Name` (example: `Microsoft Fabric`)
   - `Product Area` (default: `Data Platform`)
   - `Model`, `Diffs per Category`, and prompt template versions
4. Click `Start Workflow`.
5. Step 1 (`Generate Directive`):
   - Upload directive context and click `Generate Directive`, or click `Skip (use default)`.
6. Step 2 (`Upload Old Battlecards`):
   - Upload old battlecard and click `Extract Content`, or click `Skip`.
7. Step 3 (`Classify Product Categories`):
   - Choose `Core Product`, `Cross-Platform`, or `Skip` per row.
   - Click `Save & Continue`.
8. Step 4 (`Generate Key Differentiators`):
   - Click `Generate Key Differentiators`.
   - Review output and click `Approve & Continue`.
9. Step 5 (`Generate Key Diff Claims`):
   - Click `Generate Claims`.
   - Review output and click `Approve & Continue`.
10. Step 6 (`Fact Check Claims`):
   - Click `Run Fact Check`.
11. Step 7 (`Generate Google Slides`):
   - Click to generate slides when ready.

### Useful local API checks
Check workflow status:
```bash
curl -s http://localhost:5056/api/workflow/<SESSION_ID>/status | jq
```

Check context window artifacts:
```bash
curl -s http://localhost:5056/api/workflow/<SESSION_ID>/context | jq
```

### Database verification for one workflow session
```bash
cd /Users/tahir.fayyaz/databricks-dev/battle-station-dev/battlecard-review-app
.venv/bin/python - <<'PY'
import json
from sqlalchemy import text
from app import ENGINE

session_id = "<SESSION_ID>"
with ENGINE.begin() as conn:
    row = conn.execute(
        text("SELECT generation_id FROM workflow_sessions WHERE session_id::text = :sid"),
        {"sid": session_id},
    ).mappings().first()
    gid = row["generation_id"] if row else None
    out = {"session_id": session_id, "generation_id": gid}
    if gid:
        out["key_differentiators"] = int(conn.execute(text("SELECT COUNT(*) FROM key_differentiators WHERE generation_id = :g"), {"g": gid}).scalar() or 0)
        out["claims"] = int(conn.execute(text("SELECT COUNT(*) FROM claims WHERE generation_id = :g"), {"g": gid}).scalar() or 0)
        out["fact_checks"] = int(conn.execute(text("""SELECT COUNT(*) FROM fact_checks fc
            JOIN evidence e ON e.evidence_id = fc.evidence_id
            JOIN claims c ON c.claim_id = e.claim_id
            WHERE c.generation_id = :g"""), {"g": gid}).scalar() or 0)
print(json.dumps(out, indent=2))
PY
```

## Deploy Live App
```bash
cd /Users/tahir.fayyaz/databricks-dev/battle-station-dev/battlecard-review-app
databricks apps deploy --target dev -p fe-vm-pmt
```

### View live logs
```bash
databricks apps logs battlecards-review-pg -p fe-vm-pmt --tail-lines 200
```

## Autoscaling Dev Branch (Run-Local)

Use this flow when you want local app testing against a Lakebase Autoscaling Postgres branch instead of the provisioned instance.

### Local entrypoint
- `app.local.yaml` uses `command: [".venv/bin/python", "app.py"]`.
- This avoids local `python` (Python 2) incompatibility with `app.py`.

### Create/refresh autoscaling dev branch and endpoint
- Example resources used:
  - project: `projects/battlecards-review-dev`
  - branch: `projects/battlecards-review-dev/branches/dev`
  - endpoint: `projects/battlecards-review-dev/branches/dev/endpoints/rw`

### Migrate provisioned data to autoscaling with pg_dump/pg_restore
```bash
cd /Users/tahir.fayyaz/databricks-dev/battle-station-dev/battlecard-review-app
./scripts/migrate_provisioned_to_autoscaling.sh \
  --profile fe-vm-pmt \
  --source-instance battlestation \
  --target-endpoint projects/battlecards-review-dev/branches/main/endpoints/rw
```

Notes:
- Script exports only `public` base tables to avoid Databricks-managed internal schema/event trigger conflicts.
- Script verifies source/target row counts for all exported tables.

### Run local app against autoscaling dev branch
```bash
cd /Users/tahir.fayyaz/databricks-dev/battle-station-dev/battlecard-review-app
./scripts/run_local_autoscaling_dev.sh \
  --profile fe-vm-pmt \
  --endpoint projects/battlecards-review-dev/branches/dev/endpoints/rw \
  --entry-point app.local.yaml \
  --app-port 5060 \
  --port 5061
```

### Validate run-local is using dev branch
1. Create a workflow via local app (`http://localhost:5061/workflow/new`).
2. Query the new `session_id` on both branches.
3. Expect row exists in `dev`, not `main`.
