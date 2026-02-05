# Battlecard Review App

Postgres-backed Flask app for generating, reviewing, and managing competitive battlecards. Uses Databricks Lakebase (Postgres) for storage and Databricks Apps for deployment.

## Architecture

- **Framework**: Flask (Python 3.12)
- **Package manager**: `uv` (local dev), `requirements.txt` (Databricks Apps platform)
- **Database**: Databricks Lakebase (PostgreSQL), instance `battlestation`
- **Auth**: Databricks SDK OAuth (CLI profile locally, platform-injected creds in Apps)
- **Deploy**: Databricks Asset Bundles (DAB)

## Local Development

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html) v0.283.0+
- Valid Databricks CLI profile (`fe-vm-pmt`)

### Setup

```bash
# Install dependencies
uv sync

# Create .env from example (if needed)
cp .env.example .env
# Set DATABRICKS_PROFILE and EXA_API_KEY in .env
```

### Run Locally

Use `databricks apps run-local` which handles env injection and proxying:

```bash
databricks apps run-local --profile fe-vm-pmt --prepare-environment --app-port 5013
```

- App proxy: http://localhost:8001
- Flask server: http://localhost:5013 (direct)
- The `--prepare-environment` flag uses `uv` to install dependencies
- The `--app-port 5013` flag tells the proxy where Flask is listening

If port 5013 is already in use:
```bash
lsof -i :5013          # Find process
kill <PID>             # Kill it
```

### Alternative: Run directly with uv

```bash
uv run python app.py --port 5013
# App runs at http://localhost:5013
```

Note: Without `--port`, the app defaults to port 8000 (matching Databricks Apps platform expectations). Use `--port 5013` locally to avoid conflicts.

## Deploy to Databricks Apps

### 1. Validate

```bash
databricks bundle validate -t dev
```

### 2. Deploy

```bash
databricks bundle deploy -t dev
```

### 3. Run (required after deploy)

```bash
databricks bundle run battlecards_review -t dev
```

`bundle deploy` only uploads files. `bundle run` is required to start/restart the app with new code.

### Check Status

```bash
# App status
databricks apps get battlecards-review-pg --profile fe-vm-pmt --output json | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps({k:d.get(k) for k in ['app_status','compute_status','url']}, indent=2))"

# Logs
databricks apps logs battlecards-review-pg --profile fe-vm-pmt

# Follow logs
databricks apps logs battlecards-review-pg --profile fe-vm-pmt --follow
```

### App URL

https://battlecards-review-pg-450770322676592.aws.databricksapps.com

## Database

- **Instance**: `battlestation` (UID: `98170129-d87f-4357-b0ce-8991c128dea8`)
- **Host**: `instance-98170129-d87f-4357-b0ce-8991c128dea8.database.cloud.databricks.com`
- **Database**: `databricks_postgres` (dev)
- **Port**: 5432

The app auto-creates tables on startup via `init_db()`. Schema init runs each statement individually to handle ownership mismatches between local user and app service principal.

### Service Principal

- **Client ID**: `09f84e42-360b-47d4-b06e-47f6b2dc22a1`
- **Name**: `app-8p4191 battlecards-review-pg`
- Has `CAN_CONNECT_AND_CREATE` on `databricks_postgres` via `databricks.yml`
- Has `ALL` on `public` schema (granted manually)

### Granting SP Permissions (if needed)

If the app crashes with `permission denied for schema public`, grant permissions via Python:

```bash
uv run python3 -c "
from dotenv import load_dotenv; load_dotenv()
import os; from databricks.sdk import WorkspaceClient; import psycopg
w = WorkspaceClient(profile=os.getenv('DATABRICKS_PROFILE', 'fe-vm-pmt'))
token = w.config.oauth_token().access_token
conn = psycopg.connect(host='instance-98170129-d87f-4357-b0ce-8991c128dea8.database.cloud.databricks.com',
    port=5432, dbname='databricks_postgres', user=w.current_user.me().user_name, password=token, sslmode='require')
conn.autocommit = True; cur = conn.cursor()
sp = '09f84e42-360b-47d4-b06e-47f6b2dc22a1'
cur.execute(f'GRANT ALL ON SCHEMA public TO \"{sp}\"')
cur.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO \"{sp}\"')
cur.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO \"{sp}\"')
cur.execute(f'GRANT ALL ON ALL TABLES IN SCHEMA public TO \"{sp}\"')
cur.execute(f'GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO \"{sp}\"')
print('Done'); cur.close(); conn.close()
"
```

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main Flask application, routes, schema, and API |
| `workflow_runner.py` | Background workflow execution (LLM calls, generation) |
| `exa_fact_checker.py` | Exa-powered fact checking for battlecard claims |
| `app.yaml` | Databricks Apps entry point config |
| `databricks.yml` | DAB bundle config (app + database resources) |
| `pyproject.toml` | Python project config (uv) |
| `requirements.txt` | Dependencies for Databricks Apps platform |
| `templates/` | Jinja2 HTML templates |
| `static/` | CSS, JS, and data files |

## Dependencies

Both `pyproject.toml` (for `uv`) and `requirements.txt` (for Databricks Apps) must be kept in sync. When adding a dependency:

```bash
# Add to pyproject.toml and uv.lock
uv add <package>

# Then update requirements.txt to match
```

## Deployment Gotchas

Issues encountered during initial deployment and how they were resolved:

1. **Variable interpolation in workspace host**: `${var.workspace_host}` is not supported for `workspace.host` in `databricks.yml`. Hardcode the URL directly.

2. **Database instance name**: Use the Lakebase instance **name** (`battlestation`), not the DNS prefix (`instance-98170129-...`).

3. **`database_catalogs` resource**: Removed — requires `CREATE CATALOG` on the metastore. Not needed when the database already exists.

4. **`requirements.txt` required**: Databricks Apps platform uses `pip`/`requirements.txt` to install dependencies, not `uv`/`pyproject.toml`. Both files must exist.

5. **Port mismatch**: The platform proxy expects the app on port **8000**. The app reads `APP_PORT` env var (default 8000). For local dev with `databricks apps run-local`, use `--app-port` to match.

6. **Schema ownership**: Tables created by a user are owned by that user. The app SP cannot `ALTER TABLE` or `CREATE INDEX` on tables it doesn't own. Fixed by making `init_db()` run each schema statement individually with try/except, so ownership errors on existing objects are non-fatal.

7. **SP schema permissions**: The `CAN_CONNECT_AND_CREATE` permission in `databricks.yml` grants `CONNECT` and `CREATE` on the database but not on the `public` schema. Manual `GRANT ALL ON SCHEMA public` is needed (see "Granting SP Permissions" section above).
