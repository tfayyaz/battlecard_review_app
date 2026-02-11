#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $0 --profile <databricks-profile> \
     --source-instance <provisioned-instance-name> \
     --target-endpoint <projects/.../branches/.../endpoints/...> \
     [--source-db databricks_postgres] \
     [--target-db databricks_postgres] \
     [--output-dir /tmp]

Notes:
- Uses pg_dump/pg_restore and migrates all BASE TABLES in schema public.
- Skips Databricks-managed internal objects by doing table-scoped export only.
USAGE
}

PROFILE=""
SOURCE_INSTANCE=""
TARGET_ENDPOINT=""
SOURCE_DB="databricks_postgres"
TARGET_DB="databricks_postgres"
OUTPUT_DIR="/tmp"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --source-instance) SOURCE_INSTANCE="$2"; shift 2 ;;
    --target-endpoint) TARGET_ENDPOINT="$2"; shift 2 ;;
    --source-db) SOURCE_DB="$2"; shift 2 ;;
    --target-db) TARGET_DB="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$PROFILE" || -z "$SOURCE_INSTANCE" || -z "$TARGET_ENDPOINT" ]]; then
  usage
  exit 1
fi

if [[ -x /opt/homebrew/opt/libpq/bin/pg_dump ]]; then
  PG_BIN="/opt/homebrew/opt/libpq/bin"
else
  PG_BIN=""
fi

if [[ -n "$PG_BIN" ]]; then
  PG_DUMP="$PG_BIN/pg_dump"
  PG_RESTORE="$PG_BIN/pg_restore"
  PSQL="$PG_BIN/psql"
else
  PG_DUMP="pg_dump"
  PG_RESTORE="pg_restore"
  PSQL="psql"
fi

for cmd in databricks jq "$PG_DUMP" "$PG_RESTORE" "$PSQL"; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Missing required command: $cmd" >&2; exit 1; }
done

mkdir -p "$OUTPUT_DIR"
DUMP_FILE="$OUTPUT_DIR/${SOURCE_INSTANCE}_to_autoscaling_$(date +%Y%m%d_%H%M%S).dump"

USER_EMAIL=$(databricks current-user me -p "$PROFILE" -o json | jq -r .userName)
SOURCE_HOST=$(databricks database list-database-instances -p "$PROFILE" -o json | jq -r ".[] | select(.name==\"$SOURCE_INSTANCE\") | .read_write_dns")
TARGET_HOST=$(databricks postgres get-endpoint "$TARGET_ENDPOINT" -p "$PROFILE" -o json | jq -r '.status.hosts.host')
SOURCE_TOKEN=$(databricks database generate-database-credential -p "$PROFILE" --json "{\"instance_names\":[\"$SOURCE_INSTANCE\"]}" -o json | jq -r .token)
TARGET_TOKEN=$(databricks postgres generate-database-credential "$TARGET_ENDPOINT" -p "$PROFILE" -o json | jq -r .token)

if [[ -z "$SOURCE_HOST" || "$SOURCE_HOST" == "null" ]]; then
  echo "Could not resolve source instance host for $SOURCE_INSTANCE" >&2
  exit 1
fi
if [[ -z "$TARGET_HOST" || "$TARGET_HOST" == "null" ]]; then
  echo "Could not resolve target endpoint host for $TARGET_ENDPOINT" >&2
  exit 1
fi

TABLES=$(PGPASSWORD="$SOURCE_TOKEN" PGSSLMODE=require "$PSQL" \
  -h "$SOURCE_HOST" -p 5432 -U "$USER_EMAIL" -d "$SOURCE_DB" -Atc \
  "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE' ORDER BY table_name;")

if [[ -z "$TABLES" ]]; then
  echo "No public base tables found in source DB: $SOURCE_DB" >&2
  exit 1
fi

DUMP_ARGS=()
while IFS= read -r t; do
  [[ -n "$t" ]] || continue
  DUMP_ARGS+=("--table=public.$t")
done <<< "$TABLES"

echo "Dumping $(echo "$TABLES" | wc -l | tr -d ' ') tables from $SOURCE_INSTANCE/$SOURCE_DB"
PGPASSWORD="$SOURCE_TOKEN" PGSSLMODE=require "$PG_DUMP" \
  -h "$SOURCE_HOST" -p 5432 -U "$USER_EMAIL" -d "$SOURCE_DB" \
  -F c --no-owner --no-privileges "${DUMP_ARGS[@]}" -f "$DUMP_FILE"

echo "Recreating target DB $TARGET_DB on $TARGET_ENDPOINT"
PGPASSWORD="$TARGET_TOKEN" PGSSLMODE=require "$PSQL" \
  -h "$TARGET_HOST" -p 5432 -U "$USER_EMAIL" -d postgres \
  -v ON_ERROR_STOP=1 -c "DROP DATABASE IF EXISTS \"$TARGET_DB\";"
PGPASSWORD="$TARGET_TOKEN" PGSSLMODE=require "$PSQL" \
  -h "$TARGET_HOST" -p 5432 -U "$USER_EMAIL" -d postgres \
  -v ON_ERROR_STOP=1 -c "CREATE DATABASE \"$TARGET_DB\";"

echo "Restoring to target"
PGPASSWORD="$TARGET_TOKEN" PGSSLMODE=require "$PG_RESTORE" \
  -h "$TARGET_HOST" -p 5432 -U "$USER_EMAIL" -d "$TARGET_DB" \
  --no-owner --no-privileges "$DUMP_FILE"

echo "Comparing table row counts (source vs target)"
COUNT_QUERY="SELECT table_name, (xpath('/row/cnt/text()', query_to_xml(format('SELECT count(*) AS cnt FROM %I.%I', 'public', table_name), true, true, '')))[1]::text::bigint AS cnt FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE' ORDER BY table_name;"

SOURCE_COUNTS=$(mktemp)
TARGET_COUNTS=$(mktemp)
trap 'rm -f "$SOURCE_COUNTS" "$TARGET_COUNTS"' EXIT

PGPASSWORD="$SOURCE_TOKEN" PGSSLMODE=require "$PSQL" -h "$SOURCE_HOST" -p 5432 -U "$USER_EMAIL" -d "$SOURCE_DB" -AtF $'\t' -c "$COUNT_QUERY" > "$SOURCE_COUNTS"
PGPASSWORD="$TARGET_TOKEN" PGSSLMODE=require "$PSQL" -h "$TARGET_HOST" -p 5432 -U "$USER_EMAIL" -d "$TARGET_DB" -AtF $'\t' -c "$COUNT_QUERY" > "$TARGET_COUNTS"

if diff -u "$SOURCE_COUNTS" "$TARGET_COUNTS" >/dev/null; then
  echo "Row counts match for all public tables."
else
  echo "Row count mismatch detected. Diff:"
  diff -u "$SOURCE_COUNTS" "$TARGET_COUNTS" || true
  exit 1
fi

echo "Migration complete. Dump file: $DUMP_FILE"
