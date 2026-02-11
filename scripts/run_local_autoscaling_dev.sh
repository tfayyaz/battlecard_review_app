#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $0 --profile <databricks-profile> \
     --endpoint <projects/.../branches/.../endpoints/...> \
     [--entry-point app.local.yaml] [--app-port 5060] [--port 5061]

Example:
  $0 --profile fe-vm-pmt \
     --endpoint projects/battlecards-review-dev/branches/dev/endpoints/rw
USAGE
}

PROFILE=""
ENDPOINT=""
ENTRY_POINT="app.local.yaml"
APP_PORT="5060"
PROXY_PORT="5061"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --entry-point) ENTRY_POINT="$2"; shift 2 ;;
    --app-port) APP_PORT="$2"; shift 2 ;;
    --port) PROXY_PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$PROFILE" || -z "$ENDPOINT" ]]; then
  usage
  exit 1
fi

command -v databricks >/dev/null 2>&1 || { echo "Missing databricks CLI" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "Missing jq" >&2; exit 1; }

if [[ ! -f .env ]]; then
  echo ".env not found in repo root (needed for EXA_API_KEY)." >&2
  exit 1
fi

if [[ ! -f "$ENTRY_POINT" ]]; then
  echo "Entry point file not found: $ENTRY_POINT" >&2
  exit 1
fi

HOST=$(databricks postgres get-endpoint "$ENDPOINT" -p "$PROFILE" -o json | jq -r '.status.hosts.host')
if [[ -z "$HOST" || "$HOST" == "null" ]]; then
  echo "Could not resolve endpoint host for $ENDPOINT" >&2
  exit 1
fi

echo "Using PGHOST=$HOST"
set -a
source .env
set +a

databricks apps run-local \
  --entry-point "$ENTRY_POINT" \
  --profile "$PROFILE" \
  --app-port "$APP_PORT" \
  --port "$PROXY_PORT" \
  --env "PGHOST=$HOST" \
  --env "PGDATABASE=databricks_postgres" \
  --env "DATABRICKS_PROFILE=$PROFILE" \
  --env "EXA_API_KEY=$EXA_API_KEY"
