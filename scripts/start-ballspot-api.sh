#!/usr/bin/env bash
#
# Ballspot challenge API — uvicorn entrypoint for local / PM2.
#
# Prerequisites:
#   - Repo layout: this file lives under <repo>/scripts/; sibling dir is ballspot-challenge-api/
#   - Install: pip install -e ./custom-ballspotting && pip install -e ./ballspot-challenge-api
#   - Checkpoint paths in ballspot-challenge-api/config/app.json
#   - chmod +x scripts/start-ballspot-api.sh (once)
#
# -----------------------------------------------------------------------------
# PM2 — examples (use absolute paths on your machine)
#
# Prefer calling uvicorn from the repo venv so PATH/env matches production:
#
#   pm2 start /absolute/path/to/bas/.venv/bin/uvicorn --name ballspot-api \
#     --cwd /absolute/path/to/bas/ballspot-challenge-api -- \
#     app.main:app --host 0.0.0.0 --port 8000
#
# Alternative: run this script (uvicorn must be on PATH — less reliable for systemd/pm2):
#
#   pm2 start scripts/start-ballspot-api.sh --name ballspot-api --interpreter bash
#
# Logs / lifecycle:
#
#   pm2 logs ballspot-api
#   pm2 restart ballspot-api
#   pm2 stop ballspot-api
#   pm2 delete ballspot-api
#
# Persist process list across reboot:
#
#   pm2 save
#   pm2 startup          # follow the printed one-liner
#
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/ballspot-challenge-api"

exec uvicorn app.main:app --host 0.0.0.0 --port 40069
