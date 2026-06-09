#!/usr/bin/env bash
# One-command bootstrap for the data layer (the uv-scriptable parts).
#
# System prerequisites — uv, Docker, the Temporal CLI, and (on the box) the
# NVIDIA driver — are NOT installed here; see docs/SETUP.md. This brings up the
# uv environment, Postgres, and the schema. On the box, include MuQ:
#   WITH_MUQ=1 ./scripts/bootstrap.sh
set -euo pipefail
cd "$(dirname "$0")/.."

groups=(--group models)
[[ "${WITH_MUQ:-0}" == "1" ]] && groups+=(--group muq)

echo "==> checking prerequisites"
command -v uv     >/dev/null || { echo "  missing: uv (https://astral.sh/uv)"; exit 1; }
command -v docker >/dev/null || { echo "  missing: docker"; exit 1; }
command -v temporal >/dev/null || echo "  note: 'temporal' CLI not found — needed to run the worker (see docs/SETUP.md)"

echo "==> uv sync ${groups[*]}"
uv sync "${groups[@]}"

echo "==> starting Postgres (compose db)"
docker compose up -d db
echo "==> waiting for Postgres"
until docker compose exec -T db pg_isready -U pipeline >/dev/null 2>&1; do sleep 1; done

echo "==> applying migrations"
uv run alembic upgrade head

cat <<'NEXT'

bootstrap complete. next:
  uv run poe temporal   # self-hosted Temporal dev server (separate terminal)
  uv run poe worker     # start the worker
  uv run poe test       # run the test suite
NEXT
