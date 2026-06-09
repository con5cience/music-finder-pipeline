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

# Fresh Ubuntu installs need sudo for docker until you log out/in to pick up the
# 'docker' group. Detect daemon access and fall back to sudo so bootstrap works
# either way.
DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if sudo docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
    echo "  note: using sudo for docker (add \$USER to the 'docker' group + re-login to drop it)"
  else
    echo "  error: cannot reach the Docker daemon (is it installed and running?)"; exit 1
  fi
fi

echo "==> uv sync ${groups[*]}"
uv sync "${groups[@]}"

echo "==> starting Postgres (compose db)"
"${DOCKER[@]}" compose up -d db
echo "==> waiting for Postgres"
until "${DOCKER[@]}" compose exec -T db pg_isready -U pipeline >/dev/null 2>&1; do sleep 1; done

echo "==> applying migrations"
uv run alembic upgrade head

cat <<'NEXT'

bootstrap complete. next:
  uv run poe temporal   # self-hosted Temporal dev server (separate terminal)
  uv run poe worker     # start the worker
  uv run poe test       # run the test suite
NEXT
