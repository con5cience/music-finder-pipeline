#!/usr/bin/env bash
# One-shot ops cycle (poe ops) — the cadence the user runs on demand instead
# of cron: apply Tier-C decisions, refresh tag calibration, publish to the
# serving DB, rebuild the app's tag vectors. Each step is idempotent.
set -euo pipefail

APP_DSN="${APP_DATABASE_URL:-postgresql://musicfinder:musicfinder@localhost:5433/musicfinder}"
SIBLING="${MUSIC_FINDER_DIR:-$HOME/g/music-finder}"

echo "== review-poll =="
.venv/bin/python -m pipeline.review_poller
echo "== tag-calibrate =="
.venv/bin/python -m pipeline.tag_calibration
echo "== publish =="
APP_DATABASE_URL="$APP_DSN" .venv/bin/python -m pipeline.publish --limit 1000000
echo "== rebuild app tag vectors (auto-reindexes the tag_vector HNSW at the end) =="
# rebuild-vectors REINDEXes the HNSW index itself after a full rebuild — a mass
# tag_vector rewrite scrambles the graph otherwise (2026-06-17 crate incident).
( cd "$SIBLING/server" && DATABASE_URL="$APP_DSN" npm run --silent rebuild-vectors )
echo "== ops cycle complete =="
