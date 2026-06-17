#!/usr/bin/env bash
# poe reindex-vectors — rebuild the serving tag_vector HNSW index.
#
# REQUIRED after any FULL rebuild-vectors: a mass tag_vector rewrite scrambles
# the HNSW graph (incremental index maintenance during a full-column UPDATE
# leaves a globally inconsistent graph that returns near-orthogonal "nearest"
# neighbors — the 2026-06-17 crate-recs incident). rebuild-vectors now reindexes
# automatically at the end of a full run; this is the standalone cure (~seconds)
# for an already-corrupt index without re-running the whole vector rebuild.
set -euo pipefail
APP_DSN="${APP_DATABASE_URL:-postgresql://musicfinder:musicfinder@localhost:5433/musicfinder}"
SIBLING="${MUSIC_FINDER_DIR:-$HOME/g/music-finder}"
( cd "$SIBLING/server" && DATABASE_URL="$APP_DSN" npm run --silent reindex-vectors )
