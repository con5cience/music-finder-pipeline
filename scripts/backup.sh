#!/usr/bin/env bash
# One-shot factory backup (poe backup). The ledgers and embeddings are the
# unreconstructible state — fetch-cache and audio are not (re-fetchable).
# Dated dumps, keeps the newest 7, prints sizes. Run before risky surgery.
set -euo pipefail

DEST="${PIPELINE_BACKUP_DIR:-$HOME/backups/music-finder-pipeline}"
DSN="${PIPELINE_DATABASE_URL:-postgresql://pipeline:pipeline@localhost:5440/pipeline}"
mkdir -p "$DEST"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$DEST/pipeline-$STAMP.dump"

# No native postgres-client on this box: prefer host pg_dump when present,
# else run it inside the factory-db container (sudo prompt is fine — this is
# a user-run one-shot, never automated).
if command -v pg_dump >/dev/null; then
  pg_dump --format=custom --compress=6 --no-owner --dbname="$DSN" \
    --exclude-table-data=fetch_cache --file="$OUT"
else
  sudo docker compose -f "$(dirname "$0")/../compose.yaml" exec -T factory-db \
    pg_dump --format=custom --compress=6 --no-owner -U pipeline -d pipeline \
    --exclude-table-data=fetch_cache > "$OUT"
fi
ls -t "$DEST"/pipeline-*.dump | tail -n +8 | xargs -r rm --
echo "backup: $OUT ($(du -h "$OUT" | cut -f1)); kept: $(ls "$DEST"/pipeline-*.dump | wc -l)"
echo "restore: pg_restore --clean --if-exists --no-owner --dbname=<dsn> $OUT"
