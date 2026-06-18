#!/usr/bin/env bash
# One-shot backup of BOTH stores (poe backup). Run before risky surgery.
#
# The ledgers + embeddings (factory) and the crates + accounts (serving) are the
# unreconstructible state; fetch-cache and audio are not (re-fetchable), so the
# factory dump excludes their data. Custom format (pg_restore-able, compressed,
# parallel/selective restore). Dated dumps, newest 7 kept per store, each dump
# verified by reading its archive TOC back. Prints sizes + a restore hint.
#
# No native postgres-client on this box, so dumps run inside the DB containers
# (the user runs this in their own shell and is in the docker group — no sudo).
set -euo pipefail

DEST="${PIPELINE_BACKUP_DIR:-$HOME/backups/music-finder-pipeline}"
FACTORY_CONTAINER="${FACTORY_DB_CONTAINER:-music-finder-pipeline-factory-db-1}"
SERVING_CONTAINER="${SERVING_DB_CONTAINER:-music-finder-db-1}"
mkdir -p "$DEST"
STAMP=$(date +%Y%m%d-%H%M%S)

# dump_db <label> <container> <user> <db> [extra pg_dump args...]
# Streams a compressed custom-format dump out of the container to the host, then
# verifies it by listing the archive TOC back through the container's pg_restore
# (a truncated/garbage dump lists zero entries and fails the run). Keeps newest 7.
dump_db() {
  label="$1"; container="$2"; user="$3"; db="$4"; shift 4
  out="$DEST/${label}-$STAMP.dump"
  echo "backup: dumping $label ($db via $container) -> $out"
  # On any failure, remove the partial/empty file so it can't masquerade as a
  # real backup or be retained by the keep-7 prune (a failed run once left a
  # 0-byte .dump, 2026-06-18).
  if ! docker exec -i "$container" pg_dump --format=custom --compress=6 --no-owner \
       -U "$user" -d "$db" "$@" > "$out"; then
    echo "backup: DUMP FAILED for $label — removing partial $out" >&2
    rm -f "$out"
    return 1
  fi
  entries=$(docker exec -i "$container" pg_restore --list < "$out" | grep -cE '^[0-9]' || true)
  if [ "${entries:-0}" -lt 1 ]; then
    echo "backup: VERIFY FAILED — $out has no readable TOC entries; removing it" >&2
    rm -f "$out"
    return 1
  fi
  echo "backup: ok $label — $(du -h "$out" | cut -f1), $entries archive entries"
  ls -t "$DEST/${label}-"*.dump | tail -n +8 | xargs -r rm --
}

# Factory store: exclude the re-fetchable cache + the oauth secrets table.
dump_db pipeline "$FACTORY_CONTAINER" pipeline pipeline \
  --exclude-table-data=fetch_cache --exclude-table-data=mb_oauth

# Serving store (crates, accounts, vectors): small + irreplaceable, dump it all.
# Skip gracefully if the serving stack isn't running on this box.
if docker ps --format '{{.Names}}' | grep -q "^${SERVING_CONTAINER}$"; then
  dump_db serving "$SERVING_CONTAINER" musicfinder musicfinder
else
  echo "backup: serving container $SERVING_CONTAINER not running — skipped" >&2
fi

echo "backup: kept pipeline=$(ls "$DEST"/pipeline-*.dump 2>/dev/null | wc -l) serving=$(ls "$DEST"/serving-*.dump 2>/dev/null | wc -l) in $DEST"
echo "restore: docker exec -i <db-container> pg_restore --clean --if-exists --no-owner -U <user> -d <db> < <dump>"
