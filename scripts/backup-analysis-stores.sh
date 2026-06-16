#!/usr/bin/env bash
# ADR-021: back up the irreplaceable re-analysis stores so a future corpus
# re-analysis never needs re-acquisition —
#   1) MuLan analysis vectors (artist_analysis_vector) + the clip ledger
#      (audio_clip_archive)  -> pg_dump custom format
#   2) the compressed clip files themselves (the /archive volume) -> tar.gz
#
# LOCAL backup (writes under $BACKUP_DIR on the same host/disk): protects
# against LOGICAL loss (an accidental DROP / bad cleanup), NOT disk failure.
# For disk-failure protection, copy the output off-host (rsync/S3) — see the
# OFFHOST hint printed at the end.
set -euo pipefail

DEST="${BACKUP_DIR:-$HOME/backups/music-finder}"
STAMP="$(date +%Y%m%d-%H%M%S)"
FDB=music-finder-pipeline-factory-db-1
WORKER=music-finder-pipeline-worker-io-1
mkdir -p "$DEST"

dump="$DEST/analysis_stores_$STAMP.dump"
tarball="$DEST/audio_archive_$STAMP.tar.gz"

echo "[1/2] pg_dump artist_analysis_vector + audio_clip_archive -> $dump"
docker exec -i "$FDB" pg_dump -U pipeline -d pipeline \
  -t artist_analysis_vector -t audio_clip_archive -Fc > "$dump"

echo "[2/2] tar the /archive compressed clips -> $tarball"
# tar exits 1 if a clip is written by prep mid-read (benign, expected on a live
# archive); only exit >=2 is a real failure.
docker exec -i "$WORKER" tar czf - -C /archive . > "$tarball" || [ "$?" -le 1 ]

echo "--- verify ---"
echo -n "dump tables: "; docker exec -i "$FDB" pg_restore -l < "$dump" 2>/dev/null | grep -c "TABLE DATA" || true
echo -n "archive files in tarball: "; tar tzf "$tarball" | grep -c '\.ogg$' || true
ls -lh "$dump" "$tarball"
echo "OFFHOST: same-disk only — replicate off-host, e.g.:"
echo "  rsync -a $DEST/ <user>@<host>:/backups/music-finder/"
