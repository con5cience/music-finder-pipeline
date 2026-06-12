#!/bin/sh
# factory-doctor: the watchdog that survives every session and reboot.
# Containerized self-healing (compose service): if embeds flatline for two
# consecutive checks while the DB is reachable, force-recreate the workers
# (the wedged-reader class). Logs to stdout (docker logs factory-doctor).
apk add --no-cache postgresql-client > /dev/null 2>&1  # docker:cli is alpine
export COMPOSE_PROJECT_NAME=music-finder-pipeline  # review finding: /workspace
# would otherwise become a PARALLEL project (fresh volumes, port conflicts)
ZEROES=0
sleep 120  # let the fleet boot before judging it
while true; do
  if [ -f /workspace/.maintenance-window ]; then
    echo "$(date -u +%H:%M) doctor: maintenance window open — standing down"
    ZEROES=0
    sleep 1800
    continue
  fi
  RATE=$(psql "$PIPELINE_DATABASE_URL" -tAc \
    "SELECT count(*) FROM artist_embedding WHERE computed_at > now() - interval '30 minutes'" 2>/dev/null)
  if [ -z "$RATE" ]; then
    echo "$(date -u +%H:%M) doctor: DB unreachable — no action"
    ZEROES=0
  elif [ "$RATE" -eq 0 ]; then
    ZEROES=$((ZEROES+1))
    echo "$(date -u +%H:%M) doctor: zero embeds (strike $ZEROES/2)"
    if [ "$ZEROES" -ge 2 ]; then
      echo "$(date -u +%H:%M) doctor: AUTO-REMEDIATION — recreating workers"
      docker compose -f /workspace/compose.yaml up -d --force-recreate worker-gpu worker-io
      ZEROES=0
    fi
  else
    ZEROES=0
    echo "$(date -u +%H:%M) doctor: healthy (~$((RATE*2))/hr)"
  fi
  sleep 1800
done
