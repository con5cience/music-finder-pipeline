#!/bin/sh
# factory-doctor: the watchdog that survives every session and reboot.
# Containerized self-healing (compose service): if embeds flatline for two
# consecutive checks while the DB is reachable, force-recreate the workers
# (the wedged-reader class). Logs to stdout (docker logs factory-doctor).
apk add --no-cache postgresql-client > /dev/null 2>&1  # docker:cli is alpine
export COMPOSE_PROJECT_NAME=music-finder-pipeline  # review finding: /workspace
# would otherwise become a PARALLEL project (fresh volumes, port conflicts)
ZEROES=0
# Running-ingest-workflow count above which a zero-embed stall is a Temporal
# FLOOD (recreating workers can't fix it — the flood lives in Temporal), not a
# wedged reader. Comfortably above the seeder's clamped legit max (~1000:
# low-water 500 + batch 500 overshoot). See 2026-06-15.
FLOOD=1500
# After a force-recreate (or cold boot) embeds can take ~30+ min to resume. This
# is NOT model reload — that's ~30s (measured 2026-06-18: worker idle ~33min,
# THEN a ~27s embedder + 3 XLM-Roberta load). The gap is the embed queue
# refilling / work flowing back to the fresh worker. The doctor must not count it
# as zero-embed strikes, or it recreates a still-warming worker and resets the
# wait — a restart loop (near-miss 2026-06-18). WARMUP must cover the whole gap.
WARMUP=2700
# Liveness probe DURING warm-up. The grace legitimately covers the embed queue
# refilling, but a blind sleep also masks a DOA worker whose MuQ model never
# loads (incident 2026-06-19: a recreate brought up a worker at 784MB resident —
# no model — and the 2700s sleep hid it ~75min, until two zero-embed strikes
# finally recreated it). warm-up still waits the full grace for the queue, but
# aborts early to a recreate if the model is absent from VRAM past the load
# deadline. Only a VALID sub-threshold reading is DOA; an empty reading (worker
# not yet exec-able) is inconclusive and waits on. Capped so a host GPU/driver
# fault can't flap recreates forever.
MODEL_MIN_MB=3000        # live floor ~5221MB observed, DOA ~784MB — wide margin
MODEL_LOAD_DEADLINE=180  # model load is ~30s; absent by 3min => DOA, not slow
                         # (was 600; tightened 2026-06-21 — 10min DOA recovery was
                         # lost time; 180s is still 6x the ~30s load, near-zero FP)
DOA_RECREATES_MAX=2
# GPU util + VRAM, sampled THROUGH the worker-gpu container (the doctor's own
# alpine image has no nvidia-smi). Every incident (2026-06-14/15/18) was hard to
# triage for lack of GPU state at the decision: high util + zero embeds = stage
# downstream of the GPU wedged or a flood; low util + zero embeds = a wedged or
# still-cold-starting worker. Logging it next to every verdict makes the next
# stall a one-look call. "n/a" if the worker is down/unreachable.
gpu_stat() {
  docker compose -f /workspace/compose.yaml exec -T worker-gpu \
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | awk -F',' 'NF{gsub(/ /,"");print $1"% "$2"/"$3"MB"}' || true
}
# Just memory.used (MB) for the warm-up liveness probe; empty if the worker is
# down / not yet exec-able (treated as inconclusive by warmup, never as DOA).
gpu_mem_mb() {
  docker compose -f /workspace/compose.yaml exec -T worker-gpu \
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | tr -dc '0-9'
}
# Wait out the post-recreate / cold-start grace (queue refill), but recreate
# early if the worker is dead-on-arrival — the MuQ model never became
# GPU-resident. Replaces the old blind WARMUP-second sleep.
warmup() {
  _doa=0; _elapsed=0
  while [ "$_elapsed" -lt "$WARMUP" ]; do
    sleep 60; _elapsed=$((_elapsed+60))
    [ "$_elapsed" -lt "$MODEL_LOAD_DEADLINE" ] && continue
    [ "$_doa" -ge "$DOA_RECREATES_MAX" ] && continue
    _mem=$(gpu_mem_mb)
    if [ -n "$_mem" ] && [ "$_mem" -lt "$MODEL_MIN_MB" ]; then
      _doa=$((_doa+1))
      echo "$(date -u +%H:%M) doctor: DOA worker — MuQ model not resident (${_mem}MB < ${MODEL_MIN_MB}MB) after ${_elapsed}s, recreating ($_doa/$DOA_RECREATES_MAX)"
      docker compose -f /workspace/compose.yaml up -d --force-recreate worker-gpu worker-io
      _elapsed=0
    fi
  done
}
warmup  # bounded cold-start grace + DOA liveness probe (was a blind WARMUP sleep)
while true; do
  if [ -f /workspace/.maintenance-window ]; then
    echo "$(date -u +%H:%M) doctor: maintenance window open — standing down"
    ZEROES=0
    sleep 1800
    continue
  fi
  RATE=$(psql "$PIPELINE_DATABASE_URL" -tAc \
    "SELECT count(*) FROM artist_embedding WHERE computed_at > now() - interval '30 minutes'" 2>/dev/null)
  GPU=$(gpu_stat); GPU=${GPU:-n/a}
  # Embed work-availability: total backlog (never-embedded artists) + embed-READY
  # depth (null-embedding artists that already have discovered audio_track rows).
  # Pairs with GPU-util to triage a warm-up gap: idle GPU + ready>0 = the GPU is
  # being starved despite work existing (dispatch/feed stall); idle GPU + ready~0
  # = upstream discovery/prep is the bottleneck. ~250ms, indexed. n/a if DB down.
  QUEUE=$(psql "$PIPELINE_DATABASE_URL" -tAc \
    "SELECT 'backlog='||(SELECT count(*) FROM artist WHERE embedding_source IS NULL)||' ready='||(SELECT count(*) FROM artist a WHERE a.embedding_source IS NULL AND EXISTS (SELECT 1 FROM audio_track t WHERE t.artist_id=a.id))" 2>/dev/null)
  QUEUE=${QUEUE:-n/a}
  if [ -z "$RATE" ]; then
    echo "$(date -u +%H:%M) doctor: DB unreachable — no action [gpu $GPU | $QUEUE]"
    ZEROES=0
  elif [ "$RATE" -eq 0 ]; then
    # Fast DOA path (2026-06-21): a zero-embed worker whose MuQ model is absent
    # from VRAM is unambiguously dead — recreate NOW instead of waiting out the
    # 2-strike path (~60min lost). Only a VALID sub-threshold reading is DOA; an
    # empty reading (worker down/not exec-able) is inconclusive and falls through
    # to strikes. A flood has the model resident (mem >= MIN), so it never trips
    # this — it still reaches the flood branch below.
    _mem=$(gpu_mem_mb)
    if [ -n "$_mem" ] && [ "$_mem" -lt "$MODEL_MIN_MB" ]; then
      echo "$(date -u +%H:%M) doctor: DOA worker mid-run — MuQ not resident (${_mem}MB < ${MODEL_MIN_MB}MB) — recreating now [gpu $GPU | $QUEUE]"
      docker compose -f /workspace/compose.yaml up -d --force-recreate worker-gpu worker-io
      ZEROES=0
      warmup
      continue
    fi
    ZEROES=$((ZEROES+1))
    echo "$(date -u +%H:%M) doctor: zero embeds (strike $ZEROES/2) [gpu $GPU | $QUEUE]"
    if [ "$ZEROES" -ge 2 ]; then
      # Branch the remedy on WHY embeds stalled. A FLOOD (too many workflows in
      # flight) overwhelms Temporal; recreating workers just re-polls the flood
      # (it flapped 5+ times to no effect on 2026-06-15). A WEDGED reader (few
      # workflows, stuck worker) is the class the recreate actually fixes.
      RUNNING=$(docker compose -f /workspace/compose.yaml exec -T temporal \
        temporal workflow count --address temporal:7233 \
        -q "ExecutionStatus='Running' AND WorkflowType='IngestArtistWorkflow'" 2>/dev/null \
        | grep -oE '[0-9]+' | head -1)
      if [ -n "$RUNNING" ] && [ "$RUNNING" -ge "$FLOOD" ]; then
        echo "$(date -u +%H:%M) doctor: FLOOD ($RUNNING running >= $FLOOD, gpu $GPU, $QUEUE) — stopping seeder (recreating workers won't help; the flood lives in Temporal)"
        docker compose -f /workspace/compose.yaml stop seeder
      else
        echo "$(date -u +%H:%M) doctor: wedged-reader (running=${RUNNING:-unknown}, gpu $GPU, $QUEUE) — recreating workers"
        docker compose -f /workspace/compose.yaml up -d --force-recreate worker-gpu worker-io
        ZEROES=0
        echo "$(date -u +%H:%M) doctor: recreated — warming up (queue-refill grace + DOA probe) before resuming checks"
        warmup
        continue
      fi
      ZEROES=0
    fi
  else
    ZEROES=0
    echo "$(date -u +%H:%M) doctor: healthy (~$((RATE*2))/hr) [gpu $GPU | $QUEUE]"
  fi
  sleep 1800
done
