#!/usr/bin/env bash
# Overnight maintenance window (user-authorized, 2026-06-11):
#   1. stop worker-gpu (prep keeps staging; stage bounded by the workflow window)
#   2. legacy v2-tag re-sweep (host GPU, fetch-cache makes most of it free)
#   3. true-fp16 parity experiment (INFORMATIONAL: report only, no fleet change)
#   4. wave-3 heavy-head demo slices (Demucs stems + ASR) on the free GPU
#   5. conc-3 retest, 20 min, AUTO-REVERT to conc 2 in all paths (trap)
#   6. restart fleet exactly as found; write /tmp/maintenance-report.txt
set -u
cd /home/will/g/music-finder-pipeline
PY=.venv/bin/python
LOG=/tmp/maintenance-window.log
REPORT=/tmp/maintenance-report.txt
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

restore_fleet() {
  log "RESTORE: conc back to 2 + worker-gpu up"
  sed -i 's/PIPELINE_GPU_CONCURRENCY: "3"/PIPELINE_GPU_CONCURRENCY: "2"/' compose.yaml || true
  sg docker -c "docker compose up -d worker-gpu" >> "$LOG" 2>&1
}
trap restore_fleet EXIT

log "=== MAINTENANCE WINDOW OPEN ==="
sg docker -c "docker compose stop worker-gpu" >> "$LOG" 2>&1
log "worker-gpu stopped; GPU is ours"

log "--- stage 1: legacy v2-tag re-sweep ---"
timeout 14400 $PY -m pipeline.backfill_analysis --limit 100000 --batch 50 >> "$LOG" 2>&1
SWEEP_RC=$?
log "sweep done rc=$SWEEP_RC"
$PY -c "
import psycopg
c = psycopg.connect('postgresql://pipeline:pipeline@localhost:5440/pipeline')
n = c.execute(\"SELECT count(DISTINCT artist_id) FROM artist_tag_scores\").fetchone()[0]
print(f'artists with v2 artist-mean tags now: {n}')" >> "$LOG" 2>&1

log "--- stage 2: fp16 parity experiment (report-only) ---"
timeout 1800 $PY - >> "$LOG" 2>&1 <<'PYEOF'
import numpy as np, torch
from pipeline.embedders.registry import get_embedder
from pipeline.bench.types import Clip
import soundfile as sf, tempfile, os
sr = 24000
rng = np.random.default_rng(5)
paths = []
tmp = tempfile.mkdtemp(prefix="fp16-")
for i in range(3):
    p = os.path.join(tmp, f"c{i}.wav")
    sf.write(p, (rng.standard_normal(sr * 10) * 0.1).astype(np.float32), sr)
    paths.append(p)
clips = [Clip(id=str(i), artist_id="x", path=p) for i, p in enumerate(paths)]
emb = get_embedder()
v32 = np.asarray(emb.embed(clips))
torch.cuda.reset_peak_memory_stats()
print("fp32 peak MB:", torch.cuda.max_memory_allocated() // 2**20)
try:
    emb.model = emb.model.half()
    torch.cuda.reset_peak_memory_stats()
    v16 = np.asarray(emb.embed(clips))
    cos = (v32 * v16).sum(1) / (np.linalg.norm(v32, axis=1) * np.linalg.norm(v16, axis=1) + 1e-9)
    print("FP16 PARITY cosines:", [round(float(c), 5) for c in cos])
    print("fp16 peak MB:", torch.cuda.max_memory_allocated() // 2**20)
    print("VERDICT:", "PARITY OK (>=0.999)" if min(cos) >= 0.999 else "PARITY FAILED — do not deploy half()")
except Exception as e:
    print("fp16 experiment failed cleanly:", type(e).__name__, str(e)[:200])
PYEOF

log "--- stage 3: wave-3 heavy demos (GPU free) ---"
timeout 3600 $PY -m pipeline.wave3 --head stems --limit 12 >> "$LOG" 2>&1
timeout 1800 $PY -m pipeline.wave3 --head asr --limit 25 >> "$LOG" 2>&1
log "wave-3 demos done"

if [ "${MAINT_SKIP_CONC:-0}" = "1" ]; then
  log "stage 4 skipped (data already captured: ~1746/hr, 4 OOMs)"
  RATE=skipped; OOMS=skipped
else
log "--- stage 4: conc-3 retest (20 min, auto-revert) ---"
sed -i 's/PIPELINE_GPU_CONCURRENCY: "2"/PIPELINE_GPU_CONCURRENCY: "3"/' compose.yaml
sg docker -c "docker compose up -d worker-gpu" >> "$LOG" 2>&1
sleep 1200
OOMS=$(sg docker -c "docker logs music-finder-pipeline-worker-gpu-1 --since 20m 2>&1" | grep -cE "OutOfMemory" || true)
RATE=$($PY -c "
import psycopg
c = psycopg.connect('postgresql://pipeline:pipeline@localhost:5440/pipeline')
print(c.execute(\"SELECT count(*) FROM artist_embedding WHERE computed_at > now() - interval '20 minutes'\").fetchone()[0] * 3)")
log "conc-3 retest: ~$RATE/hr, OOMs=$OOMS"
fi
# trap restores conc 2 + restarts

{
  echo "MAINTENANCE REPORT $(date)"
  echo "sweep rc=$SWEEP_RC (0=done, 124=hit 4h cap — rerun to continue, it resumes)"
  grep -E "artists with v2|PARITY|peak MB|VERDICT" "$LOG"
  grep -E '"head"|"done"|"skipped"' "$LOG" | tail -8
  echo "conc-3 retest: ~$RATE/hr with $OOMS OOMs (fleet restored to conc 2 regardless)"
  echo "Full log: $LOG"
} > "$REPORT"
log "=== WINDOW CLOSED — report at $REPORT ==="
