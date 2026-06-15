# HANDOFF — the state of the world

Written 2026-06-11 near the end of intensive AI-assisted build sessions.
The RUNBOOK covers incidents and the operator's week; this covers WHAT
EXISTS, WHAT'S RUNNING, and WHAT'S WAITING, so any future operator or
assistant can orient in five minutes.

## The system in one paragraph

A containerized corpus factory (this repo) discovers, analyzes, and
publishes artists to the serving app (`../music-finder`, crates.ltd). MB
dump → identities → per-platform scans (deezer/bandcamp/soundcloud;
youtube discovery-only) → staged embed pipeline (CPU prep queue feeds a
pure-inference GPU lane: MuQ centroids, MuLan tags + mood axes, CPU
analysis, fingerprints) → hourly incremental publish into the app. The
ADR-019 discovery organ crawls Bandcamp's tag tree for artists nobody
indexes, admits them as provisional (mbid NULL, ✦ badge), and the MB
contribution lane submits them back to MusicBrainz. Mission: "discovery
infrastructure for the underground."

## What runs unattended (compose services, all restart-on-boot)

| Service | Job |
|---|---|
| factory-db | Corpus state (artists, embeddings) |
| temporal-db / temporal / temporal-ui | Workflow engine on its OWN Postgres, isolated from the corpus DB (2026-06-15); 512 history shards, 24h retention |
| worker-io | Scans, prep (staging), pipeline activities, hourly stage GC |
| worker-gpu | Pure inference (conc 2 — the proven ceiling; do not raise casually) |
| seeder | Refills the workflow window from the ledgers (window clamped ≤500 in code); exits 0 when corpus done |
| publish-sync | HOURLY incremental publish (watermark; app-then-factory commit law) |
| factory-doctor | DEPLOYED stall watchdog. On a 2-strike zero-embed stall it branches: FLOOD (≥1500 running ingest workflows) → `stop seeder` (the flood lives in Temporal, not the workers); else wedged-reader → `up -d --force-recreate worker-gpu worker-io`. A flood-stopped seeder needs a manual restart |

## The queues and gates as of handoff

- Corpus: ~22k of 451k embedded; ETA fluctuates 10–12 days with platform
  mix. Watch the admin Factory card.
- Tag re-sweep: 3,151 artists with reset ledgers awaiting one maintenance
  window (`MAINT_SKIP_CONC=1 MAINT_ONLY_SWEEP=1 bash scripts/maintenance-window.sh`).
- Fingerprint calibration: auto-relevant at ~25k embedded — produce a
  real-dup report, then decide automation (flag-only until then).
- bc_candidate ledger: discovery valve is MANUAL (`poe discover -- --admit N`).
- mb_submission: 25 artists staged at spot_check. Lane armed as crates_bot.
  WAITING on MetaBrainz community feedback to the forum post; then
  `--submit-tags` may run, and phase-1b (artist-creation edit driver,
  rehearsed on test.musicbrainz.org) becomes buildable.
- Match Review: grows with binding waves; no deadline semantics.

## Decisions that are LAWS (do not relitigate without the operator)

See memory/acquisition-design-decisions.md (the canonical list). The ones
that bite hardest: proxy law (all crawl traffic), fetch-cache 2xx/404-only,
full-analysis-is-the-admission-bar, quality-gates-exclude/relatedness-ranks,
no-cron except the publish-sync exception, commit-order law (app before
watermark), GC age-is-not-orphanhood, NaN ≠ detectable via x!=x in pg.

## Where things live

- ADRs: ../music-finder/docs/adr/ (015 heads, 017 acquisition, 018 MB
  refresh, 019 discovery).
- Incident playbooks + operator's week: docs/RUNBOOK.md.
- MB forum draft: ../music-finder/docs/mb-forum-announcement-draft.md.
- The memory file (for AI assistants): laws, state, gotchas — keep it
  current; it is the institutional memory.

## Known debts, smallest first

- trust-proxy test needs generous timeouts under full-suite load (env
  interplay documented in the test).
- Demucs/ASR at scale: gated on demonstrated value; run slices via
  `poe wave3 -- --published --head stems|asr`.
- MuQ is CC-BY-NC: fine for the no-ads/no-subs/donations posture chosen;
  MusicFM (MIT) re-embed (~12 GPU-days) is the documented escape hatch.
- Offsite backups: deliberate post-corpus plan (pg_dump → snapshotting
  cloud provider). Until then, backups are LOCAL ONLY.
