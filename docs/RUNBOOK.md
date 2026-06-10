# Runbook — music-finder-pipeline factory

Operational truth for the corpus factory. Every playbook here was earned by
a real incident; dates reference the session logs.

## Topology

| Service (compose) | What | Port (loopback only) |
|---|---|---|
| factory-db | Postgres 16 + pgvector (corpus, ledgers, Temporal's DBs) | 5440 |
| temporal / temporal-ui | Workflow engine (persisted in factory-db) / UI | 7233 / 8233 |
| worker-io | Workflows + DB activities + platform crawl queues (CPU) | — |
| worker-gpu | Embed queue only (`--role gpu`, nvidia reservation) | — |

The serving app (sibling repo) joins the `music-factory` network and reads
the factory DB as `pipeline-db:5432`. Bring the factory up before the app.

## Start / stop / deploy

```bash
sudo docker compose up -d                  # whole factory
sudo docker compose up -d --build worker-io worker-gpu   # deploy code change
sudo docker compose logs -f worker-io      # fleet logs
```

- **Workflow-code law**: `workflows.py` changes deploy ONLY via image rebuild
  + deliberate restart. The Temporal sandbox re-imports workflow files from
  disk under a live worker while passthrough modules stay in-memory — a
  mid-flight mismatch broke 72 workflows once. Never hot-edit.
- **`systemctl restart docker` bounces EVERY container** including the DB
  under a live build (incident 2026-06-10). Pause the wave seeder first.
- Liveness: `worker_heartbeat` table (30s beat; >90s stale) — surfaced on
  the admin Pipeline Fleet card and `uv run poe factory-status -- --watch`.

## Reboot

Everything self-starts: docker is boot-enabled, every service (both
composes) carries a restart policy, volumes/networks persist, and the
SEEDER is a compose service that resumes the mass build from the ledgers.
After any reboot, verify with one glance at the admin Pipeline Fleet card
(two live workers) or `uv run poe factory-status`. NOT auto-resumed (host
one-shots by design): search-bind waves, backfill sweeps — rerun via poe if
they were mid-flight. Config-change caveat: `docker compose up -d` RECREATES
services whose definition changed — that bounces factory-db too; pause
nothing, everything reconnects, but expect a ~60s blip.

## One-shot ops (no cron by design)

| Command | What |
|---|---|
| `uv run poe ops` | review-poll → tag-calibrate → publish → rebuild app tag vectors |
| `uv run poe backup` | dated pg_dump (excludes fetch_cache data), keep-7 |
| `uv run poe wave-seed -- --total N` | bounded-window mass seeding (resumable, ledger-driven) |
| `uv run poe search-bind -- --limit N` | Tier-B binding waves over unbound artists |
| `uv run poe factory-status -- --watch` | live terminal dashboard |

## Monthly MB sync (ADR-018)

```bash
uv run poe mb-sync                  # dry-run: download latest, gates, diff report
uv run poe mb-sync -- --apply       # after THREE consecutive clean dry-runs
```
- Run in a network-quiet window (~10GB download; direct, not proxied).
- Skips automatically when the latest serial is already applied.
- Both-embedded merge conflicts land in the admin Tier-C queue.
- Every run is ledgered in mb_refresh_run (serial, gates, diff, applied_at).

## Incident playbooks

### Platform 429s (rate limiting)
Bandcamp tripped at 5/s sustained (held in bursts). The system fails SAFE:
429s are never cached, identities stay `pending`, cascades fall through.
1. Check the proxy is active: `PIPELINE_PROXY_URL` in `.env` (ADR-017 proxy
   law — its absence caused the 2026-06-10 incident; direct home-IP traffic
   burns the IP and persists through rate drops).
2. If proxied and still 429ing: lower the platform's `io_rate` in
   `queues.py` PLATFORMS, rebuild workers. Identities re-scan naturally.

### Factory DB down / bounced
Symptoms: connection-refused storms in worker logs, fleet card stale.
1. `sudo docker compose up -d factory-db`; workers self-heal (restart policy
   + idle-client error handling).
2. Pause the wave seeder during planned bounces (`pkill -INT -f wave_seeder`
   — resumes losslessly; ledgers are the checkpoint).

### Postgres image upgrades: collation mismatch
`template database "template1" has a collation version mismatch` after an
image pull moved glibc (blocks CREATE DATABASE — broke Temporal's first
boot). Fix as superuser, then reindex:
```sql
ALTER DATABASE template1 REFRESH COLLATION VERSION;  -- + postgres, pipeline
REINDEX DATABASE CONCURRENTLY pipeline;
```

### Signed audio URL rot (expected, self-healing)
Deezer previews ~1-2h, Bandcamp ~24h, SoundCloud CloudFront-signed. The
fetch path refreshes-on-403 per platform; persistent failures skip the
track (stays pending). No action unless skip rates spike on the fleet card.

### Stale-image deploys (the silent no-op build)
`docker compose build <svc>` NO-OPS silently for services without a build:
key — a fix can ship to the image while the container runs old code
("Started" proves nothing). All image-sharing services now carry build:,
and the standard is: VERIFY THE CHANGE IN-CONTAINER after deploy
(`docker exec <c> .venv/bin/python -c 'import …; print(…)'`).

### Unregistered-activity hang
Workflows park silently at a step whose activity no worker registers (three
occurrences). Coherence tests now assert every workflow-dispatched activity
is registered — if you see eternal parking anyway, check task-queue
pollers in the Temporal UI (:8233) first.

### GPU OOM / embed stalls
`max_concurrent_activities=2` on the gpu worker is the VRAM cap (16GB card;
MuQ+MuLan+activations). Raise only after watching `nvidia-smi` headroom at
sustained load. Container restart policy covers crashes; embeds retry.

## New-box checklist

1. `.env`: `PIPELINE_PROXY_URL` (crawl traffic MUST be proxied),
   `SOUNDCLOUD_CLIENT_ID/SECRET`. Without the proxy var the box's IP burns
   within hours of mass scanning.
2. nvidia-container-toolkit + `nvidia-ctk runtime configure` before
   `worker-gpu` (pause seeder: the docker restart bounces everything).
3. `uv run alembic upgrade head` against factory-db.
4. MB bootstrap: `uv run poe mb-bootstrap --dir <mbdump>` (10 tables incl.
   genre/genre_alias — tags are dead without the vocabulary).
5. First `temporal` boot creates its databases — needs the collation fix if
   the volume predates the image's glibc.

## Secrets

- Pipeline `.env` (gitignored): proxy URL, SoundCloud app creds.
- App `.env` (gitignored): the same SC creds (embed tokens), DB DSNs,
  `PIPELINE_DATABASE_URL_DOCKER` convention — see the app compose comments.
- Nothing secret lives in compose files or code; creds enter via env_file.
- Rotation: SoundCloud secret has transited chat logs — rotate it at
  soundcloud.com/you/apps when convenient.
