"""Seed IngestArtistWorkflow runs from bootstrapped Tier-A identities.

Deterministic workflow ids ("ingest-artist-{artist_id}") make seeding
idempotent: re-running skips identities whose workflow already exists (Temporal
rejects duplicate ids). Use --limit for calibration-scale runs (ADR-017 gates
mass ingest behind the 1k calibration).

Run:  uv run python -m pipeline.seed_ingest --platform deezer --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta

import psycopg
from temporalio.client import Client
from temporalio.exceptions import TemporalError

from pipeline.config import Settings
from pipeline.workflows import IngestArtistInput, IngestArtistWorkflow


def workflow_id(artist_id: str) -> str:
    return f"ingest-artist-{artist_id}"


# Cap a single ingest run's wall-clock so a workflow wedged on a stuck activity
# (or parked indefinitely behind the 0.1/s youtube queue) self-expires into a
# terminal state instead of (a) holding a window slot forever so the seeder's
# low-water gate never drains, and (b) pinning its history past the 24h
# retention. 12h is generous vs the real worst case — even a full youtube-only
# window behind the 0.1/s cap clears in ~3-4h — and well under the 24h retention
# so terminal runs still self-purge. Kills the 4-day zombie runs seen 2026-06-15.
INGEST_EXECUTION_TIMEOUT = timedelta(hours=12)


async def start_ingest_workflow(client, artist_id: str, settings) -> bool:
    """Start IngestArtistWorkflow idempotently. Returns True if newly started,
    False if a run with this (deterministic) id already exists.

    The single place that owns how an ingest run is launched — especially
    execution_timeout — for every producer (mass seeder, calibration seed,
    homonym front-run), so the timeout can't drift across call sites.
    """
    try:
        await client.start_workflow(
            IngestArtistWorkflow.run,
            IngestArtistInput(str(artist_id)),
            id=workflow_id(str(artist_id)),
            task_queue=settings.temporal_task_queue,
            execution_timeout=INGEST_EXECUTION_TIMEOUT,
        )
        return True
    except TemporalError:  # already-started → idempotent skip
        return False


def pending_artists(conn, platform: str | None, limit: int) -> list[str]:
    """Artist ids with at least one audio-role identity (optionally: on one
    platform), for cascade seeding.

    Deliberately unfiltered beyond that: idempotence lives in the deterministic
    workflow id (Temporal rejects duplicates) and in scan verdicts (a re-run
    cascade skips scanned sources), so re-seeding the same window is safe.
    """
    from pipeline.queues import EMBED_PRIORITY

    sql = """
        SELECT DISTINCT pi.artist_id::text
        FROM platform_identity pi
        WHERE pi.artist_id IS NOT NULL
          AND pi.platform = ANY(%(audio)s)
          AND (%(platform)s::text IS NULL OR pi.platform = %(platform)s)
        ORDER BY 1
        LIMIT %(limit)s
    """
    return [
        r[0]
        for r in conn.execute(sql, {"audio": EMBED_PRIORITY, "platform": platform, "limit": limit}).fetchall()
    ]


async def seed(platform: str | None, limit: int) -> tuple[int, int]:
    settings = Settings()
    client = await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)
    started = skipped = 0
    with psycopg.connect(settings.database_url) as conn:
        artist_ids = pending_artists(conn, platform, limit)
    for artist_id in artist_ids:
        if await start_ingest_workflow(client, artist_id, settings):
            started += 1
        else:
            skipped += 1
    return started, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="seed ingest workflows from Tier-A identities")
    ap.add_argument("--platform", help="restrict to one platform (default: all)")
    ap.add_argument("--limit", type=int, default=10, help="max workflows to start (default 10)")
    args = ap.parse_args()
    started, skipped = asyncio.run(seed(args.platform, args.limit))
    print(f"started={started} skipped={skipped}")


if __name__ == "__main__":
    main()
