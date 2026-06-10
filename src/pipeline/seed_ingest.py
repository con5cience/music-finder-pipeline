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

import psycopg
from temporalio.client import Client
from temporalio.exceptions import TemporalError

from pipeline.config import Settings
from pipeline.workflows import IngestArtistInput, IngestArtistWorkflow


def workflow_id(artist_id: str) -> str:
    return f"ingest-artist-{artist_id}"


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
        try:
            await client.start_workflow(
                IngestArtistWorkflow.run,
                IngestArtistInput(artist_id),
                id=workflow_id(artist_id),
                task_queue=settings.temporal_task_queue,
            )
            started += 1
        except TemporalError:  # already-started → idempotent skip
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
