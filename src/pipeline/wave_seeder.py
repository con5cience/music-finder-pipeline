"""Mass-scale wave seeder: keep a bounded window of cascades in flight.

Never seeds the whole corpus at once — the dev Temporal server's history is
the one component we haven't scale-hardened, so the window stays small and
constant (seed BATCH more whenever Running < LOW_WATER, up to --total).
Every component underneath is the tested machinery: deterministic workflow
ids, terminal scan verdicts, budget caps, self-healing fetches. Kill it any
time; re-running resumes exactly where the ledgers say.

Run:  uv run poe wave-seed -- --total 20000 --batch 500 --low-water 200
Watch: uv run poe factory-status -- --watch
"""

from __future__ import annotations

import asyncio
import time

# Audio platforms whose discovery queues run at interactive rates. youtube is
# deliberately absent: its queue is server-capped at 0.1/s (politeness law), so
# a yt-only artist occupies a window slot for HOURS while these need minutes.
_FAST_PLATFORMS = ("deezer", "bandcamp", "soundcloud")

# Hard ceiling on the seeded window. The window is the ONE knob that can melt
# the un-scale-hardened dev Temporal: low-water 2000 / batch 1000 (a one-line
# compose edit) kept ~2000 workflows in flight and flatlined embeds for 6.5h on
# 2026-06-15. The ceiling lives in code so an arg can't undo the lesson.
MAX_LOW_WATER = 500
MAX_BATCH = 500


def clamp_window(low_water: int, batch: int) -> tuple[int, int]:
    """Clamp the window args to the safe ceiling, warning loudly if they were
    over-sized. Returns (low_water, batch)."""
    cl, cb = min(low_water, MAX_LOW_WATER), min(batch, MAX_BATCH)
    if cl != low_water or cb != batch:
        print(
            f"WARNING: window clamped to low_water={cl} batch={cb} "
            f"(requested {low_water}/{batch}; ceiling {MAX_LOW_WATER}/{MAX_BATCH} — "
            "larger windows melt the dev Temporal, see 2026-06-15)",
            flush=True,
        )
    return cl, cb


def select_seed_batch(conn, limit: int) -> list[str]:
    """Next artists to seed: fast lanes first, yt-only as last resort.

    One mixed query melted the factory on 2026-06-11: every batch carried its
    natural share of yt-only artists (~8%), each parking in the window for
    hours behind the 0.1/s youtube queue, until the whole window was
    yt-waiters and throughput pinned to the yt ceiling (~300/hr). Seeding
    yt-only artists ONLY when no fast-lane work remains mirrors the cascade's
    own floor=4 last-resort ranking for youtube. Discovery artists (mbid NULL,
    ADR-019 provisional) still sort first within each lane: the trickle is
    dozens/day and must surface in hours, not at corpus-completion.
    """
    rows = [r[0] for r in conn.execute(
        """
        SELECT DISTINCT pi.artist_id::text, (a.mbid IS NULL) AS prio
        FROM platform_identity pi
        JOIN artist a ON a.id = pi.artist_id
        WHERE pi.platform = ANY(%s)
          AND pi.scan_status = 'pending'
          AND a.embedding_source IS NULL
        ORDER BY prio DESC, 1 LIMIT %s
        """, (list(_FAST_PLATFORMS), limit),
    ).fetchall()]
    if len(rows) < limit:  # fast lane drained — top up with the yt-only cohort
        rows += [r[0] for r in conn.execute(
            """
            SELECT DISTINCT pi.artist_id::text, (a.mbid IS NULL) AS prio
            FROM platform_identity pi
            JOIN artist a ON a.id = pi.artist_id
            WHERE pi.platform = 'youtube'
              AND pi.scan_status = 'pending'
              AND a.embedding_source IS NULL
              AND NOT EXISTS (
                SELECT 1 FROM platform_identity p2
                WHERE p2.artist_id = pi.artist_id
                  AND p2.platform = ANY(%s)
                  AND p2.scan_status = 'pending')
            ORDER BY prio DESC, 1 LIMIT %s
            """, (list(_FAST_PLATFORMS), limit - len(rows)),
        ).fetchall()]
    return rows


async def run(total: int, batch: int, low_water: int) -> None:
    import psycopg
    from temporalio.client import Client

    from pipeline.config import Settings
    from pipeline.seed_ingest import workflow_id
    from pipeline.workflows import IngestArtistInput, IngestArtistWorkflow

    settings = Settings()
    client = await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)
    seeded = 0
    while seeded < total:
        running = 0
        async for _ in client.list_workflows('ExecutionStatus="Running" AND WorkflowId STARTS_WITH "ingest-artist-"'):
            running += 1
            if running >= low_water:
                break
        if running >= low_water:
            await asyncio.sleep(30)
            continue
        with psycopg.connect(settings.database_url) as conn:
            # natural pagination: completed artists drop out (embedded, or
            # every audio identity carries a terminal verdict).
            rows = select_seed_batch(conn, min(batch, total - seeded))
        if not rows:
            print(f"corpus exhausted at {seeded} seeded — done", flush=True)
            return
        started = 0
        for artist_id in rows:
            try:
                await client.start_workflow(
                    IngestArtistWorkflow.run, IngestArtistInput(str(artist_id)),
                    id=workflow_id(str(artist_id)), task_queue=settings.temporal_task_queue,
                )
                started += 1
            except Exception:  # noqa: BLE001 — already-started is fine, skip
                pass
        seeded += started
        print(f"{time.strftime('%H:%M:%S')} wave: +{started} (total {seeded}/{total}, running≈{running})", flush=True)
        await asyncio.sleep(10)
    print(f"target reached: {seeded} seeded", flush=True)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="bounded-window mass seeder")
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--low-water", type=int, default=200)
    args = ap.parse_args()
    low_water, batch = clamp_window(args.low_water, args.batch)
    asyncio.run(run(args.total, batch, low_water))


if __name__ == "__main__":
    main()
