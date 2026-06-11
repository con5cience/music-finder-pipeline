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
            # every audio identity carries a terminal verdict). Discovery
            # artists (mbid NULL — ADR-019 provisional identity) sort FIRST:
            # the trickle is dozens/day and must surface in hours, not at
            # corpus-completion; the corpus pays minutes for it.
            rows = [r[0] for r in conn.execute(
                """
                SELECT DISTINCT pi.artist_id::text, (a.mbid IS NULL) AS prio
                FROM platform_identity pi
                JOIN artist a ON a.id = pi.artist_id
                WHERE pi.platform IN ('deezer','bandcamp','soundcloud')
                  AND pi.scan_status = 'pending'
                  AND a.embedding_source IS NULL
                ORDER BY prio DESC, 1 LIMIT %s
                """, (min(batch, total - seeded),),
            ).fetchall()]
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
    asyncio.run(run(args.total, args.batch, args.low_water))


if __name__ == "__main__":
    main()
