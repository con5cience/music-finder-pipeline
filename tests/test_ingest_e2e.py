"""End-to-end ingest: REAL activities (DB-backed classify/bind + embed) under
the time-skipping Temporal env, against a REAL bootstrapped identity.

Read-only by construction: we pick an artist with no embeddable tracks, so the
workflow exercises classify → Tier-A bind → embed(0 pending) without writing.
Skips when the Temporal test server or bootstrapped data is unavailable.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from pipeline import activities
from pipeline.seed_ingest import pending_artists, workflow_id
from pipeline.workflows import IngestArtistInput, IngestArtistWorkflow


def _artist_without_audio_identities(db_url: str) -> str | None:
    # An artist whose identities are ALL non-audio platforms (tidal/apple/...):
    # the cascade has nothing to scan or embed → terminal 'unbound', no IO.
    from pipeline.queues import EMBED_PRIORITY

    with psycopg.connect(db_url) as conn:
        row = conn.execute(
            """
            SELECT pi.artist_id::text
            FROM platform_identity pi
            GROUP BY pi.artist_id
            HAVING bool_and(pi.platform != ALL(%s))
            ORDER BY pi.artist_id
            LIMIT 1
            """,
            (EMBED_PRIORITY,),
        ).fetchone()
        return row[0] if row else None


async def test_cascade_ingest_end_to_end_unbound(migrated_db, monkeypatch):
    # Real activities against the TEST db (env pinned — activities read
    # Settings()), with a SYNTHETIC artist (review altitude finding: no more
    # corpus archaeology; the test owns its fixture and never skips).
    monkeypatch.setenv("PIPELINE_DATABASE_URL", migrated_db)
    with psycopg.connect(migrated_db) as setup:
        setup.execute(
            "DELETE FROM artist WHERE mbid = '00000000-feed-4bad-9bad-000000000e2e'")
        artist_id = str(setup.execute(
            "INSERT INTO artist (display_name, mbid) VALUES ('E2E Fixture', "
            "'00000000-feed-4bad-9bad-000000000e2e') RETURNING id").fetchone()[0])
        setup.execute(
            "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type) "
            "VALUES (%s, 'tidal', 'zz-e2e-tidal', 'artist') ON CONFLICT DO NOTHING",
            (artist_id,))
        setup.commit()

    try:
        env = await WorkflowEnvironment.start_time_skipping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Temporal test server unavailable: {exc}")

    async with env:
        tq = "e2e-" + uuid.uuid4().hex
        async with Worker(
            env.client,
            task_queue=tq,
            workflows=[IngestArtistWorkflow],
            activities=[
                activities.cascade_plan,
                activities.record_scan,
                activities.choose_embed_source,
                activities.embed_artist,
            ],
        ):
            res = await env.client.execute_workflow(
                IngestArtistWorkflow.run,
                IngestArtistInput(artist_id),
                id="e2e-" + uuid.uuid4().hex,
                task_queue=tq,
            )
    assert res == {"status": "unbound"}


def test_workflow_id_is_deterministic():
    assert workflow_id("abc-123") == "ingest-artist-abc-123"
    assert workflow_id("abc-123") == workflow_id("abc-123")


def test_pending_artists_audio_role_only(conn):
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Seed Fixture', '00000000-feed-4bad-9bad-000000000777') "
        "RETURNING id"
    ).fetchone()[0]
    a2 = conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Tidal Only', '00000000-feed-4bad-9bad-000000000778') "
        "RETURNING id"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type) "
        "VALUES (%s, 'deezer', 'zz-seed-d1', 'artist'), (%s, 'tidal', 'zz-seed-t1', 'artist')",
        (a, a2),
    )
    ids = pending_artists(conn, None, limit=10_000_000)
    assert str(a) in ids
    assert str(a2) not in ids  # tidal-only: playback asset, no audio cascade
