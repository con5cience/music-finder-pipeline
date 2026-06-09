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
from pipeline.seed_ingest import pending_identities, workflow_id
from pipeline.workflows import IngestArtistInput, IngestArtistWorkflow


def _real_identity(db_url: str) -> tuple[str, str, str] | None:
    with psycopg.connect(db_url) as conn:
        return conn.execute(
            """
            SELECT pi.artist_id::text, pi.platform, pi.platform_id
            FROM platform_identity pi
            WHERE NOT EXISTS (
                SELECT 1 FROM audio_track t
                WHERE t.artist_id = pi.artist_id AND t.audio_url IS NOT NULL
            )
            ORDER BY pi.platform, pi.platform_id
            LIMIT 1
            """
        ).fetchone()


async def test_tier_a_ingest_end_to_end(migrated_db):
    ident = _real_identity(migrated_db)
    if ident is None:
        pytest.skip("no bootstrapped identities (run `poe mb-bootstrap` first)")
    artist_id, platform, platform_id = ident

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
            activities=[activities.classify_page, activities.bind_source, activities.embed_artist],
        ):
            res = await env.client.execute_workflow(
                IngestArtistWorkflow.run,
                IngestArtistInput(artist_id, platform, platform_id),
                id="e2e-" + uuid.uuid4().hex,
                task_queue=tq,
            )
    assert res["status"] == "embedded"
    assert res["tier"] == "A"
    assert res["page_type"] == "artist"
    assert res["embedded"] == 0  # chosen artist has no embeddable tracks


def test_workflow_id_is_deterministic():
    assert workflow_id("deezer", "123") == "ingest-deezer-123"
    assert workflow_id("deezer", "123") == workflow_id("deezer", "123")


def test_pending_identities_filters_by_platform(conn):
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Seed Fixture', '00000000-feed-4bad-9bad-000000000777') "
        "RETURNING id"
    ).fetchone()[0]
    for plat, pid in [("deezer", "zz-seed-d1"), ("bandcamp", "zz-seed-b1")]:
        conn.execute(
            "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type) "
            "VALUES (%s, %s, %s, 'artist')",
            (a, plat, pid),
        )
    rows = pending_identities(conn, "deezer", limit=10_000_000)
    plats = {r[1] for r in rows}
    assert plats == {"deezer"}
    assert any(r[2] == "zz-seed-d1" for r in rows)
