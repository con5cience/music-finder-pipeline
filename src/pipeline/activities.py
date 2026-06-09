"""Activities for the ingest pipeline.

`embed_artist` is real (registry-driven embed + stamped storage, ADR-016);
classify_page / bind_source remain stubs until the acquisition slice. The
embedder is a process-level lazy singleton so model weights load once per
worker, not once per activity run.
"""

from __future__ import annotations

import asyncio
import functools

import psycopg
from temporalio import activity

from pipeline.config import Settings


def _classify_sync(platform: str, platform_id: str) -> str:
    from pipeline.bind import classify_identity

    with psycopg.connect(Settings().database_url) as conn:
        return classify_identity(conn, platform, platform_id)


@activity.defn
async def classify_page(platform: str, platform_id: str) -> str:
    """Page type from identity records (MB-derived pages are pre-classified).

    'unknown' means we have no authoritative record — live classification is
    the B-tier slice's job."""
    return await asyncio.to_thread(_classify_sync, platform, platform_id)


def _bind_sync(artist_id: str, platform: str, platform_id: str) -> dict | None:
    from pipeline.bind import tier_a_binding

    with psycopg.connect(Settings().database_url) as conn:
        return tier_a_binding(conn, artist_id, platform, platform_id)


@activity.defn
async def bind_source(artist_id: str, platform: str, platform_id: str) -> dict | None:
    """Tier-A binding from MB url-rel provenance, or None when no authoritative
    link exists (search-based B-tier binding is a later slice)."""
    return await asyncio.to_thread(_bind_sync, artist_id, platform, platform_id)


def _discover_deezer_sync(artist_id: str) -> int:
    from pipeline.sources.deezer import discover_deezer

    settings = Settings()
    with psycopg.connect(settings.database_url) as conn:
        row = conn.execute(
            "SELECT platform_id FROM platform_identity WHERE platform = 'deezer' AND artist_id = %s",
            (artist_id,),
        ).fetchone()
        if row is None:
            return 0
        n = discover_deezer(conn, artist_id, row[0])
        conn.commit()
    return n


@activity.defn
async def discover_deezer_tracks(artist_id: str) -> int:
    """Discover Deezer preview tracks for a bound artist (runs on deezer-io,
    rate-capped server-side). Returns new audio_track rows written."""
    return await asyncio.to_thread(_discover_deezer_sync, artist_id)


@functools.cache
def _embedder():
    # Lazy: torch + model deps only load in workers that run this activity.
    from pipeline.embedders.registry import get_embedder

    settings = Settings()
    return get_embedder(settings.embedding_model, settings.effective_device)


def _embed_artist_sync(artist_id: str) -> int:
    from pipeline.embed_job import embed_artist_clips

    settings = Settings()
    with psycopg.connect(settings.database_url) as conn:
        n = embed_artist_clips(conn, _embedder(), artist_id)
        conn.commit()
    return n


@activity.defn
async def embed_artist(artist_id: str) -> int:
    """Embed the artist's pending tracks with the configured model (default MuQ,
    PIPELINE_EMBEDDING_MODEL to swap); store stamped clip rows + refresh the
    artist centroid. Returns the number of clips embedded."""
    return await asyncio.to_thread(_embed_artist_sync, artist_id)
