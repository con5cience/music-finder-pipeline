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


@activity.defn
async def classify_page(platform_id: str) -> str:
    """Return the page_type for a platform page (artist/label/compilation/topic). STUB."""
    return "artist"


@activity.defn
async def bind_source(artist_id: str, platform: str, platform_id: str) -> dict:
    """Bind a source to an artist under a verification tier (A/B1/B2/C). STUB → Tier A."""
    return {"tier": "A", "track_count": 0}


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
