"""Activity stubs for the ingest pipeline.

Real implementations (platform I/O, page classification, tiered binding, GPU
CLAP) land in later slices. These establish the names + signatures the workflow
orchestrates and let the workflow be tested end-to-end against mocks.
"""

from __future__ import annotations

from temporalio import activity


@activity.defn
async def classify_page(platform_id: str) -> str:
    """Return the page_type for a platform page (artist/label/compilation/topic). STUB."""
    return "artist"


@activity.defn
async def bind_source(artist_id: str, platform: str, platform_id: str) -> dict:
    """Bind a source to an artist under a verification tier (A/B1/B2/C). STUB → Tier A."""
    return {"tier": "A", "track_count": 0}


@activity.defn
async def clap_embed(artist_id: str) -> int:
    """Download + CLAP-embed the artist's tracks; return the count embedded. STUB → 0."""
    return 0
