"""Activities for the ingest pipeline.

Cascade activities (plan/scan/choose) are thin DB reads; embed_artist is the
GPU pass (embed + analysis heads + tags). Models are process-level lazy
singletons so weights load once per worker, not once per activity run.
(classify_page/bind_source were removed with the per-identity workflow —
B-tier binding returns in slice 3d with evidence scoring, per ADR-017.)
"""

from __future__ import annotations

import asyncio
import functools
import logging

import psycopg
from temporalio import activity

from pipeline.config import Settings


def _cascade_plan_sync(artist_id: str) -> dict:
    from pipeline.cascade import audio_identities

    with psycopg.connect(Settings().database_url) as conn:
        idents = audio_identities(conn, artist_id)
    return {
        "has_audio_identities": bool(idents),
        "pending": [[p, pid] for p, pid, status in idents if status == "pending"],
    }


@activity.defn
async def cascade_plan(artist_id: str) -> dict:
    """The artist's audio-role identities in cascade order + which still need
    scanning. Non-audio platforms (tidal/apple/qobuz) never appear."""
    return await asyncio.to_thread(_cascade_plan_sync, artist_id)


def _record_scan_sync(artist_id: str, platform: str, platform_id: str) -> int:
    from pipeline.cascade import mark_scanned, source_yields

    with psycopg.connect(Settings().database_url) as conn:
        total = source_yields(conn, artist_id).get(platform, 0)
        mark_scanned(conn, platform, platform_id, total)
        conn.commit()
    return total


@activity.defn
async def record_scan(artist_id: str, platform: str, platform_id: str) -> int:
    """Write the terminal scan verdict for an identity; returns the platform's
    TOTAL embeddable yield (not just newly-discovered) for floor decisions."""
    return await asyncio.to_thread(_record_scan_sync, artist_id, platform, platform_id)


def _choose_embed_source_sync(artist_id: str) -> dict | None:
    from pipeline.cascade import choose_source, source_yields

    with psycopg.connect(Settings().database_url) as conn:
        choice = choose_source(source_yields(conn, artist_id))
    if choice is None:
        return None
    return {"source": choice[0], "ratio": choice[1]}


@activity.defn
async def choose_embed_source(artist_id: str) -> dict | None:
    """Pick the artist's embedding source: floor-met by priority, else best
    floor-ratio thin source; None when nothing usable exists anywhere."""
    return await asyncio.to_thread(_choose_embed_source_sync, artist_id)


def _resolve_platform_id(conn, platform: str, artist_id: str, platform_id: str | None) -> str | None:
    """Use the cascade-supplied identity; fall back to lookup only for legacy
    calls. Review finding: fetchone()-an-arbitrary-identity scanned the wrong
    subdomain for artists with 2+ identities on one platform."""
    if platform_id is not None:
        return platform_id
    row = conn.execute(
        "SELECT platform_id FROM platform_identity WHERE platform = %s AND artist_id = %s",
        (platform, artist_id),
    ).fetchone()
    return row[0] if row else None


def _discover_deezer_sync(artist_id: str, platform_id: str | None) -> int:
    from pipeline.sources.deezer import discover_deezer

    settings = Settings()
    with psycopg.connect(settings.database_url) as conn:
        pid = _resolve_platform_id(conn, "deezer", artist_id, platform_id)
        if pid is None:
            return 0
        n = discover_deezer(conn, artist_id, pid)
        conn.commit()
    return n


@activity.defn
async def discover_deezer_tracks(artist_id: str, platform_id: str | None = None) -> int:
    """Discover Deezer preview tracks for ONE bound identity (runs on
    deezer-io, rate-capped server-side). Returns new audio_track rows."""
    return await asyncio.to_thread(_discover_deezer_sync, artist_id, platform_id)


def _discover_bandcamp_sync(artist_id: str, platform_id: str | None) -> int:
    from pipeline.sources.bandcamp import discover_bandcamp

    settings = Settings()
    with psycopg.connect(settings.database_url) as conn:
        pid = _resolve_platform_id(conn, "bandcamp", artist_id, platform_id)
        if pid is None:
            return 0
        n = discover_bandcamp(conn, artist_id, pid)
        conn.commit()
    return n


@activity.defn
async def discover_bandcamp_tracks(artist_id: str, platform_id: str | None = None) -> int:
    """Walk ONE Bandcamp identity's discography (rate-capped on bandcamp-io);
    store ALL streamable tracks. Returns new audio_track rows written."""
    return await asyncio.to_thread(_discover_bandcamp_sync, artist_id, platform_id)


def _discover_soundcloud_sync(artist_id: str, platform_id: str | None) -> int:
    from pipeline.sources.soundcloud import discover_soundcloud

    settings = Settings()
    with psycopg.connect(settings.database_url) as conn:
        pid = _resolve_platform_id(conn, "soundcloud", artist_id, platform_id)
        if pid is None:
            return 0
        n = discover_soundcloud(conn, artist_id, pid)
        conn.commit()
    return n


@activity.defn
async def discover_soundcloud_tracks(artist_id: str, platform_id: str | None = None) -> int:
    """List ONE SoundCloud identity's newest tracks via the official API
    (rate-capped on soundcloud-io); stores 30s intro previews (the app-only
    API tier streams nothing longer — see sources/soundcloud.py)."""
    return await asyncio.to_thread(_discover_soundcloud_sync, artist_id, platform_id)


def _discover_youtube_sync(artist_id: str, platform_id: str | None) -> int:
    from pipeline.sources.youtube import discover_youtube

    settings = Settings()
    with psycopg.connect(settings.database_url) as conn:
        pid = _resolve_platform_id(conn, "youtube", artist_id, platform_id)
        if pid is None:
            return 0
        n = discover_youtube(conn, artist_id, pid)
        conn.commit()
    return n


@activity.defn
async def discover_youtube_tracks(artist_id: str, platform_id: str | None = None) -> int:
    """EXPERIMENTAL: flat-extract ONE channel's newest videos (0.1/s budget),
    store the 2-8min band as unembeddable candidates (audio_url NULL)."""
    return await asyncio.to_thread(_discover_youtube_sync, artist_id, platform_id)


def _prep_sync(artist_id: str, source: str | None) -> int:
    from pipeline.embedders.registry import DEFAULT_EMBEDDER
    from pipeline.staging import prep_artist

    settings = Settings()
    model = settings.embedding_model or DEFAULT_EMBEDDER
    with psycopg.connect(settings.database_url) as conn:
        n = prep_artist(conn, artist_id, source, model)
        conn.commit()
    return n


@activity.defn
async def prep_artist_clips(artist_id: str, source: str | None = None) -> int:
    """Throughput campaign: CPU staging (fetch/decode/windows/CPU heads) on
    the prep queue so the GPU lane runs pure inference."""
    return await asyncio.to_thread(_prep_sync, artist_id, source)


def _embed_staged_sync(artist_id: str, source: str | None, ratio: float | None) -> int:
    from pipeline.heads import build_heads
    from pipeline.staging import embed_staged

    settings = Settings()
    with psycopg.connect(settings.database_url) as conn:
        n = embed_staged(conn, _embedder(), artist_id, source, ratio,
                         heads=build_heads(_tag_scorer()))
        conn.commit()
    return n


@activity.defn
async def embed_artist_staged(artist_id: str, source: str | None = None, ratio: float | None = None) -> int:
    """Pure-inference embed from the staged manifest (legacy fallback inside)."""
    return await asyncio.to_thread(_embed_staged_sync, artist_id, source, ratio)


@functools.cache
def _embedder():
    # Lazy: torch + model deps only load in workers that run this activity.
    from pipeline.embedders.registry import get_embedder

    settings = Settings()
    return get_embedder(settings.embedding_model, settings.effective_device)


_tag_scorer_memo: list = []  # [MulanTagScorer] once successfully built


def _tag_scorer():
    """Lazy per-process scorer; the vocabulary matrix embeds once and is
    reused. Deliberately NOT functools.cache: an empty vocabulary (worker
    started before the genre tables were loaded) must not be memoized as
    None forever — we warn loudly and re-check on the next artist (review
    finding: tags were silently disabled for the process lifetime)."""
    if _tag_scorer_memo:
        return _tag_scorer_memo[0]
    from pipeline.tags import MulanTagScorer, load_vocabulary

    settings = Settings()
    with psycopg.connect(settings.database_url) as conn:
        vocab = load_vocabulary(conn)
    if not vocab:
        logging.getLogger(__name__).warning(
            "tag vocabulary empty (mb_raw.genre) — tag head SKIPPED for this artist; "
            "load the genre tables (poe mb-bootstrap) to enable tags"
        )
        return None  # not memoized — recovers as soon as the vocabulary exists
    scorer = MulanTagScorer(vocab)
    _tag_scorer_memo.append(scorer)
    return scorer


def _embed_artist_sync(artist_id: str, source: str | None, ratio: float | None) -> int:
    from pipeline.embed_job import embed_artist_clips
    from pipeline.heads import build_heads

    settings = Settings()
    with psycopg.connect(settings.database_url) as conn:
        n = embed_artist_clips(
            conn, _embedder(), artist_id, source, ratio, heads=build_heads(_tag_scorer())
        )
        conn.commit()
    return n


@activity.defn
async def embed_artist(artist_id: str, source: str | None = None, ratio: float | None = None) -> int:
    """Embed the artist's pending tracks from ONE source (centroid purity) with
    the configured model (default MuQ, PIPELINE_EMBEDDING_MODEL to swap); store
    stamped clips, refresh the centroid with its signal_ratio, lock
    artist.embedding_source. Returns the number of clips embedded."""
    return await asyncio.to_thread(_embed_artist_sync, artist_id, source, ratio)
