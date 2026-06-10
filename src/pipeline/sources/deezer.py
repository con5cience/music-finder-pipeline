"""Deezer track discovery — the embedding-primary source (ADR-017 §2).

Top tracks first (`/artist/{id}/top?limit=12`), albums walk as the long-tail
fallback (sibling forensics: ~67% of long-tail artists have no top tracks, and
~78% of those recover via albums). Source-correctness (ADR-015): a track
counts only when the bound artist IS the track's main artist — features and
compilation appearances are contamination, on both paths.

audio_track.duration_s is the duration of the audio AT audio_url (the 30s
preview); the full track duration is preserved in binding_evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from psycopg import Connection

from pipeline.fetch_cache import cached_fetch
from pipeline.sources.common import identity_row, insert_audio_track, store_refreshed_url

_API = "https://api.deezer.com"
_TOP_LIMIT = 12  # ADR-017 sampling policy: 10-12 previews per artist
_ALBUM_WALK_LIMIT = 5
_PREVIEW_S = 30


@dataclass(frozen=True)
class Track:
    track_id: str
    title: str
    preview_url: str
    track_duration_s: int


def parse_tracks(body: bytes, artist_platform_id: str) -> list[Track]:
    """Extract main-artist tracks with previews from a Deezer track-list payload."""
    payload = json.loads(body)
    out: list[Track] = []
    for t in payload.get("data", []):
        if str(t.get("artist", {}).get("id")) != str(artist_platform_id):
            continue  # source-correctness: not this artist's track
        if not t.get("preview"):
            continue
        out.append(Track(str(t["id"]), t.get("title", ""), t["preview"], int(t.get("duration") or 0)))
    return out


def refresh_preview(
    conn: Connection,
    platform_track_id: str,
    *,
    fetcher=None,
    cache_dir: Path | str | None = None,
) -> str | None:
    """Re-resolve a track's preview URL live (signed URLs expire: hdnea/hmac).

    Updates audio_track.audio_url and returns the fresh URL, or None when the
    track no longer offers a preview. Bypasses the cache READ by design — the
    cached payload is exactly what went stale.
    """
    body = cached_fetch(
        conn, "deezer", f"{_API}/track/{platform_track_id}", fetcher=fetcher, cache_dir=cache_dir, refresh=True
    ).body
    preview = json.loads(body).get("preview") or None
    if preview:
        store_refreshed_url(conn, "deezer", platform_track_id, preview)
    return preview


def discover_deezer(
    conn: Connection,
    artist_id: str,
    platform_id: str,
    *,
    fetcher=None,
    cache_dir: Path | str | None = None,
) -> int:
    """Discover preview tracks for a Tier-A-bound artist; returns NEW rows written."""
    identity_id = identity_row(conn, "deezer", artist_id, platform_id)

    def get(path: str) -> bytes:
        return cached_fetch(conn, "deezer", f"{_API}/{path}", fetcher=fetcher, cache_dir=cache_dir).body

    tracks = parse_tracks(get(f"artist/{platform_id}/top?limit={_TOP_LIMIT}"), platform_id)
    source = "deezer_top"
    if not tracks:
        source = "deezer_albums"
        albums = json.loads(get(f"artist/{platform_id}/albums?limit={_ALBUM_WALK_LIMIT}")).get("data", [])
        for album in albums[:_ALBUM_WALK_LIMIT]:
            tracks.extend(parse_tracks(get(f"album/{album['id']}/tracks?limit=50"), platform_id))
            if len(tracks) >= _TOP_LIMIT:
                break
        tracks = tracks[:_TOP_LIMIT]

    written = 0
    for t in tracks:
        evidence = {"source": source, "track_duration_s": t.track_duration_s, "title": t.title}
        if insert_audio_track(
            conn, artist_id, "deezer", t.track_id, t.preview_url, _PREVIEW_S, identity_id, evidence
        ):
            written += 1
    return written
