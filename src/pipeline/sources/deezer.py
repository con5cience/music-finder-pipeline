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


def _identity_row(conn: Connection, artist_id: str, platform_id: str) -> str:
    row = conn.execute(
        "SELECT id FROM platform_identity WHERE platform = 'deezer' AND platform_id = %s AND artist_id = %s",
        (platform_id, artist_id),
    ).fetchone()
    if row is None:
        raise LookupError(f"no deezer identity {platform_id} for artist {artist_id}")
    return row[0]


def discover_deezer(
    conn: Connection,
    artist_id: str,
    platform_id: str,
    *,
    fetcher=None,
    cache_dir: Path | str | None = None,
) -> int:
    """Discover preview tracks for a Tier-A-bound artist; returns NEW rows written."""
    identity_id = _identity_row(conn, artist_id, platform_id)

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
        evidence = json.dumps(
            {"source": source, "track_duration_s": t.track_duration_s, "title": t.title}
        )
        row = conn.execute(
            """
            INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s,
                                     from_identity_id, binding_tier, binding_evidence, verification_status)
            VALUES (%s, 'deezer', %s, %s, %s, %s, 'A', %s, 'verified')
            ON CONFLICT (platform, platform_track_id) DO NOTHING
            RETURNING id
            """,
            (artist_id, t.track_id, t.preview_url, _PREVIEW_S, identity_id, evidence),
        ).fetchone()
        if row is not None:
            written += 1
    return written
