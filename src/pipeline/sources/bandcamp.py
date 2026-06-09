"""Bandcamp track discovery — full-track fallback source (ADR-017 §2).

Tier-A subdomains come from MB url-rels; discovery walks the artist's
discography (`/music`, newest-first as rendered; root-page fallback for
single-release artists), parses each release page's `data-tralbum` JSON, and
stores ALL streamable tracks (playback derivation wants the catalog; the
embedder selects its 3). Stream URLs carry ~24h expiry tokens (`ts=`) —
`refresh_bandcamp` re-resolves via a cache-bypassing page re-fetch, same
self-healing pattern as Deezer.

Source-correctness note: the subdomain IS the artist's page (Tier-A claim), so
tralbum-level attribution is accepted; per-track `artist` overrides (typical on
compilations) are recorded in evidence for the calibration audit, not dropped.
"""

from __future__ import annotations

import html as _html
import json
import re
from dataclasses import dataclass
from pathlib import Path

from psycopg import Connection

from pipeline.fetch_cache import cached_fetch

_RELEASE_WALK_LIMIT = 5
_HREF_RE = re.compile(r'href="(/(?:album|track)/[^"#?]+)"')
_TRALBUM_RE = re.compile(r'data-tralbum="([^"]+)"')


@dataclass(frozen=True)
class BcTrack:
    track_id: str
    title: str
    duration_s: int
    stream_url: str
    track_artist: str | None  # per-track override when it differs from the page


def parse_discography(body: bytes) -> list[str]:
    """Release paths in DOCUMENT order (the page renders newest-first)."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _HREF_RE.finditer(body.decode("utf-8", errors="replace")):
        p = m.group(1)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def parse_tralbum(body: bytes) -> dict | None:
    """{artist, release_date, tracks: [BcTrack]} from a release page, or None."""
    m = _TRALBUM_RE.search(body.decode("utf-8", errors="replace"))
    if m is None:
        return None
    d = json.loads(_html.unescape(m.group(1)))
    page_artist = d.get("artist")
    tracks: list[BcTrack] = []
    for t in d.get("trackinfo") or []:
        stream = (t.get("file") or {}).get("mp3-128")
        if not stream or not t.get("duration"):
            continue  # not streamable / no audio
        tid = t.get("track_id") or t.get("id")
        if tid is None:
            continue
        tracks.append(
            BcTrack(
                track_id=str(tid),
                title=t.get("title") or "",
                duration_s=int(t["duration"]),
                stream_url=stream,
                track_artist=t.get("artist") if t.get("artist") and t.get("artist") != page_artist else None,
            )
        )
    return {"artist": page_artist, "release_date": d.get("album_release_date"), "tracks": tracks}


def _identity_row(conn: Connection, artist_id: str, subdomain: str) -> str:
    row = conn.execute(
        "SELECT id FROM platform_identity WHERE platform = 'bandcamp' AND platform_id = %s AND artist_id = %s",
        (subdomain, artist_id),
    ).fetchone()
    if row is None:
        raise LookupError(f"no bandcamp identity {subdomain} for artist {artist_id}")
    return row[0]


def discover_bandcamp(
    conn: Connection,
    artist_id: str,
    subdomain: str,
    *,
    fetcher=None,
    cache_dir: Path | str | None = None,
) -> int:
    """Walk newest releases, store ALL streamable tracks; returns NEW rows."""
    identity_id = _identity_row(conn, artist_id, subdomain)
    base = f"https://{subdomain}.bandcamp.com"

    def get(path: str) -> bytes:
        return cached_fetch(conn, "bandcamp", base + path, fetcher=fetcher, cache_dir=cache_dir).body

    releases = parse_discography(get("/music"))[:_RELEASE_WALK_LIMIT]
    if not releases:
        releases = [""]  # single-release artists: the root page IS the album

    written = 0
    for release_index, path in enumerate(releases):
        parsed = parse_tralbum(get(path))
        if parsed is None:
            continue
        for track_index, t in enumerate(parsed["tracks"]):
            evidence: dict = {
                "source": "bandcamp_tralbum",
                "album_path": path or "/",
                # walk order IS the newest-first truth; selection sorts by
                # (release_index, track_index) — audio_track.id is a uuid, so
                # physical row order is a lottery
                "release_index": release_index,
                "track_index": track_index,
                "release_date": parsed["release_date"],
                "title": t.title,
            }
            if t.track_artist:
                evidence["track_artist_override"] = t.track_artist  # calibration audit, not a drop
            row = conn.execute(
                """
                INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s,
                                         from_identity_id, binding_tier, binding_evidence, verification_status)
                VALUES (%s, 'bandcamp', %s, %s, %s, %s, 'A', %s, 'verified')
                ON CONFLICT (platform, platform_track_id) DO NOTHING
                RETURNING id
                """,
                (artist_id, t.track_id, t.stream_url, t.duration_s, identity_id, json.dumps(evidence)),
            ).fetchone()
            if row is not None:
                written += 1
    return written


def refresh_bandcamp(
    conn: Connection,
    platform_track_id: str,
    *,
    fetcher=None,
    cache_dir: Path | str | None = None,
) -> str | None:
    """Re-resolve an expired stream URL by re-fetching its release page live."""
    row = conn.execute(
        """
        SELECT t.binding_evidence->>'album_path', pi.platform_id
        FROM audio_track t JOIN platform_identity pi ON pi.id = t.from_identity_id
        WHERE t.platform = 'bandcamp' AND t.platform_track_id = %s
        """,
        (platform_track_id,),
    ).fetchone()
    if row is None:
        return None
    album_path, subdomain = row
    url = f"https://{subdomain}.bandcamp.com" + (album_path if album_path != "/" else "")
    parsed = parse_tralbum(
        cached_fetch(conn, "bandcamp", url, fetcher=fetcher, cache_dir=cache_dir, refresh=True).body
    )
    if parsed is None:
        return None
    fresh = next((t.stream_url for t in parsed["tracks"] if t.track_id == platform_track_id), None)
    if fresh:
        conn.execute(
            "UPDATE audio_track SET audio_url = %s WHERE platform = 'bandcamp' AND platform_track_id = %s",
            (fresh, platform_track_id),
        )
    return fresh
