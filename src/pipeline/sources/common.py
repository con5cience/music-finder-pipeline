"""Shared per-source plumbing (review findings: deezer/bandcamp had verbatim
copies of identity lookup, audio_track insertion, and refreshed-URL storage —
soundcloud and youtube would have made it four)."""

from __future__ import annotations

import json

from psycopg import Connection


def identity_row(conn: Connection, platform: str, artist_id: str, platform_id: str) -> str:
    """The platform_identity.id binding this (artist, platform page) — raises
    LookupError when absent (an unbound page is never discovered against)."""
    row = conn.execute(
        "SELECT id FROM platform_identity WHERE platform = %s AND platform_id = %s AND artist_id = %s",
        (platform, platform_id, artist_id),
    ).fetchone()
    if row is None:
        raise LookupError(f"no {platform} identity {platform_id} for artist {artist_id}")
    return row[0]


def insert_audio_track(
    conn: Connection,
    artist_id: str,
    platform: str,
    platform_track_id: str,
    audio_url: str,
    duration_s: int,
    identity_id: str,
    evidence: dict,
) -> bool:
    """Tier-A verified track row; True when NEW (conflict on platform+track id
    means an earlier discovery already stored it)."""
    row = conn.execute(
        """
        INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s,
                                 from_identity_id, binding_tier, binding_evidence, verification_status)
        VALUES (%s, %s, %s, %s, %s, %s, 'A', %s, 'verified')
        ON CONFLICT (platform, platform_track_id) DO NOTHING
        RETURNING id
        """,
        (artist_id, platform, platform_track_id, audio_url, duration_s, identity_id, json.dumps(evidence)),
    ).fetchone()
    return row is not None


def store_refreshed_url(conn: Connection, platform: str, platform_track_id: str, url: str) -> None:
    conn.execute(
        "UPDATE audio_track SET audio_url = %s WHERE platform = %s AND platform_track_id = %s",
        (url, platform, platform_track_id),
    )
