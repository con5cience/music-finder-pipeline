"""Identity classification + Tier-A binding (ADR-017 §3, DB-truth path).

MB-derived identities were classified at bootstrap (page_type='artist') and
carry url-rel provenance, so classification and Tier-A binding are pure DB
reads — no platform I/O. Search-based binding (B1/B2/C, evidence scoring)
lands in the calibration slice and will extend these functions, not replace
them: an authoritative identity always short-circuits search.
"""

from __future__ import annotations

from psycopg import Connection


def classify_identity(conn: Connection, platform: str, platform_id: str) -> str:
    """Return the page_type for a platform page ('unknown' if we've never seen it)."""
    row = conn.execute(
        "SELECT page_type FROM platform_identity WHERE platform = %s AND platform_id = %s",
        (platform, platform_id),
    ).fetchone()
    return row[0] if row else "unknown"


def tier_a_binding(conn: Connection, artist_id: str, platform: str, platform_id: str) -> dict | None:
    """Tier-A binding when the identity exists AND belongs to this artist.

    Returns {"tier": "A", "evidence": {...}} or None (no authoritative link —
    the B-tier search path's job, not ours). Cross-artist identities refuse to
    bind: an identity derived for one artist is never evidence for another.
    """
    row = conn.execute(
        "SELECT artist_id, vanity_url FROM platform_identity WHERE platform = %s AND platform_id = %s",
        (platform, platform_id),
    ).fetchone()
    if row is None or str(row[0]) != str(artist_id):
        return None
    return {
        "tier": "A",
        "evidence": {"source": "mb_url_rel", "platform": platform, "platform_id": platform_id, "vanity_url": row[1]},
    }
