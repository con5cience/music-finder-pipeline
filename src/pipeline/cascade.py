"""Audio-source cascade (ADR-017 §2 + floors): which source embeds an artist.

Pure DB functions; the workflow orchestrates, these decide. Floors are both
gate and unit conversion: a source "wins" outright by meeting its floor in
priority order; when none does, the best THIN source wins by floor-ratio
(yield/floor — 2 Bandcamp tracks at 2/3 beat 1 Deezer preview at 1/10).
Experimental sources (floor None) are scanned and recorded, never auto-chosen.
"""

from __future__ import annotations

from psycopg import Connection

from pipeline.queues import EMBED_FLOORS, EMBED_PRIORITY


def audio_identities(conn: Connection, artist_id: str) -> list[tuple[str, str, str]]:
    """(platform, platform_id, scan_status) for the artist's audio-role
    identities, in cascade priority order. Non-audio platforms never appear."""
    rows = conn.execute(
        "SELECT platform, platform_id, scan_status FROM platform_identity "
        "WHERE artist_id = %s AND platform = ANY(%s)",
        (artist_id, EMBED_PRIORITY),
    ).fetchall()
    order = {p: i for i, p in enumerate(EMBED_PRIORITY)}
    return sorted(rows, key=lambda r: order[r[0]])


def source_yields(conn: Connection, artist_id: str) -> dict[str, int]:
    """platform → discovered-track count. Embeddability is audio_url IS NOT
    NULL for audio platforms; youtube candidates are NULL BY DESIGN (ADR-017
    extraction gate) yet a fruitful scan must still verdict 'scanned' — the
    old embeddable-only count recorded 846 video-bearing channels as 'empty'.
    choose_source() never picks youtube (no floor entry), so counting its
    candidates here cannot leak into embed decisions."""
    return dict(
        conn.execute(
            "SELECT platform, count(*) FROM audio_track "
            "WHERE artist_id = %s "
            "  AND (audio_url IS NOT NULL OR platform = 'youtube') "
            "  AND verification_status NOT IN ('rejected','quarantined') "
            "GROUP BY platform",
            (artist_id,),
        ).fetchall()
    )


def floor_ratio(platform: str, yield_n: int) -> float | None:
    """yield/floor, or None for experimental sources (never auto-chosen)."""
    floor = EMBED_FLOORS.get(platform)
    if floor is None:
        return None
    return yield_n / floor


def choose_source(yields: dict[str, int]) -> tuple[str, float] | None:
    """The source that embeds this artist, or None when nothing qualifies.

    1. First source IN PRIORITY ORDER meeting its floor wins (ratio >= 1).
    2. Else best thin source by floor-ratio (> 0) wins.
    Experimental (floor None) sources never qualify on either path.
    """
    for platform in EMBED_PRIORITY:
        r = floor_ratio(platform, yields.get(platform, 0))
        if r is not None and r >= 1.0:
            return platform, r
    best: tuple[str, float] | None = None
    for platform in EMBED_PRIORITY:
        r = floor_ratio(platform, yields.get(platform, 0))
        if r is not None and r > 0 and (best is None or r > best[1]):
            best = (platform, r)
    return best


def mark_scanned(conn: Connection, platform: str, platform_id: str, yield_n: int) -> None:
    """Record the terminal scan verdict for an identity ('scanned' or 'empty').

    Transient failures must NOT call this — 'pending' is the retry state.
    """
    status = "scanned" if yield_n > 0 else "empty"
    conn.execute(
        "UPDATE platform_identity SET scan_status = %s, scanned_at = now() "
        "WHERE platform = %s AND platform_id = %s",
        (status, platform, platform_id),
    )
