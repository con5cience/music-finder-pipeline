"""YouTube discovery — EXPERIMENTAL (floor=None: scanned + recorded, never
auto-embedded; ADR-017).

Probe-verified 2026-06-10: 123,560/123,606 of our YT identities are clean UC
channel ids; channel RSS is live and keyless; yt-dlp flat extraction returns
the latest videos WITH durations in one request per channel. A sampled music
channel had 14/15 videos inside the 2-8 minute band — the duration filter is
a real music-vs-noise separator.

Safety is double-walled: rows are stored with audio_url=NULL, so they can
NEVER enter the embed path (pending_tracks requires a URL) — independent of
the floor=None wall. The watch URL + duration live in evidence; actual audio
extraction (ToS-gray, yt-dlp streams) stays design-gated for a future slice
with its own decision. yt-dlp itself is Unlicense (license-clean dep).
"""

from __future__ import annotations

from psycopg import Connection

from pipeline.sources.common import identity_row

MIN_S = 120  # below: idents/teasers/shorts
MAX_S = 480  # above: mixes/interviews/full sets
_WALK_LIMIT = 15  # newest videos considered per channel


def fetch_channel_videos(channel_id: str) -> list[dict]:
    """Newest videos w/ durations — ONE request via yt-dlp flat extraction."""
    import yt_dlp

    opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "playlist_items": f"1-{_WALK_LIMIT}",
    }
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(f"https://www.youtube.com/channel/{channel_id}/videos", download=False)
    return [e for e in (info.get("entries") or []) if e]


def music_band(entries: list[dict]) -> list[dict]:
    """The 2-8 minute filter: where music lives, mixes and teasers don't."""
    return [e for e in entries if e.get("duration") and MIN_S <= e["duration"] <= MAX_S]


def discover_youtube(conn: Connection, artist_id: str, channel_id: str, *, fetcher=None) -> int:
    """Store in-band videos as UNEMBEDDABLE candidate tracks (audio_url NULL).
    Returns NEW rows written (the scan verdict's yield)."""
    identity_id = identity_row(conn, "youtube", artist_id, channel_id)
    entries = (fetcher or fetch_channel_videos)(channel_id)
    written = 0
    for track_index, e in enumerate(music_band(entries)):
        import json

        evidence = {
            "source": "youtube_flat",
            "watch_url": f"https://www.youtube.com/watch?v={e['id']}",
            "title": e.get("title"),
            "release_index": track_index,
            "track_index": track_index,
            "experimental": True,  # floor=None: never auto-embeds
        }
        row = conn.execute(
            """
            INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s,
                                     from_identity_id, binding_tier, binding_evidence, verification_status)
            VALUES (%s, 'youtube', %s, NULL, %s, %s, 'A', %s, 'verified')
            ON CONFLICT (platform, platform_track_id) DO NOTHING
            RETURNING id
            """,
            (artist_id, str(e["id"]), int(e["duration"]), identity_id, json.dumps(evidence)),
        ).fetchone()
        if row is not None:
            written += 1
    return written
