"""YT experimental discovery: 2-8min band filter, the audio_url=NULL safety
wall (rows must NEVER be embeddable), walk order, idempotency."""

from __future__ import annotations

from pipeline.embed_job import pending_tracks
from pipeline.sources.youtube import MAX_S, MIN_S, discover_youtube, music_band

MBID = "00000000-feed-4bad-9bad-000000000999"


def _artist(conn) -> str:
    return conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('YT Fixture', %s) RETURNING id", (MBID,)
    ).fetchone()[0]


def _identity(conn, a, cid="UCzzfixture0000000000000"):
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type) "
        "VALUES (%s, 'youtube', %s, 'artist')",
        (a, cid),
    )


ENTRIES = [
    {"id": "zzvid-short", "duration": 45, "title": "teaser"},          # below band
    {"id": "zzvid-song1", "duration": 212, "title": "song one"},
    {"id": "zzvid-song2", "duration": 300, "title": "song two"},
    {"id": "zzvid-mix", "duration": 3600, "title": "live set 1h"},     # above band
    {"id": "zzvid-nodur", "title": "premiere"},                        # no duration
]


def test_music_band_filter():
    band = music_band(ENTRIES)
    assert [e["id"] for e in band] == ["zzvid-song1", "zzvid-song2"]
    assert all(MIN_S <= e["duration"] <= MAX_S for e in band)


def test_discover_stores_yt_scheme_candidates(conn):
    a = _artist(conn)
    _identity(conn, a)
    n = discover_youtube(conn, str(a), "UCzzfixture0000000000000", fetcher=lambda cid: ENTRIES)
    assert n == 2
    rows = conn.execute(
        "SELECT audio_url, duration_s, binding_evidence FROM audio_track "
        "WHERE artist_id = %s ORDER BY (binding_evidence->>'track_index')::int", (a,),
    ).fetchall()
    assert all(r[0] and r[0].startswith("yt:") for r in rows)  # gate open: yt: scheme, governed extraction
    assert [r[1] for r in rows] == [212, 300]
    assert rows[0][2]["watch_url"].endswith("zzvid-song1")
    assert rows[0][2]["experimental"] is True
    # the wall holds: NULL audio_url can never reach the embed path
    assert pending_tracks(conn, str(a), "mock-model") == []
    assert pending_tracks(conn, str(a), "mock-model", source="youtube") == []


def test_discover_is_idempotent(conn):
    a = _artist(conn)
    _identity(conn, a)
    f = lambda cid: ENTRIES  # noqa: E731
    assert discover_youtube(conn, str(a), "UCzzfixture0000000000000", fetcher=f) == 2
    assert discover_youtube(conn, str(a), "UCzzfixture0000000000000", fetcher=f) == 0


def test_channel_without_videos_tab_is_empty_not_fatal(conn, monkeypatch):
    # Mass-scale finding: some channels have no /videos tab — yt-dlp raises;
    # discovery must yield 0 (→ terminal 'empty' verdict), not a retry storm.
    import yt_dlp

    import pipeline.sources.youtube as yt

    def boom(url, download=False):
        raise yt_dlp.utils.DownloadError("This channel does not have a videos tab")

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        extract_info = staticmethod(boom)

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    assert yt.fetch_channel_videos("UCzz-no-tab") == []
