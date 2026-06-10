"""SoundCloud discovery (official API, preview-grade source) — REAL captured
fixtures (tests/fixtures/soundcloud_*.json, corpus artist '0-5w'), fake
fetcher/resolver, no network. Encodes the empirical law: app-only tokens
stream 30s intro previews, so duration_s is ALWAYS 30 and the metadata
duration lives in evidence."""

from __future__ import annotations

from pathlib import Path

from pipeline.sources.soundcloud import discover_soundcloud, parse_tracks, refresh_soundcloud

FIX = Path(__file__).parent / "fixtures"
MBID = "00000000-feed-4bad-9bad-000000000777"


def _artist(conn) -> str:
    return conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('SC Fixture', %s) RETURNING id", (MBID,)
    ).fetchone()[0]


def _identity(conn, a, pid="zz-sc-user"):
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type) "
        "VALUES (%s, 'soundcloud', %s, 'artist')",
        (a, pid),
    )


def _fake_fetcher(url: str):
    if "/resolve" in url:
        return 200, "application/json", FIX.joinpath("soundcloud_resolve.json").read_bytes()
    if "/tracks?" in url:
        return 200, "application/json", FIX.joinpath("soundcloud_tracks.json").read_bytes()
    raise AssertionError(f"unexpected fetch: {url}")


def test_parse_tracks_streamable_only():
    tracks = parse_tracks(FIX.joinpath("soundcloud_tracks.json").read_bytes())
    assert len(tracks) >= 5  # the captured artist has 9 streamable uploads
    assert all(t["stream_url"] for t in tracks)


def test_discover_stores_previews_with_walk_order(conn):
    a = _artist(conn)
    _identity(conn, a)
    resolved = []

    def fake_resolver(track_api_id):
        resolved.append(track_api_id)
        return f"https://cdn.example/{track_api_id}.128.mp3?Policy=zz"

    n = discover_soundcloud(conn, a, "zz-sc-user", fetcher=_fake_fetcher, stream_resolver=fake_resolver)
    assert n == len(resolved) > 0
    rows = conn.execute(
        "SELECT duration_s, audio_url, binding_evidence FROM audio_track "
        "WHERE artist_id = %s ORDER BY (binding_evidence->>'track_index')::int",
        (a,),
    ).fetchall()
    for i, (dur, url, ev) in enumerate(rows):
        assert dur == 30                      # ACTUAL audio length, not metadata
        assert "cdn.example" in url           # the resolved signed URL is stored
        assert ev["track_index"] == i         # walk order recorded
        assert ev["preview_only"] is True
        assert ev["full_duration_s"] >= 0     # real duration kept for the record
    # at least one fixture track is a long upload — metadata duration preserved
    assert any(r[2]["full_duration_s"] >= 60 for r in rows)


def test_discover_is_idempotent(conn):
    a = _artist(conn)
    _identity(conn, a)
    resolver = lambda tid: f"https://cdn.example/{tid}.mp3"  # noqa: E731
    n1 = discover_soundcloud(conn, a, "zz-sc-user", fetcher=_fake_fetcher, stream_resolver=resolver)
    n2 = discover_soundcloud(conn, a, "zz-sc-user", fetcher=_fake_fetcher, stream_resolver=resolver)
    assert n1 > 0 and n2 == 0  # conflict-skipped, not duplicated


def test_unresolvable_stream_is_skipped_not_fatal(conn):
    a = _artist(conn)
    _identity(conn, a)
    n = discover_soundcloud(conn, a, "zz-sc-user", fetcher=_fake_fetcher, stream_resolver=lambda _t: None)
    assert n == 0  # nothing stored, no exception — account stays scannable


def test_refresh_updates_stored_url(conn, monkeypatch):
    a = _artist(conn)
    _identity(conn, a)
    conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
        "binding_tier, verification_status) VALUES (%s,'soundcloud','zz-sc-t1','https://old/x.mp3',30,'A','verified')",
        (a,),
    )
    import pipeline.sources.soundcloud as sc

    monkeypatch.setattr(sc, "resolve_stream_url", lambda tid: f"https://fresh/{tid}.mp3")
    fresh = refresh_soundcloud(conn, "zz-sc-t1")
    assert fresh == "https://fresh/zz-sc-t1.mp3"
    url = conn.execute(
        "SELECT audio_url FROM audio_track WHERE platform = 'soundcloud' AND platform_track_id = 'zz-sc-t1'"
    ).fetchone()[0]
    assert url == fresh
