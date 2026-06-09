"""Deezer track discovery (3c): top-tracks first, albums fallback, ADR-015
source-correctness (only main-artist tracks), Tier-A provenance on audio_track.

The top-tracks fixture is a REAL captured API response (tests/fixtures/
deezer_top.json); fallback shapes were probe-verified 2026-06-09. All platform
ids are synthetic (shared factory DB).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.sources.deezer import discover_deezer, parse_tracks

FIXTURE = Path(__file__).parent / "fixtures" / "deezer_top.json"
DZ_ARTIST = "990000001000"  # synthetic deezer artist id
MBID = "00000000-feed-4bad-9bad-000000000333"


def _fixture_body(artist_id: int = 6281) -> bytes:
    """Real fixture, with track artist ids rewritten to the synthetic artist."""
    d = json.loads(FIXTURE.read_text())
    for t in d["data"]:
        t["artist"]["id"] = artist_id
    return json.dumps(d).encode()


def _artist_with_identity(conn) -> str:
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Deezer Fixture', %s) RETURNING id", (MBID,)
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type) "
        "VALUES (%s, 'deezer', %s, 'artist')",
        (a, DZ_ARTIST),
    )
    return a


def _serve(routes: dict[str, bytes]):
    def fetch(url: str) -> tuple[int, str, bytes]:
        for frag, body in routes.items():
            if frag in url:
                return 200, "application/json", body
        raise AssertionError(f"unexpected fetch: {url}")

    return fetch


def test_parse_keeps_only_main_artist_tracks():
    body = json.loads(_fixture_body(artist_id=42))
    body["data"][1]["artist"]["id"] = 777  # a feature/collab — not our artist
    tracks = parse_tracks(json.dumps(body).encode(), "42")
    assert len(tracks) == 2  # 3 fixture tracks minus the foreign one
    assert all(t.preview_url.startswith("https://") for t in tracks)
    assert all(t.track_duration_s > 30 for t in tracks)


def test_discover_writes_tier_a_tracks(conn, tmp_path):
    a = _artist_with_identity(conn)
    fetch = _serve({f"artist/{DZ_ARTIST}/top": _fixture_body(int(DZ_ARTIST))})
    n = discover_deezer(conn, a, DZ_ARTIST, fetcher=fetch, cache_dir=tmp_path)
    assert n == 3
    rows = conn.execute(
        "SELECT platform, binding_tier, verification_status, duration_s, audio_url, "
        "binding_evidence->>'source', (binding_evidence->>'track_duration_s')::int "
        "FROM audio_track WHERE artist_id = %s",
        (a,),
    ).fetchall()
    assert len(rows) == 3
    for platform, tier, status, dur, url, src, full_dur in rows:
        assert (platform, tier, status) == ("deezer", "A", "verified")
        assert dur == 30  # duration of the audio AT audio_url (the preview)
        assert url.startswith("https://")
        assert src == "deezer_top"
        assert full_dur > 30  # full track duration preserved as evidence


def test_discover_is_idempotent(conn, tmp_path):
    a = _artist_with_identity(conn)
    fetch = _serve({f"artist/{DZ_ARTIST}/top": _fixture_body(int(DZ_ARTIST))})
    assert discover_deezer(conn, a, DZ_ARTIST, fetcher=fetch, cache_dir=tmp_path) == 3
    assert discover_deezer(conn, a, DZ_ARTIST, fetcher=fetch, cache_dir=tmp_path) == 0
    assert conn.execute("SELECT count(*) FROM audio_track WHERE artist_id = %s", (a,)).fetchone()[0] == 3


def test_discover_albums_fallback_for_empty_top(conn, tmp_path):
    a = _artist_with_identity(conn)
    empty_top = json.dumps({"data": [], "total": 0}).encode()  # real bogus-artist shape
    albums = json.dumps({"data": [{"id": 990001, "title": "LP", "record_type": "album"}], "total": 1}).encode()
    album_tracks = json.dumps(
        {
            "data": [
                {"id": 990002, "title": "T1", "duration": 200,
                 "preview": "https://cdn.test/p1.mp3", "artist": {"id": int(DZ_ARTIST)}},
                {"id": 990003, "title": "T2 (feat)", "duration": 180,
                 "preview": "https://cdn.test/p2.mp3", "artist": {"id": 777}},
            ],
            "total": 2,
        }
    ).encode()
    fetch = _serve({
        f"artist/{DZ_ARTIST}/top": empty_top,
        f"artist/{DZ_ARTIST}/albums": albums,
        "album/990001/tracks": album_tracks,
    })
    n = discover_deezer(conn, a, DZ_ARTIST, fetcher=fetch, cache_dir=tmp_path)
    assert n == 1  # T2 excluded: source-correctness applies on the fallback too
    tid = conn.execute(
        "SELECT platform_track_id FROM audio_track WHERE artist_id = %s", (a,)
    ).fetchone()[0]
    assert tid == "990002"


def test_discover_requires_identity(conn, tmp_path):
    a = conn.execute("INSERT INTO artist (display_name) VALUES ('No Identity') RETURNING id").fetchone()[0]
    with pytest.raises(LookupError):
        discover_deezer(conn, a, DZ_ARTIST, fetcher=_serve({}), cache_dir=tmp_path)
