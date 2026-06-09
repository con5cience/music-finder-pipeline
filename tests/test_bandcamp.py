"""Bandcamp discovery (parse fixtures are REAL captured pages, 2026-06-09).

Covers: discography parsing (document order = newest-first preserved),
data-tralbum extraction, streamable filtering, discovery row writing with
evidence, single-release-artist root fallback, idempotency, URL refresh.
All DB ids synthetic (shared factory DB)."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.sources.bandcamp import (
    discover_bandcamp,
    parse_discography,
    parse_tralbum,
    refresh_bandcamp,
)

FIX = Path(__file__).parent / "fixtures"
MUSIC = (FIX / "bandcamp_music.html").read_bytes()
ALBUM = (FIX / "bandcamp_album.html").read_bytes()
MBID = "00000000-feed-4bad-9bad-000000000bca"
SUB = "zz-test-bc-fixture"


def _artist(conn) -> str:
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('BC Fixture', %s) RETURNING id", (MBID,)
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, vanity_url, page_type) "
        "VALUES (%s, 'bandcamp', %s, %s, 'artist')",
        (a, SUB, f"https://{SUB}.bandcamp.com"),
    )
    return a


def _retitled_album(track_ids: list[int]) -> bytes:
    """The real album fixture with synthetic track ids (avoid real-id collisions)."""
    import html as ihtml
    import re

    src = ALBUM.decode()
    blob = ihtml.unescape(re.search(r'data-tralbum="([^"]+)"', src).group(1))
    d = json.loads(blob)
    for t, new_id in zip(d["trackinfo"], track_ids, strict=False):
        t["track_id"] = t["id"] = new_id
    new_blob = ihtml.escape(json.dumps(d), quote=True)
    # lambda replacement: the JSON blob contains backslashes that re.sub would
    # otherwise treat as group references
    return re.sub(r'data-tralbum="[^"]+"', lambda _m: f'data-tralbum="{new_blob}"', src).encode()


def _serve(routes: dict[str, bytes]):
    def fetch(url: str) -> tuple[int, str, bytes]:
        for frag, body in routes.items():
            if frag in url:
                return 200, "text/html", body
        return 404, "text/html", b"<html>not found</html>"

    return fetch


def test_parse_discography_preserves_document_order():
    paths = parse_discography(MUSIC)
    assert len(paths) == 15
    assert all(p.startswith(("/album/", "/track/")) for p in paths)
    # document order, not sorted: fixture order is the page's order
    raw = MUSIC.decode()
    assert paths == sorted(paths, key=raw.index)


def test_parse_tralbum_real_shape():
    t = parse_tralbum(ALBUM)
    assert t is not None
    assert t["artist"] == "Burial"
    assert len(t["tracks"]) == 5
    tr = t["tracks"][0]
    assert tr.stream_url.startswith("https://t4.bcbits.com/stream/")
    assert tr.duration_s > 60
    assert tr.title


def test_parse_tralbum_absent_returns_none():
    assert parse_tralbum(b"<html><body>no player here</body></html>") is None


def test_discover_writes_rows_with_evidence(conn, tmp_path):
    a = _artist(conn)
    album = _retitled_album([990000003000 + i for i in range(5)])
    fetch = _serve({f"{SUB}.bandcamp.com/music": MUSIC, "/album/": album, "/track/": album})
    n = discover_bandcamp(conn, a, SUB, fetcher=fetch, cache_dir=tmp_path)
    # 5 releases walked x 5 tracks each, deduped by track id → 5 unique tracks
    assert n == 5
    rows = conn.execute(
        "SELECT platform, duration_s, binding_evidence->>'source', binding_evidence->>'album_path' "
        "FROM audio_track WHERE artist_id = %s",
        (a,),
    ).fetchall()
    assert len(rows) == 5
    for platform, dur, src, album_path in rows:
        assert platform == "bandcamp"
        assert dur > 60  # REAL full-track duration, not 30
        assert src == "bandcamp_tralbum"
        assert album_path.startswith(("/album/", "/track/"))


def test_discover_is_idempotent(conn, tmp_path):
    a = _artist(conn)
    album = _retitled_album([990000003100 + i for i in range(5)])
    fetch = _serve({f"{SUB}.bandcamp.com/music": MUSIC, "/album/": album, "/track/": album})
    assert discover_bandcamp(conn, a, SUB, fetcher=fetch, cache_dir=tmp_path) == 5
    assert discover_bandcamp(conn, a, SUB, fetcher=fetch, cache_dir=tmp_path) == 0


def test_discover_single_release_artist_root_fallback(conn, tmp_path):
    # No /music grid: artist's root page IS the album page.
    a = _artist(conn)
    album = _retitled_album([990000003200 + i for i in range(5)])
    fetch = _serve({"/music": b"<html><body>nothing here</body></html>", f"{SUB}.bandcamp.com": album})
    n = discover_bandcamp(conn, a, SUB, fetcher=fetch, cache_dir=tmp_path)
    assert n == 5


def test_refresh_bandcamp_updates_stream_url(conn, tmp_path):
    a = _artist(conn)
    tid = 990000003300
    album = _retitled_album([tid + i for i in range(5)])
    fetch = _serve({f"{SUB}.bandcamp.com/music": MUSIC, "/album/": album, "/track/": album})
    discover_bandcamp(conn, a, SUB, fetcher=fetch, cache_dir=tmp_path)
    conn.execute(
        "UPDATE audio_track SET audio_url = 'https://t4.bcbits.com/stream/EXPIRED' "
        "WHERE platform = 'bandcamp' AND platform_track_id = %s",
        (str(tid),),
    )
    fresh = refresh_bandcamp(conn, str(tid), fetcher=fetch, cache_dir=tmp_path)
    assert fresh is not None and "EXPIRED" not in fresh
    url = conn.execute(
        "SELECT audio_url FROM audio_track WHERE platform = 'bandcamp' AND platform_track_id = %s",
        (str(tid),),
    ).fetchone()[0]
    assert url == fresh
