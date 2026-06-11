"""ADR-019 BC discovery: crawl parse/upsert-once, dedup gates (identity,
exact-unique name → binding not discovery), trickle-valve admit creating
mbid-NULL artists, and the mbid-NULL publish path."""

from __future__ import annotations

import json

from pipeline.discovery import _subdomain, admit, crawl_tag, dedup_gate

MBID = "00000000-feed-4bad-9bad-000000000dc1"


def _fake_fetcher(pages):
    calls = {"n": 0}

    def f(url):
        i = min(calls["n"], len(pages) - 1)
        calls["n"] += 1
        return 200, "application/json", json.dumps(pages[i]).encode()

    return f


def _item(pid, name, loc="Berlin, Germany"):
    return {"band_name": name, "band_url": f"https://{pid}.bandcamp.com?from=discover_page",
            "band_location": loc, "band_genre_id": 10, "release_date": "2026-06-10 00:00:00 UTC"}


def test_subdomain_extraction():
    assert _subdomain("https://kiandray.bandcamp.com?from=x") == "kiandray"
    assert _subdomain("https://bandcamp.com/whatever") is None
    assert _subdomain("https://evil.example.com") is None


def test_crawl_upserts_once_and_paginates(conn, monkeypatch):
    monkeypatch.setenv("PIPELINE_FETCH_CACHE_DIR", "/tmp/test-disc-cache")
    pages = [
        {"results": [_item("zz-disc-a", "Disc Band A"), _item("zz-disc-b", "Disc Band B")], "cursor": "c2"},
        {"results": [_item("zz-disc-a", "Disc Band A")], "cursor": ""},  # dup on page 2
    ]
    rep = crawl_tag(conn, "zz-testtag", pages=2, fetcher=_fake_fetcher(pages))
    assert rep["new"] == 2 and rep["seen"] == 3
    rows = conn.execute("SELECT platform_id, location, status FROM bc_candidate ORDER BY platform_id").fetchall()
    assert [r[0] for r in rows] == ["zz-disc-a", "zz-disc-b"]
    assert rows[0][1] == "Berlin, Germany" and rows[0][2] == "candidate"


def test_dedup_gates_and_admit(conn, monkeypatch):
    monkeypatch.setenv("PIPELINE_FETCH_CACHE_DIR", "/tmp/test-disc-cache")
    # existing artist with a bound bandcamp identity
    known = conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Known Band', %s) RETURNING id", (MBID,)
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type) "
        "VALUES (%s, 'bandcamp', 'zz-disc-known', 'artist')", (known,))
    # existing artist with NO bandcamp identity but an exact-unique name
    named = conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Unique Namer', "
        "'00000000-feed-4bad-9bad-000000000dc2') RETURNING id").fetchone()[0]
    pages = [{"results": [
        _item("zz-disc-known", "Known Band"),
        _item("zz-disc-namer", "Unique Namer"),
        _item("zz-disc-fresh", "Genuinely New Band"),
    ], "cursor": ""}]
    crawl_tag(conn, "zz-tag2", pages=1, fetcher=_fake_fetcher(pages))
    out = dedup_gate(conn)
    assert out == {"identity": 1, "name": 1}
    # name-unique candidate handed its identity to the EXISTING artist
    got = conn.execute(
        "SELECT artist_id FROM platform_identity WHERE platform='bandcamp' AND platform_id='zz-disc-namer'"
    ).fetchone()
    assert got and got[0] == named
    # the genuinely new band admits as an mbid-NULL artist with a pending identity
    assert admit(conn, 5) == 1
    aid, mbid = conn.execute(
        "SELECT a.id, a.mbid FROM artist a JOIN bc_candidate c ON c.artist_id = a.id "
        "WHERE c.platform_id = 'zz-disc-fresh'").fetchone()
    assert mbid is None
    st = conn.execute(
        "SELECT scan_status FROM platform_identity WHERE platform='bandcamp' AND platform_id='zz-disc-fresh'"
    ).fetchone()[0]
    assert st == 'pending'  # the wave seeder's food
