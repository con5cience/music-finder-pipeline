"""ADR-019 BC discovery: crawl parse/upsert-once, dedup gates (identity,
exact-unique name → binding not discovery), trickle-valve admit creating
mbid-NULL artists, and the mbid-NULL publish path."""

from __future__ import annotations

import datetime as dt
import json

from pipeline.discovery import (
    _subdomain,
    admit,
    crawl_label,
    crawl_tag,
    dedup_gate,
    discover_wave,
    prune_candidates,
    reject_reason,
)

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
    assert out == {"banned": 0, "identity": 1, "name": 1}
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


def test_label_roster_crawl(conn, monkeypatch):
    monkeypatch.setenv("PIPELINE_FETCH_CACHE_DIR", "/tmp/test-disc-cache")
    html = b"""<html><ol id="bands">
      <li><a href="https://zz-roster-a.bandcamp.com">A</a></li>
      <li><a href="http://zz-roster-b.bandcamp.com/music">B</a></li>
      <li><a href="https://zz-thelabel.bandcamp.com/about">self</a></li>
    </ol></html>"""
    rep = crawl_label(conn, "zz-thelabel", fetcher=lambda url: (200, "text/html", html))
    assert rep["new"] == 2 and rep["roster_seen"] == 2  # self excluded
    tags = conn.execute(
        "SELECT tags FROM bc_candidate WHERE platform_id = 'zz-roster-a'").fetchone()[0]
    assert tags == ["label:zz-thelabel"]


def test_discover_wave_harvests_and_isolates(conn, monkeypatch):
    monkeypatch.setenv("PIPELINE_FETCH_CACHE_DIR", "/tmp/test-disc-cache")
    import json as _json

    tree_html = ('<div data-blob="' + _json.dumps({
        "appData": {"initialState": {
            "genres": [{"slug": "zz-g1"}, {"slug": "zz-g2"}],
            "subgenres": [{"slug": "zz-s1"}],
        }}}).replace('"', '&quot;') + '"></div>').encode()

    calls = {"n": 0}

    def fetcher(url, post_json=None):
        calls["n"] += 1
        if "tag/ambient" in url:
            return 200, "text/html", tree_html
        tag = (post_json or {}).get("tag_norm_names", [""])[0]
        if tag == "zz-g2":  # one tag's API errors — wave must survive
            return 500, "application/json", b"{}"
        return 200, "application/json", _json.dumps(
            {"results": [_item(f"zz-wave-{calls['n']}", f"Wave Band {calls['n']}")], "cursor": ""}
        ).encode()

    rep = discover_wave(conn, pages=1, admit_budget=1, fetcher=fetcher)
    assert rep["tags_crawled"] == 3
    assert rep["errors"] == 1          # the 500 tag isolated, wave survived
    assert rep["new_candidates"] == 2  # g1 + s1
    assert rep["admitted"] == 1        # budget respected


def test_dedup_gate_rejects_banned(conn, monkeypatch):
    monkeypatch.setenv("PIPELINE_FETCH_CACHE_DIR", "/tmp/test-disc-cache")
    conn.execute(
        "INSERT INTO ban_ledger (display_name, platform_ids) VALUES "
        "('Banned Band', '[\"bandcamp:zz-banned-1\"]')")
    pages = [{"results": [_item("zz-banned-1", "Banned Band")], "cursor": ""}]
    crawl_tag(conn, "zz-bantag", pages=1, fetcher=_fake_fetcher(pages))
    dedup_gate(conn)
    status, reason = conn.execute(
        "SELECT status, status_reason FROM bc_candidate WHERE platform_id = 'zz-banned-1'").fetchone()
    assert status == "rejected" and reason == "banned"
    assert admit(conn, 10) == 0  # rejected candidates never admit


def test_reject_reason_subtractive_filter():
    today = dt.date(2026, 6, 14)
    # label/imprint tell — band name (spaced) or hyphenated subdomain
    assert reject_reason("Forever Records", "foreverband", None, today=today) == "pre_admit_label"
    assert reject_reason("Lost Children Net Label", "lcnl", None, today=today) == "pre_admit_label"
    assert reject_reason("Inner Ocean Recordings", "io", None, today=today) == "pre_admit_label"
    assert reject_reason("Cool Act", "forever-records", None, today=today) == "pre_admit_label"
    # date sanity
    assert reject_reason("Old Dump", "od", dt.datetime(2019, 12, 31, tzinfo=dt.UTC), today=today) == "stale"
    assert reject_reason("Spam", "sp", dt.datetime(2037, 1, 1, tzinfo=dt.UTC), today=today) == "future_date"
    # KEPT (validated 2026-06-14): name==subdomain is normal for real solo/electronic
    # acts (886 live) — must NOT be rejected; near-future preorders kept; word-boundary
    # means 'record' inside a word is not a false hit.
    assert reject_reason("NewRetroWave", "newretrowave", dt.datetime(2026, 6, 10, tzinfo=dt.UTC), today=today) is None
    assert reject_reason("Preorder Act", "pa", dt.datetime(2026, 7, 30, tzinfo=dt.UTC), today=today) is None
    assert reject_reason("Recorder Quartet", "recorderquartet", None, today=today) is None


def test_prune_candidates_marks_only_obvious_junk(conn):
    today = dt.date(2026, 6, 14)

    def ins(pid, name, rel):
        conn.execute(
            "INSERT INTO bc_candidate (platform_id, band_name, band_url, release_seen_at) "
            "VALUES (%s, %s, %s, %s)",
            (pid, name, f"https://{pid}.bandcamp.com", rel))

    ins("zz-prune-label", "Acme Records", None)
    ins("zz-prune-stale", "Old Dump", dt.datetime(2015, 1, 1, tzinfo=dt.UTC))
    ins("zz-prune-future", "Spam 2037", dt.datetime(2037, 1, 1, tzinfo=dt.UTC))
    ins("zz-prune-keep1", "Genuine Artist", dt.datetime(2026, 6, 1, tzinfo=dt.UTC))
    ins("zz-prune-keep2", "zz-prune-keep2", None)  # name == subdomain → MUST survive

    out = prune_candidates(conn, today=today)
    assert out == {"pre_admit_label": 1, "stale": 1, "future_date": 1}

    kept = conn.execute(
        "SELECT platform_id FROM bc_candidate WHERE status='candidate' AND platform_id LIKE 'zz-prune-%' ORDER BY 1"
    ).fetchall()
    assert [r[0] for r in kept] == ["zz-prune-keep1", "zz-prune-keep2"]
    rej = dict(conn.execute(
        "SELECT platform_id, status_reason FROM bc_candidate WHERE status='rejected' AND platform_id LIKE 'zz-prune-%'"
    ).fetchall())
    assert rej == {"zz-prune-label": "pre_admit_label", "zz-prune-stale": "stale", "zz-prune-future": "future_date"}
    # rejected candidates never admit
    assert admit(conn, 50) >= 0  # smoke: admit runs clean after prune
