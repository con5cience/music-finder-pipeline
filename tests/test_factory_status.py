"""factory-status seed-queue breakdown (provisional vs MB-bound, by lane).

The board mirrors wave_seeder.select_seed_batch eligibility: an artist counts as
queued only if it is unembedded (embedding_source IS NULL) AND has a pending
audio identity. Fast lanes (deezer/bandcamp/soundcloud) take precedence over
youtube-only; within both, mbid-NULL (provisional/discovery) is the front-runner.
"""

from __future__ import annotations

from pipeline.factory_status import render, snapshot


def _artist(conn, *, mbid, embedded):
    return conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES ('Q', %s, %s) RETURNING id",
        (mbid, "deezer" if embedded else None),
    ).fetchone()[0]


def _identity(conn, artist_id, platform, scan_status):
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type, scan_status) "
        "VALUES (%s, %s, %s, 'artist', %s)",
        (artist_id, platform, f"q-{platform}-{artist_id}", scan_status),
    )


def test_seed_queue_buckets_mirror_the_seeder(conn):
    base = snapshot(conn)["queue"]

    a = _artist(conn, mbid=None, embedded=False)  # tier1 fast, provisional
    _identity(conn, a, "deezer", "pending")
    b = _artist(conn, mbid="11111111-1111-1111-1111-111111111111", embedded=False)  # tier1 fast, MB-bound
    _identity(conn, b, "bandcamp", "pending")
    c = _artist(conn, mbid=None, embedded=False)  # tier2 yt-only, provisional
    _identity(conn, c, "youtube", "pending")
    d = _artist(conn, mbid="22222222-2222-2222-2222-222222222222", embedded=False)  # tier2 yt-only, MB-bound
    _identity(conn, d, "youtube", "pending")
    e = _artist(conn, mbid="33333333-3333-3333-3333-333333333333", embedded=False)  # fast+yt → counts as fast
    _identity(conn, e, "deezer", "pending")
    _identity(conn, e, "youtube", "pending")
    f = _artist(conn, mbid=None, embedded=True)  # excluded: already embedded
    _identity(conn, f, "deezer", "pending")
    g = _artist(conn, mbid=None, embedded=False)  # excluded: scanned, not pending
    _identity(conn, g, "soundcloud", "scanned")

    q = snapshot(conn)["queue"]
    assert q["fast"]["provisional"] - base["fast"]["provisional"] == 1  # a
    assert q["fast"]["mb"] - base["fast"]["mb"] == 2  # b, e (e's yt does not double-count)
    assert q["yt"]["provisional"] - base["yt"]["provisional"] == 1  # c
    assert q["yt"]["mb"] - base["yt"]["mb"] == 1  # d
    # f (embedded) and g (scanned/no pending) add nothing to any bucket


def test_render_includes_seed_queue():
    s = {
        "rates": {"10m": {"embeds": 1, "scans": 2, "head_runs": 3, "searches": 4}},
        "recent": [],
        "scans": {},
        "queue": {"fast": {"provisional": 5, "mb": 365368}, "yt": {"provisional": 0, "mb": 37718}},
    }
    out = render(s)
    assert "seed queue" in out.lower()
    assert "provisional" in out.lower()
    assert "365368" in out and "37718" in out
