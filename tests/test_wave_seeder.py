"""Seeder batch selection: fast lanes first, yt-only as last resort.

The 2026-06-11 collapse: youtube's queue is server-capped at 0.1/s, so a
yt-only artist occupies a window slot for hours while a deezer artist needs
minutes. With one mixed query, every 1000-batch carried ~8% yt-only artists
that accumulated until the 2000-slot window was all yt-waiters and the
seeder gated shut — factory throughput pinned to the yt ceiling (~300/hr).
select_seed_batch seeds yt-only artists ONLY when no fast-lane work remains
(the same last-resort philosophy as youtube's floor=4 cascade rank).
"""

from __future__ import annotations

from pipeline.wave_seeder import MAX_BATCH, MAX_LOW_WATER, clamp_window, select_seed_batch


def _artist(conn, name: str, mbid_tail: str, platforms: list[str]) -> str:
    artist_id = str(conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES (%s, "
        f"'00000000-feed-4bad-9bad-{mbid_tail:>012}') RETURNING id", (name,)
    ).fetchone()[0])
    for p in platforms:
        conn.execute(
            "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type) "
            "VALUES (%s, %s, %s, 'artist')", (artist_id, p, f"zz-seed-{mbid_tail}-{p}"),
        )
    return artist_id


def test_fast_lane_fills_before_any_yt_only(conn):
    fast = _artist(conn, "Seed Fast", "0000000a0001", ["deezer"])
    yt = _artist(conn, "Seed YtOnly", "0000000a0002", ["youtube"])
    both = _artist(conn, "Seed Both", "0000000a0003", ["soundcloud", "youtube"])
    batch = select_seed_batch(conn, 2)
    assert fast in batch and both in batch
    assert yt not in batch  # yt-only never displaces fast-lane work


def test_yt_only_seeds_when_fast_lane_exhausted(conn):
    fast = _artist(conn, "Seed Fast2", "0000000b0001", ["bandcamp"])
    yt = _artist(conn, "Seed YtOnly2", "0000000b0002", ["youtube"])
    batch = select_seed_batch(conn, 10)
    assert fast in batch and yt in batch  # top-up after fast lane drained
    # fast-lane rows come first in the batch
    assert batch.index(fast) < batch.index(yt)


def test_yt_with_exhausted_fast_identities_joins_yt_lane(conn):
    # fast identity already terminally scanned -> only youtube is pending:
    # the artist must still be reachable, via the yt lane.
    a = _artist(conn, "Seed Spent", "0000000c0001", ["deezer", "youtube"])
    conn.execute(
        "UPDATE platform_identity SET scan_status = 'empty' "
        "WHERE artist_id = %s AND platform = 'deezer'", (a,),
    )
    batch = select_seed_batch(conn, 10)
    assert a in batch


def test_non_audio_platforms_never_seed(conn):
    tidal_only = _artist(conn, "Seed Tidal", "0000000d0001", ["tidal"])
    batch = select_seed_batch(conn, 10)
    assert tidal_only not in batch


def test_embedded_artists_drop_out(conn):
    done = _artist(conn, "Seed Done", "0000000e0001", ["deezer"])
    conn.execute("UPDATE artist SET embedding_source = 'deezer' WHERE id = %s", (done,))
    batch = select_seed_batch(conn, 10)
    assert done not in batch


def test_clamp_window_caps_oversized_args():
    # the 2026-06-15 footgun: low-water 2000 / batch 1000 melted dev Temporal
    assert clamp_window(2000, 1000) == (MAX_LOW_WATER, MAX_BATCH)


def test_clamp_window_passes_safe_args_through():
    assert clamp_window(200, 500) == (200, 500)
    assert clamp_window(MAX_LOW_WATER, MAX_BATCH) == (MAX_LOW_WATER, MAX_BATCH)


def test_clamp_window_caps_each_dimension_independently():
    assert clamp_window(2000, 100) == (MAX_LOW_WATER, 100)
    assert clamp_window(100, 2000) == (100, MAX_BATCH)


def test_provisional_discovery_artists_sort_first(conn):
    fast = _artist(conn, "Seed MbBound", "0000000f0001", ["deezer"])
    prov = str(conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Seed Provisional', NULL) RETURNING id"
    ).fetchone()[0])
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type) "
        "VALUES (%s, 'bandcamp', 'zz-seed-prov', 'artist')", (prov,),
    )
    batch = select_seed_batch(conn, 2)
    assert batch.index(prov) < batch.index(fast)
