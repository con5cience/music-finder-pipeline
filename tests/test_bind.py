"""Tier-A binding + classification against platform_identity (slice 3b).

DB-truth path only: MB-derived identities are pre-classified 'artist' pages and
bind at Tier A with mb_url_rel evidence. Search-based (B-tier) binding is a
later slice. Fixture rows are fully synthetic (shared factory DB — see
test_mb_bootstrap.py).
"""

from __future__ import annotations

from pipeline.bind import classify_identity, tier_a_binding

MBID = "00000000-feed-4bad-9bad-000000000099"


def _artist(conn, name: str = "Bind Fixture") -> str:
    return conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES (%s, %s) RETURNING id", (name, MBID)
    ).fetchone()[0]


def _identity(conn, artist_id, platform: str, platform_id: str, page_type: str = "artist") -> None:
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, vanity_url, page_type) "
        "VALUES (%s, %s, %s, %s, %s)",
        (artist_id, platform, platform_id, f"https://example.test/{platform_id}", page_type),
    )


def test_classify_known_identity(conn):
    a = _artist(conn)
    _identity(conn, a, "deezer", "zz-bind-001")
    assert classify_identity(conn, "deezer", "zz-bind-001") == "artist"


def test_classify_unknown_identity(conn):
    assert classify_identity(conn, "deezer", "zz-bind-does-not-exist") == "unknown"


def test_tier_a_binding_for_mb_identity(conn):
    a = _artist(conn)
    _identity(conn, a, "bandcamp", "zz-bind-002")
    b = tier_a_binding(conn, a, "bandcamp", "zz-bind-002")
    assert b is not None
    assert b["tier"] == "A"
    assert b["evidence"]["source"] == "mb_url_rel"
    assert b["evidence"]["vanity_url"].endswith("zz-bind-002")


def test_tier_a_binding_requires_matching_artist(conn):
    a1, a2 = _artist(conn, "Owner"), conn.execute(
        "INSERT INTO artist (display_name) VALUES ('Other') RETURNING id"
    ).fetchone()[0]
    _identity(conn, a1, "soundcloud", "zz-bind-003")
    # identity belongs to a1; binding it for a2 must refuse (no cross-artist binds)
    assert tier_a_binding(conn, a2, "soundcloud", "zz-bind-003") is None


def test_tier_a_binding_unknown_identity_is_none(conn):
    a = _artist(conn)
    assert tier_a_binding(conn, a, "deezer", "zz-bind-nope") is None
