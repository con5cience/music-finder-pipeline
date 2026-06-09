"""Baseline schema: tables exist, and the fanout guard behaves (the DB building block)."""

from __future__ import annotations

import psycopg
import pytest


def _artist(conn, name: str) -> str:
    return conn.execute("INSERT INTO artist (display_name) VALUES (%s) RETURNING id", (name,)).fetchone()[0]


def _identity(conn, artist_id, platform_id: str, page_type: str) -> str:
    return conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type) "
        "VALUES (%s,'soundcloud',%s,%s) RETURNING id",
        (artist_id, platform_id, page_type),
    ).fetchone()[0]


def _track(conn, artist_id, track_id: str, from_identity_id, tier: str = "A") -> None:
    conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, from_identity_id, binding_tier) "
        "VALUES (%s,'soundcloud',%s,%s,%s)",
        (artist_id, track_id, from_identity_id, tier),
    )


def test_core_tables_exist(conn):
    names = {
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        ).fetchall()
    }
    assert {"artist", "platform_identity", "audio_track", "review_item"} <= names


def test_platform_identity_is_unique_per_platform(conn):
    a = _artist(conn, "A")
    _identity(conn, a, "dup", "artist")
    with pytest.raises(psycopg.errors.UniqueViolation):
        _identity(conn, a, "dup", "artist")


def test_artist_page_must_name_its_artist(conn):
    # page_type='artist' with NULL artist_id violates the CHECK.
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO platform_identity (platform, platform_id, page_type) VALUES ('soundcloud','x','artist')"
        )


def test_fanout_guard_blocks_artist_page_crediting_another_artist(conn):
    a1, a2 = _artist(conn, "A1"), _artist(conn, "A2")
    page = _identity(conn, a1, "111", "artist")
    _track(conn, a1, "t-own", page)  # crediting the page's own artist is fine
    with pytest.raises(psycopg.errors.CheckViolation):
        _track(conn, a2, "t-other", page)  # crediting a different artist is rejected


def test_label_page_may_credit_various_artists(conn):
    a1, a2 = _artist(conn, "A1"), _artist(conn, "A2")
    label = conn.execute(
        "INSERT INTO platform_identity (platform, platform_id, page_type) "
        "VALUES ('bandcamp','lab','label') RETURNING id"
    ).fetchone()[0]
    _track(conn, a1, "lt1", label, "B1")
    _track(conn, a2, "lt2", label, "B1")  # both allowed — labels are exempt
