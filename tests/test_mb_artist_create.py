"""MB artist-creation driver: the four-flow matrix (test/live x tags/artists)
and the safety model — live is approved-only, integrity flags block at the
door, test mbids never leak, link types come from the dump.
"""

from __future__ import annotations

import json

import pytest

from pipeline.mb_artist_create import create_artist, login, submit_artists

MBID_NEW = "12345678-1234-1234-1234-1234567890ab"


class FakeMB:
    """Records every request; serves CSRF forms and redirects like MB."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, data=None):
        self.calls.append((url, data))
        if url.endswith("/login") and data is None:
            return 200, {}, b'<input name="csrf_token" value="tok-login"/>'
        if url.endswith("/login"):
            return 302, {"Location": "/user/bot"}, b""
        if url.endswith("/artist/create") and data is None:
            return 200, {}, b'<input name="csrf_token" value="tok-create"/>'
        if url.endswith("/artist/create"):
            return 302, {"Location": f"/artist/{MBID_NEW}"}, b""
        return 404, {}, b""


@pytest.fixture
def link_types(conn):
    conn.execute(
        "INSERT INTO mb_raw.link_type (id, gid, entity_type0, entity_type1, name, "
        "description, link_phrase, reverse_link_phrase, long_link_phrase) VALUES "
        "(9901, gen_random_uuid(), 'artist','url','bandcamp','d','p','rp','lp'), "
        "(9902, gen_random_uuid(), 'artist','url','soundcloud','d','p','rp','lp') "
        "ON CONFLICT DO NOTHING")


def _staged(conn, name, status="spot_check"):
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES (%s, NULL) RETURNING id", (name,)
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO mb_submission (artist_id, payload, status) VALUES (%s, %s, %s)",
        (a, json.dumps({"name": name, "sort_name": name,
                        "urls": [{"platform": "bandcamp", "url": f"https://{name}.bandcamp.com"}]}),
         status))
    return a


def test_login_carries_csrf_and_credentials(conn):
    mb = FakeMB()
    login(mb, "test", "crates_bot", "hunter2")
    url, form = mb.calls[1]
    assert "test.musicbrainz.org" in url
    assert form["csrf_token"] == "tok-login"
    assert form["username"] == "crates_bot"


def test_create_artist_maps_fields_and_parses_mbid(conn, link_types):
    mb = FakeMB()
    mbid = create_artist(mb, conn, "test",
                         {"name": "Echo Unit", "sort_name": "Echo Unit",
                          "urls": [{"platform": "bandcamp", "url": "https://echounit.bandcamp.com"},
                                   {"platform": "tidal", "url": "https://tidal.com/artist/1"}]},
                         edit_note="rehearsal")
    assert mbid == MBID_NEW
    _url, form = mb.calls[1]
    assert form["edit-artist.name"] == "Echo Unit"
    assert form["edit-artist.url.0.text"] == "https://echounit.bandcamp.com"
    assert form["edit-artist.url.0.link_type_id"] == "9901"  # dump-resolved
    assert "edit-artist.url.1.text" not in form  # tidal: no mapped rel name


def test_link_type_ids_picks_artist_url_not_label_or_genre(conn, link_types):
    """MB reuses a link-type name across entity types; we must resolve the
    ARTIST<->url one, not an arbitrary label/genre-url with the same name."""
    from pipeline.mb_artist_create import link_type_ids

    conn.execute(
        "INSERT INTO mb_raw.link_type (id, gid, entity_type0, entity_type1, name, "
        "description, link_phrase, reverse_link_phrase, long_link_phrase) VALUES "
        "(9903, gen_random_uuid(), 'label','url','bandcamp','d','p','rp','lp'), "
        "(9904, gen_random_uuid(), 'genre','url','bandcamp','d','p','rp','lp') ON CONFLICT DO NOTHING")
    ids = link_type_ids(conn)
    assert ids["bandcamp"] == 9901  # artist-url, never the label(9903)/genre(9904)-url


def test_rehearsal_consumes_spot_check_but_live_requires_approved(conn, link_types):
    a = _staged(conn, "stagedband")
    mb = FakeMB()
    out = submit_artists(conn, target="live", fetch=mb, limit=10)
    assert out["submitted"] == 0  # spot_check is NOT enough for live
    out = submit_artists(conn, target="test", fetch=mb, limit=10)
    assert out["submitted"] == 1  # rehearsal uses staged payloads
    row = conn.execute(
        "SELECT created_mbid::text, target, status FROM mb_submission WHERE artist_id=%s", (a,)
    ).fetchone()
    assert row == (MBID_NEW, "test", "submitted")
    # the TEST-world mbid must never attach to our artist
    assert conn.execute("SELECT mbid FROM artist WHERE id=%s", (a,)).fetchone()[0] is None


def test_integrity_flags_block_at_the_door(conn, link_types):
    a = _staged(conn, "frozenband", status="approved")
    conn.execute(
        "INSERT INTO review_item (kind, subject_type, subject_id, reason, evidence, status) "
        "VALUES ('source_binding','artist',%s,'ai_slop','{}','pending')", (a,))
    out = submit_artists(conn, target="test", fetch=FakeMB(), limit=10)
    assert out["submitted"] == 0


def test_rejected_create_marks_failed_and_continues(conn, link_types):
    _staged(conn, "okband", status="approved")
    bad = _staged(conn, "badband", status="approved")

    class Rejecting(FakeMB):
        def __call__(self, url, data=None):
            if data and data.get("edit-artist.name") == "badband":
                return 200, {}, b"validation error page"  # no redirect = rejected
            return super().__call__(url, data)

    out = submit_artists(conn, target="test", fetch=Rejecting(), limit=10, pace_s=0)
    assert out["submitted"] == 1
    assert out["failed"] == 1
    assert conn.execute(
        "SELECT status FROM mb_submission WHERE artist_id=%s", (bad,)).fetchone()[0] == "failed"
