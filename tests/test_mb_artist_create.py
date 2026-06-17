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


def test_edit_form_detected_without_csrf():
    """The live MB /artist/create form carries NO csrf_token field (verified
    2026-06-17) — the old preflight gated on csrf and would falsely abort an
    authenticated run. Detect the required name input instead. Anonymous (302)
    and unverified-email (401) responses never render it."""
    from pipeline.mb_artist_create import _edit_form_present

    real_form = ('<form action="/artist/create" method="post" class="edit-artist">'
                 '<input name="edit-artist.name" value=""/>'
                 '<input name="edit-artist.sort_name" value=""/></form>')
    assert _edit_form_present(real_form) is True          # no csrf needed
    assert "csrf" not in real_form                        # documents the reality
    assert _edit_form_present('<a href="/login">Log in</a>') is False  # anon page
    assert _edit_form_present('Unauthorized request - verify your email') is False  # 401


def test_rehearsal_consumes_spot_check_but_live_requires_approved(conn, link_types):
    a = _staged(conn, "stagedband")
    mb = FakeMB()
    out = submit_artists(conn, target="live", fetch=mb, limit=10, url_owner=lambda *a, **k: None)
    assert out["submitted"] == 0  # spot_check is NOT enough for live
    out = submit_artists(conn, target="test", fetch=mb, limit=10, url_owner=lambda *a, **k: None)
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
    out = submit_artists(conn, target="test", fetch=FakeMB(), limit=10,
                         url_owner=lambda *a, **k: None)
    assert out["submitted"] == 0


def test_rejected_create_marks_failed_and_continues(conn, link_types):
    _staged(conn, "okband", status="approved")
    bad = _staged(conn, "badband", status="approved")

    class Rejecting(FakeMB):
        def __call__(self, url, data=None):
            if data and data.get("edit-artist.name") == "badband":
                return 200, {}, b"validation error page"  # no redirect = rejected
            return super().__call__(url, data)

    out = submit_artists(conn, target="test", fetch=Rejecting(), limit=10, pace_s=0,
                         url_owner=lambda *a, **k: None)
    assert out["submitted"] == 1
    assert out["failed"] == 1
    assert conn.execute(
        "SELECT status FROM mb_submission WHERE artist_id=%s", (bad,)).fetchone()[0] == "failed"


def test_mb_artist_url_owner_detects_existing_and_404(monkeypatch):
    """The URL is the duplicate discriminator: a URL already on an MB artist
    returns that artist's MBID (link, don't re-create); a 404 means the URL is
    unknown to MB (None -> proceed to create)."""
    import time
    import urllib.error
    import urllib.request

    import pipeline.mb_artist_create as mac
    monkeypatch.setattr(time, "sleep", lambda _s: None)  # skip the rate-limit pause

    class Resp:
        def __init__(self, payload): self._p = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._p

    existing = "11111111-2222-3333-4444-555555555555"
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=30: Resp(
                            json.dumps({"relations": [{"artist": {"id": existing}}]}).encode()))
    assert mac.mb_artist_url_owner("test", "https://x.bandcamp.com") == existing

    def _404(req, timeout=30):
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", _404)
    assert mac.mb_artist_url_owner("test", "https://new.bandcamp.com") is None


def test_create_artist_disambiguation_retry_on_collision(conn, link_types):
    """A name collision re-renders (HTTP 200 + possible-duplicates) and demands a
    disambiguation; create_artist retries with edit-artist.comment (from area_hint)
    and creates anyway."""
    class Colliding(FakeMB):
        def __call__(self, url, data=None):
            if url.endswith("/artist/create") and data is not None:
                self.calls.append((url, data))                                 # record the POSTs
                if "edit-artist.comment" not in data:
                    return 200, {}, b'<div id="possible-duplicates"></div>'  # 1st POST: collision
                return 302, {"Location": f"/artist/{MBID_NEW}"}, b""          # 2nd POST: created
            return super().__call__(url, data)

    mb = Colliding()
    mbid = create_artist(mb, conn, "test",
                         {"name": "Heaven's Gate", "sort_name": "Heaven's Gate",
                          "area_hint": "Linz, Austria"},
                         edit_note="rehearsal")
    assert mbid == MBID_NEW
    _url, form = mb.calls[-1]                       # the retry carried the disambiguation
    assert form["edit-artist.comment"] == "Linz, Austria"


def test_submit_artists_skips_url_duplicates(conn, link_types):
    """A staged artist whose URL is already in MB is a TRUE duplicate: recorded
    'duplicate' with the existing MBID, and NO create is attempted."""
    a = _staged(conn, "dupeband", status="approved")
    existing = "99999999-8888-7777-6666-555555555555"

    class NoCreate(FakeMB):
        def __call__(self, url, data=None):
            if url.endswith("/artist/create") and data is not None:
                raise AssertionError("create must not run for a URL-duplicate")
            return super().__call__(url, data)

    out = submit_artists(conn, target="test", fetch=NoCreate(), limit=10, pace_s=0,
                         url_owner=lambda target, u: existing)
    assert out["duplicate"] == 1 and out["submitted"] == 0
    row = conn.execute(
        "SELECT status, created_mbid::text FROM mb_submission WHERE artist_id=%s", (a,)).fetchone()
    assert row == ("duplicate", existing)
