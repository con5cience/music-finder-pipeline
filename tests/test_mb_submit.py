"""MB contribution lane: payload completeness, eligibility gating (embedded
only), consent URL shape, tag-submission XML build (HTTP mocked)."""

from __future__ import annotations

from pipeline.mb_submit import build_payload, consent_url, queue_eligible

MBID = "00000000-feed-4bad-9bad-000000000eb1"


def test_consent_url_shape(monkeypatch):
    monkeypatch.setenv("MB_CLIENT_ID", "cid-x")
    monkeypatch.setenv("MB_CLIENT_SECRET", "sec-x")
    u = consent_url()
    assert "oauth2/authorize" in u and "client_id=cid-x" in u and "scope=" in u


def test_payload_and_eligibility(conn):
    # discovered artist: admitted + embedded → eligible
    aid = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) "
        "VALUES ('Submit Me', NULL, 'bandcamp') RETURNING id").fetchone()[0]
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type) "
        "VALUES (%s, 'bandcamp', 'zz-sub-1', 'artist')", (aid,))
    conn.execute(
        "INSERT INTO bc_candidate (platform_id, band_name, band_url, location, status, artist_id) "
        "VALUES ('zz-sub-1', 'Submit Me', 'https://zz-sub-1.bandcamp.com', 'Kraków, Poland', 'admitted', %s)",
        (aid,))
    # admitted but NOT embedded → ineligible
    aid2 = conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Not Yet', NULL) RETURNING id").fetchone()[0]
    conn.execute(
        "INSERT INTO bc_candidate (platform_id, band_name, band_url, status, artist_id) "
        "VALUES ('zz-sub-2', 'Not Yet', 'https://zz-sub-2.bandcamp.com', 'admitted', %s)", (aid2,))

    p = build_payload(conn, aid)
    assert p["name"] == "Submit Me" and p["area_hint"] == "Kraków, Poland"
    assert any(u["platform"] == "bandcamp" for u in p["urls"])

    assert queue_eligible(conn, 10) == 1  # only the embedded one
    status, payload = conn.execute(
        "SELECT status, payload FROM mb_submission WHERE artist_id = %s", (aid,)).fetchone()
    assert status == "spot_check"
    assert payload["area_hint"] == "Kraków, Poland"
    assert queue_eligible(conn, 10) == 0  # idempotent


def test_payload_urls_exclude_unconfirmed_tier_b(conn):
    """MB submissions are our signature upstream — only MB-declared (A) or
    human-confirmed (C) bindings may ride in url-rels. Tier-B is a machine
    guess: the typo tier shipped 191 wrong artists before it was caught
    (2026-06-12), and exact-unique carries silent homonym risk. A B-tier
    URL in an MB edit would push OUR mistake into the commons."""
    from pipeline.mb_submit import build_payload

    a = conn.execute(
        "INSERT INTO artist (display_name) VALUES ('Provenance Fixture') RETURNING id"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type, binding_tier) VALUES "
        "(%s, 'bandcamp', 'zz-prov-own', 'artist', 'A'), "
        "(%s, 'deezer', '990200', 'artist', 'B'), "
        "(%s, 'soundcloud', 'zz-prov-conf', 'artist', 'C')",
        (a, a, a),
    )
    urls = {u["platform"] for u in build_payload(conn, a)["urls"]}
    assert urls == {"bandcamp", "soundcloud"}  # B-tier deezer guess excluded


def test_submit_tags_leads_with_bandcamp_tags(conn, monkeypatch):
    """Bandcamp's human tags are submitted to MB ahead of our audio tags
    (dedup case-insensitive, cap 5). HTTP + token mocked."""
    import time
    import urllib.request

    import pipeline.mb_submit as mb

    monkeypatch.setattr(mb, "access_token", lambda conn, target="live": "tok")
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    captured = {}

    class FakeResp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b""

    def fake_urlopen(req, timeout=60):
        captured["body"] = req.data.decode()
        return FakeResp()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    mbid = "00000000-feed-4bad-9bad-00000000af01"
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES ('SubBC', %s, 'bandcamp') RETURNING id",
        (mbid,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO artist_tag_scores (artist_id, tag, score, model) VALUES (%s,'zz-audiotag',0.9,'muq-mulan-large')",
        (a,),
    )
    conn.execute(
        "INSERT INTO bc_candidate (platform_id, band_name, band_url, status, artist_id, tags) "
        "VALUES ('zz-bc-1','SubBC','https://x.bandcamp.com','admitted',%s,%s)",
        (a, ["zz-bandcamptag"]),
    )
    assert mb.submit_tags(conn, limit=10, target="test") == 1
    body = captured["body"]
    assert "zz-bandcamptag" in body and "zz-audiotag" in body
    assert body.index("zz-bandcamptag") < body.index("zz-audiotag")  # bandcamp leads
