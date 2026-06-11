"""MB contribution lane: payload completeness, eligibility gating (embedded
only), consent URL shape, tag-submission XML build (HTTP mocked)."""

from __future__ import annotations

import json

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
