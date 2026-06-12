"""Corroborator: acoustic verification of the single-source blind spot.

Artists embedded from ONE machine-guessed (B-tier) source have no second
source for the coherence gate to compare. But most have OTHER MB-declared
(A-tier) pages — probe those and compare against the centroid:
confirmed (>=0.8) promotes the binding B->C (MB-payload eligible);
refuted (<0.5) files a source_coherence flag (publish + MB hold);
unprobeable is recorded so re-runs skip. All idempotent.
"""

from __future__ import annotations

import json

import pytest

from pipeline import corroborate as cb
from pipeline.corroborate import corroborate_blind_spot

MBID = "00000000-feed-4bad-9bad-00000000f0f0"


def _blind_artist(conn, name, tail, *, a_platform="deezer"):
    """Embedded from a B-tier bandcamp source; one A-tier page elsewhere."""
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES (%s, %s, 'bandcamp') RETURNING id",
        (name, MBID[:-4] + tail),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type, binding_tier, binding_evidence) "
        "VALUES (%s, 'bandcamp', %s, 'artist', 'B', %s)",
        (a, f"zz-cb-{tail}", json.dumps({"method": "search_exact_unique"})),
    )
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type, binding_tier) "
        "VALUES (%s, %s, %s, 'artist', 'A')",
        (a, a_platform, f"zz-cb-a-{tail}"),
    )
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s, 'mock-model', 2, '[1,0]', 4, 1.0)", (a,),
    )
    t = conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
        "binding_tier, verification_status) VALUES (%s,'bandcamp',%s,'/x.wav',90,'B1','verified') RETURNING id",
        (a, f"zz-cb-t-{tail}"),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO clip_embedding (track_id, segment_start_s, segment_end_s, model, dim, embedding) "
        "VALUES (%s, 0, 30, 'mock-model', 2, '[1,0]')", (t,),
    )
    return a


@pytest.fixture
def cosines(monkeypatch):
    table = {}

    def fake_cosine(conn, centroid, platform, platform_id, embedder, workdir, fetch):
        v = table.get(platform_id, "ERROR")
        if v == "ERROR":
            return None, False
        if v == "EMPTY":
            return None, True
        return v, False

    monkeypatch.setattr(cb, "candidate_cosine", fake_cosine)
    return table


def test_confirmed_promotes_b_to_c(conn, cosines):
    a = _blind_artist(conn, "Cb Confirm", "0001")
    cosines["zz-cb-a-0001"] = 0.91
    out = corroborate_blind_spot(conn, embedder=object(), model="mock-model")
    assert out["confirmed"] == 1
    tier, ev = conn.execute(
        "SELECT binding_tier, binding_evidence FROM platform_identity "
        "WHERE artist_id=%s AND platform='bandcamp'", (a,)).fetchone()
    assert tier == "C"
    assert ev["corroboration"]["status"] == "confirmed"
    assert ev["corroboration"]["cosine"] == 0.91
    assert ev["method"] == "search_exact_unique"  # provenance history preserved


def test_refuted_files_coherence_flag_and_holds_publish(conn, cosines):
    from pipeline.publish import publishable_artists

    a = _blind_artist(conn, "Cb Refute", "0002")
    cosines["zz-cb-a-0002"] = 0.21
    out = corroborate_blind_spot(conn, embedder=object(), model="mock-model")
    assert out["refuted"] == 1
    tier = conn.execute(
        "SELECT binding_tier FROM platform_identity WHERE artist_id=%s AND platform='bandcamp'",
        (a,)).fetchone()[0]
    assert tier == "B"  # never promoted
    flag = conn.execute(
        "SELECT evidence FROM review_item WHERE subject_id=%s AND reason='source_coherence' "
        "AND status='pending'", (a,)).fetchone()
    assert flag is not None
    assert flag[0]["coherence"]["min_cosine"] == 0.21
    assert [r for r in publishable_artists(conn, 1000) if r[0] == a] == []  # held


def test_unprobeable_marked_and_skipped(conn, cosines):
    a = _blind_artist(conn, "Cb Silent", "0003")
    cosines["zz-cb-a-0003"] = "EMPTY"
    out = corroborate_blind_spot(conn, embedder=object(), model="mock-model")
    assert out["unprobeable"] == 1
    ev = conn.execute(
        "SELECT binding_evidence FROM platform_identity WHERE artist_id=%s AND platform='bandcamp'",
        (a,)).fetchone()[0]
    assert ev["corroboration"]["status"] == "unprobeable"
    # idempotent: marked identities never re-probe
    assert corroborate_blind_spot(conn, embedder=object(), model="mock-model")["processed"] == 0
