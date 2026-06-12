"""Title-overlap corroboration: MB recordings as ground truth for bindings
audio can't reach (tidal/qobuz-only and page-less artists).
"""

from __future__ import annotations

import json

import pytest

from pipeline.title_corroborate import (
    ensure_title_tables, normalize_title, title_corroborate)

MBID = "00000000-feed-4bad-9bad-0000000a1a1a"


def test_normalize_title_strips_feat_and_punct():
    assert normalize_title("Titans Return (feat. MC Zz)") == "titans return"
    assert normalize_title("Last–Light!!") == "last light"


@pytest.fixture
def mb_recordings(conn):
    ensure_title_tables(conn)

    def seed(mbid_tail: str, dump_artist_id: int, titles: list[str]):
        gid = MBID[:-6] + mbid_tail
        conn.execute(
            "INSERT INTO mb_raw.artist (id, gid, name, sort_name, comment, edits_pending, last_updated, ended) "
            "VALUES (%s, %s::uuid, 'T', 'T', '', 0, now()::text::timestamptz, false)",
            (dump_artist_id, gid))
        conn.execute(
            "INSERT INTO mb_raw.artist_credit_name (artist_credit, position, artist, name) "
            "VALUES (%s, 0, %s, 'T')", (dump_artist_id * 10, dump_artist_id))
        for i, t in enumerate(titles):
            conn.execute(
                "INSERT INTO mb_raw.recording (id, gid, name, artist_credit) VALUES (%s, gen_random_uuid(), %s, %s)",
                (dump_artist_id * 100 + i, t, dump_artist_id * 10))
        return gid

    return seed


def _unreachable_artist(conn, name, mbid, titles: list[str]):
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES (%s, %s, 'soundcloud') RETURNING id",
        (name, mbid)).fetchone()[0]
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type, binding_tier, binding_evidence) "
        "VALUES (%s, 'soundcloud', %s, 'artist', 'B', %s)",
        (a, f"zz-tc-{name}", json.dumps({"method": "search_exact_unique",
                                         "corroboration": {"status": "no_a_pages"}})))
    for i, t in enumerate(titles):
        conn.execute(
            "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
            "binding_tier, verification_status, binding_evidence) "
            "VALUES (%s,'soundcloud',%s,'/x.mp3',90,'B1','verified',%s)",
            (a, f"zz-tc-{name}-{i}", json.dumps({"title": t})))
    return a


def test_overlap_confirms_and_promotes(conn, mb_recordings):
    mbid = mb_recordings("000001", 9100001, ["Titans Return", "Last Light", "Deep Field", "Echoes"])
    a = _unreachable_artist(conn, "tc1", mbid, ["Titans Return", "Last Light", "Deep Field (feat. X)"])
    out = title_corroborate(conn)
    assert out["confirmed"] == 1
    tier, ev = conn.execute(
        "SELECT binding_tier, binding_evidence FROM platform_identity WHERE artist_id=%s", (a,)).fetchone()
    assert tier == "C"
    assert ev["corroboration"]["method"] == "title_overlap"
    assert ev["corroboration"]["matches"] == 3


def test_zero_overlap_with_depth_refutes(conn, mb_recordings):
    mbid = mb_recordings("000002", 9100002,
                         ["Alpha Song", "Beta Song", "Gamma Song", "Delta Song", "Epsilon Song"])
    a = _unreachable_artist(conn, "tc2", mbid,
                            ["Completely Other", "Different Tune", "Wrong Artist", "Not Them", "Nope Track"])
    out = title_corroborate(conn)
    assert out["refuted"] == 1
    assert conn.execute(
        "SELECT count(*) FROM review_item WHERE subject_id=%s AND reason='source_coherence' "
        "AND status='pending'", (a,)).fetchone()[0] == 1


def test_sparse_mb_side_is_gray_not_refuted(conn, mb_recordings):
    # MB knows only 2 recordings; absence of overlap is sparsity, not proof
    mbid = mb_recordings("000003", 9100003, ["Rare Cut", "Obscure Tune"])
    a = _unreachable_artist(conn, "tc3", mbid, ["Some Track", "Other Track", "Third", "Fourth", "Fifth"])
    out = title_corroborate(conn)
    assert out["gray"] == 1
    assert conn.execute(
        "SELECT binding_tier FROM platform_identity WHERE artist_id=%s", (a,)).fetchone()[0] == "B"


def test_stop_titles_never_confirm(conn, mb_recordings):
    mbid = mb_recordings("000004", 9100004, ["Intro", "Untitled", "Live", "Real Song Here"])
    a = _unreachable_artist(conn, "tc4", mbid, ["Intro", "Untitled", "Live", "Unrelated Banger"])
    out = title_corroborate(conn)
    assert out["confirmed"] == 0  # 3 raw matches, all stoplisted


def test_no_mb_recordings_marked(conn, mb_recordings):
    mbid = mb_recordings("000005", 9100005, [])
    _unreachable_artist(conn, "tc5", mbid, ["A Song"])
    out = title_corroborate(conn)
    assert out["no_mb_recordings"] == 1
    # idempotent: marker prevents reprocessing
    assert title_corroborate(conn)["processed"] == 0
