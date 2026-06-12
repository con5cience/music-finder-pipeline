"""Acoustic auto-adjudication: the decision matrix that empties the queue.

Pins: unique-confirm approves with decision.method=auto_coherence; all-below-
reject rejects; gray and partly-unprobeable items stay pending but gain
per-candidate acoustic annotations; already-adjudicated items are skipped;
the poller binds auto decisions with their method (not admin_review).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline import adjudicate
from pipeline.adjudicate import adjudicate_pending

MBID = "00000000-feed-4bad-9bad-00000000d0d0"


class VecEmbedder:
    """Returns the vector encoded in each clip path (probe-<vec> markers)."""

    def embed(self, clips):
        return [json.loads(open(c.path).read()) for c in clips]


def _artist(conn, name, tail):
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES (%s, %s, 'deezer') RETURNING id",
        (name, MBID[:-4] + tail),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s, 'mock-model', 2, '[1,0]', 4, 1.0)", (a,),
    )
    return a


def _item(conn, artist_id, candidates, platform="deezer"):
    return conn.execute(
        "INSERT INTO review_item (kind, subject_type, subject_id, reason, evidence, status) "
        "VALUES ('source_binding','artist',%s,'2 exact-name candidates',%s,'pending') RETURNING id",
        (artist_id, json.dumps({"platform": platform, "query": "x", "candidates": candidates})),
    ).fetchone()[0]


@pytest.fixture
def probes(monkeypatch, tmp_path):
    """Route each platform_id to a fixed clip vector via fake probe+fetch."""
    table: dict[str, list[float] | None] = {}

    def fake_probe(conn, platform, platform_id, **kw):
        v = table.get(platform_id, "ERROR")
        if v == "ERROR":
            return None  # probe error — unknown evidence
        if v == "EMPTY":
            return []  # page exists, zero streamable tracks
        return [f"vec:{platform_id}"]

    def fake_fetch(url, workdir):
        pid = url.split(":", 1)[1]
        p = tmp_path / f"{pid}.json"
        p.write_text(json.dumps(table[pid]))
        return str(p)

    monkeypatch.setattr(adjudicate, "probe_candidate_urls", fake_probe)
    monkeypatch.setattr(adjudicate, "_center_clip", lambda path, wd: path)
    table["_fetch"] = fake_fetch  # smuggle for callers
    return table


def test_unique_confirm_approves_with_auto_method(conn, probes):
    a = _artist(conn, "Adj Confirm", "0001")
    rid = _item(conn, a, [{"name": "Right", "platform_id": "p-right", "popularity": 1},
                          {"name": "Wrong", "platform_id": "p-wrong", "popularity": 9}])
    probes["p-right"] = [0.99, 0.01]   # ~cos 1 vs [1,0]
    probes["p-wrong"] = [0.0, 1.0]     # orthogonal
    out = adjudicate_pending(conn, embedder=VecEmbedder(), fetch=probes["_fetch"], model="mock-model")
    assert out["approved"] == 1
    status, ev = conn.execute(
        "SELECT status, evidence FROM review_item WHERE id = %s", (rid,)).fetchone()
    assert status == "approved"
    assert ev["decision"] == {"platform": "deezer", "platform_id": "p-right",
                              "method": "auto_coherence", "cosine": ev["decision"]["cosine"]}
    assert ev["decision"]["cosine"] >= 0.8


def test_all_rejected_closes_item(conn, probes):
    a = _artist(conn, "Adj Reject", "0002")
    rid = _item(conn, a, [{"name": "A", "platform_id": "p-a", "popularity": 1},
                          {"name": "B", "platform_id": "p-b", "popularity": 2}])
    probes["p-a"] = [0.0, 1.0]
    probes["p-b"] = [-1.0, 0.0]
    out = adjudicate_pending(conn, embedder=VecEmbedder(), fetch=probes["_fetch"], model="mock-model")
    assert out["rejected"] == 1
    assert conn.execute("SELECT status FROM review_item WHERE id=%s", (rid,)).fetchone()[0] == "rejected"


def test_gray_zone_stays_pending_with_annotations(conn, probes):
    a = _artist(conn, "Adj Gray", "0003")
    rid = _item(conn, a, [{"name": "Mid", "platform_id": "p-mid", "popularity": 1}])
    probes["p-mid"] = [0.7, 0.714]  # cos ~0.7: above reject, below confirm
    out = adjudicate_pending(conn, embedder=VecEmbedder(), fetch=probes["_fetch"], model="mock-model")
    assert out["annotated"] == 1
    status, ev = conn.execute(
        "SELECT status, evidence FROM review_item WHERE id=%s", (rid,)).fetchone()
    assert status == "pending"
    assert 0.5 < ev["candidates"][0]["acoustic"] < 0.8
    # second run skips it (adjudicated marker) — no repeat probing
    assert adjudicate_pending(conn, embedder=VecEmbedder(), fetch=probes["_fetch"], model="mock-model")["processed"] == 0


def test_unprobeable_candidate_blocks_auto_verdict(conn, probes):
    # one confirmable + one we could not hear: NEVER auto-approve on partial
    # evidence — the silent candidate could be the real one.
    a = _artist(conn, "Adj Partial", "0004")
    rid = _item(conn, a, [{"name": "Heard", "platform_id": "p-heard", "popularity": 1},
                          {"name": "Silent", "platform_id": "p-silent", "popularity": 2}])
    probes["p-heard"] = [0.99, 0.01]
    probes["p-silent"] = "ERROR"  # probe ERROR — unknown evidence blocks
    out = adjudicate_pending(conn, embedder=VecEmbedder(), fetch=probes["_fetch"], model="mock-model")
    assert out["annotated"] == 1
    assert conn.execute("SELECT status FROM review_item WHERE id=%s", (rid,)).fetchone()[0] == "pending"


def test_empty_account_rival_does_not_block_confirm(conn, probes):
    # a rival page with ZERO streamable tracks cannot be the audio source —
    # known-empty is accounted evidence, unlike a probe error
    a = _artist(conn, "Adj Empty", "0007")
    rid = _item(conn, a, [{"name": "Real", "platform_id": "p-real", "popularity": 1},
                          {"name": "Husk", "platform_id": "p-husk", "popularity": 2}])
    probes["p-real"] = [0.99, 0.01]
    probes["p-husk"] = "EMPTY"
    out = adjudicate_pending(conn, embedder=VecEmbedder(), fetch=probes["_fetch"], model="mock-model")
    assert out["approved"] == 1
    ev = conn.execute("SELECT evidence FROM review_item WHERE id=%s", (rid,)).fetchone()[0]
    assert ev["decision"]["platform_id"] == "p-real"
    assert ev["candidates"][1]["no_audio"] is True


def test_nan_cosine_is_unprobeable_not_poison(conn, probes):
    # silent/corrupt probe audio -> NaN vector -> NaN cosine; json.dumps
    # would emit literal NaN (invalid JSON to pg, the NaN-saga lesson).
    # A NaN candidate is UNPROBEABLE evidence, never a verdict or a write.
    a = _artist(conn, "Adj NaN", "0006")
    rid = _item(conn, a, [{"name": "Quiet", "platform_id": "p-nan", "popularity": 1}])
    probes["p-nan"] = [float("nan"), float("nan")]
    out = adjudicate_pending(conn, embedder=VecEmbedder(), fetch=probes["_fetch"], model="mock-model")
    assert out["unprobeable"] == 1
    ev = conn.execute("SELECT evidence FROM review_item WHERE id=%s", (rid,)).fetchone()[0]
    assert ev["candidates"][0]["acoustic"] is None  # stored as null, not NaN


def test_poller_carries_auto_method_into_binding(conn):
    from pipeline.review_poller import apply_approved_bindings

    a = _artist(conn, "Adj Poll", "0005")
    conn.execute(
        "INSERT INTO review_item (kind, subject_type, subject_id, reason, evidence, status) "
        "VALUES ('source_binding','artist',%s,'x',%s,'approved')",
        (a, json.dumps({"platform": "deezer", "candidates": [],
                        "decision": {"platform": "deezer", "platform_id": "990300",
                                     "method": "auto_coherence", "cosine": 0.91}})),
    )
    apply_approved_bindings(conn)
    tier, ev = conn.execute(
        "SELECT binding_tier, binding_evidence FROM platform_identity "
        "WHERE artist_id=%s AND platform='deezer'", (a,)).fetchone()
    assert tier == "C"
    assert ev["method"] == "auto_coherence"
    assert ev["cosine"] == 0.91
