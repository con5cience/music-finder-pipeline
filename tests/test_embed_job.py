"""embed_job: registry-driven embed-and-store with model stamping (slice 2).

Uses the DB `conn` fixture + the torch-free MockEmbedder; audio "fetching" is a
no-op for local paths, and MockEmbedder hashes clip ids rather than reading
audio, so these tests exercise the full store path without model weights.
"""

from __future__ import annotations

import math

from pipeline.bench.mock import MockEmbedder
from pipeline.embed_job import embed_artist_clips, pending_tracks


def _artist(conn, name: str) -> str:
    return conn.execute("INSERT INTO artist (display_name) VALUES (%s) RETURNING id", (name,)).fetchone()[0]


def _track(conn, artist_id, track_id: str, audio_url: str | None, status: str = "verified", dur: int = 30) -> str:
    return conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
        "binding_tier, verification_status) VALUES (%s,'deezer',%s,%s,%s,'A',%s) RETURNING id",
        (artist_id, track_id, audio_url, dur, status),
    ).fetchone()[0]


def _embedder() -> MockEmbedder:
    return MockEmbedder(dim=8, name="mock-model")


def test_pending_tracks_requires_audio_url_and_good_status(conn):
    a = _artist(conn, "A")
    t_ok = _track(conn, a, "ok", "/audio/ok.mp3")
    _track(conn, a, "no-url", None)
    _track(conn, a, "rejected", "/audio/r.mp3", status="rejected")
    _track(conn, a, "quarantined", "/audio/q.mp3", status="quarantined")
    ids = [r[0] for r in pending_tracks(conn, a, "mock-model")]
    assert ids == [t_ok]


def test_embed_stores_stamped_rows(conn):
    a = _artist(conn, "A")
    t1 = _track(conn, a, "t1", "/audio/t1.mp3")
    t2 = _track(conn, a, "t2", "/audio/t2.mp3", dur=25)
    n = embed_artist_clips(conn, _embedder(), a)
    assert n == 2
    rows = conn.execute(
        "SELECT track_id, segment_start_s, segment_end_s, model, dim, vector_dims(embedding) "
        "FROM clip_embedding ORDER BY segment_end_s DESC"
    ).fetchall()
    assert rows == [(t1, 0, 30, "mock-model", 8, 8), (t2, 0, 25, "mock-model", 8, 8)]


def test_embed_is_idempotent_per_model(conn):
    a = _artist(conn, "A")
    _track(conn, a, "t1", "/audio/t1.mp3")
    assert embed_artist_clips(conn, _embedder(), a) == 1
    assert embed_artist_clips(conn, _embedder(), a) == 0  # nothing pending second time
    assert conn.execute("SELECT count(*) FROM clip_embedding").fetchone()[0] == 1


def test_second_model_is_additive(conn):
    # The ADR-016 swap story: a different model re-embeds the same clips.
    a = _artist(conn, "A")
    _track(conn, a, "t1", "/audio/t1.mp3")
    embed_artist_clips(conn, _embedder(), a)
    embed_artist_clips(conn, MockEmbedder(dim=8, name="mock-model-v2"), a)
    models = {r[0] for r in conn.execute("SELECT model FROM clip_embedding").fetchall()}
    assert models == {"mock-model", "mock-model-v2"}


def test_centroid_upserted_normalized_and_counted(conn):
    a = _artist(conn, "A")
    _track(conn, a, "t1", "/audio/t1.mp3")
    _track(conn, a, "t2", "/audio/t2.mp3")
    embed_artist_clips(conn, _embedder(), a)
    model, dim, emb_text, clip_count = conn.execute(
        "SELECT model, dim, embedding::text, clip_count FROM artist_embedding WHERE artist_id=%s", (a,)
    ).fetchone()
    assert (model, dim, clip_count) == ("mock-model", 8, 2)
    vec = [float(x) for x in emb_text.strip("[]").split(",")]
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-5)

    # New clip → centroid refreshes (upsert, not insert-fail).
    _track(conn, a, "t3", "/audio/t3.mp3")
    embed_artist_clips(conn, _embedder(), a)
    assert conn.execute(
        "SELECT clip_count FROM artist_embedding WHERE artist_id=%s AND model='mock-model'", (a,)
    ).fetchone()[0] == 3


def test_no_pending_tracks_is_a_clean_noop(conn):
    a = _artist(conn, "A")
    assert embed_artist_clips(conn, _embedder(), a) == 0
    assert conn.execute("SELECT count(*) FROM artist_embedding WHERE artist_id=%s", (a,)).fetchone()[0] == 0
