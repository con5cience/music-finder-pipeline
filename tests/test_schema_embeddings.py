"""Embedding schema (0002): model-stamped clip/artist vectors + constraints (ADR-016)."""

from __future__ import annotations

import psycopg
import pytest


def _artist(conn, name: str) -> str:
    return conn.execute("INSERT INTO artist (display_name) VALUES (%s) RETURNING id", (name,)).fetchone()[0]


def _track(conn, artist_id, track_id: str) -> str:
    return conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, binding_tier) "
        "VALUES (%s,'deezer',%s,'A') RETURNING id",
        (artist_id, track_id),
    ).fetchone()[0]


def _vec(dim: int) -> str:
    return "[" + ",".join(["0.1"] * dim) + "]"


def _clip_embedding(conn, track_id, model: str, dim: int, start: int = 0, end: int = 30) -> None:
    conn.execute(
        "INSERT INTO clip_embedding (track_id, segment_start_s, segment_end_s, model, dim, embedding) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        (track_id, start, end, model, dim, _vec(dim)),
    )


def test_embedding_tables_exist(conn):
    names = {
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        ).fetchall()
    }
    assert {"clip_embedding", "artist_embedding"} <= names


def test_pgvector_extension_installed(conn):
    assert conn.execute("SELECT count(*) FROM pg_extension WHERE extname='vector'").fetchone()[0] == 1


def test_clip_embedding_roundtrip_and_mixed_dims(conn):
    # One column stores different models' dims (untyped vector + dim stamp).
    a = _artist(conn, "A")
    t = _track(conn, a, "t1")
    _clip_embedding(conn, t, "muq-large-msd", 1024)
    _clip_embedding(conn, t, "laion-clap-music", 512)
    rows = conn.execute(
        "SELECT model, dim, vector_dims(embedding) FROM clip_embedding WHERE track_id=%s ORDER BY model", (t,)
    ).fetchall()
    assert rows == [("laion-clap-music", 512, 512), ("muq-large-msd", 1024, 1024)]


def test_same_clip_same_model_is_unique(conn):
    a = _artist(conn, "A")
    t = _track(conn, a, "t1")
    _clip_embedding(conn, t, "muq-large-msd", 1024)
    with pytest.raises(psycopg.errors.UniqueViolation):
        _clip_embedding(conn, t, "muq-large-msd", 1024)


def test_same_clip_different_model_coexists(conn):
    # The ADR-016 swap story: a re-embed under a new model is additive.
    a = _artist(conn, "A")
    t = _track(conn, a, "t1")
    _clip_embedding(conn, t, "muq-large-msd", 1024)
    _clip_embedding(conn, t, "musicfm-msd", 1024)  # no violation


def test_dim_stamp_must_match_vector(conn):
    a = _artist(conn, "A")
    t = _track(conn, a, "t1")
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO clip_embedding (track_id, segment_start_s, segment_end_s, model, dim, embedding) "
            "VALUES (%s,0,30,'muq-large-msd',1024,%s)",
            (t, _vec(512)),  # claims 1024, stores 512
        )


def test_segment_must_be_positive_range(conn):
    a = _artist(conn, "A")
    t = _track(conn, a, "t1")
    with pytest.raises(psycopg.errors.CheckViolation):
        _clip_embedding(conn, t, "muq-large-msd", 1024, start=30, end=30)


def test_clip_embeddings_die_with_track(conn):
    a = _artist(conn, "A")
    t = _track(conn, a, "t1")
    _clip_embedding(conn, t, "muq-large-msd", 1024)
    conn.execute("DELETE FROM audio_track WHERE id=%s", (t,))
    assert conn.execute("SELECT count(*) FROM clip_embedding WHERE track_id=%s", (t,)).fetchone()[0] == 0


def test_artist_embedding_one_row_per_model(conn):
    a = _artist(conn, "A")
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count) VALUES (%s,%s,%s,%s,%s)",
        (a, "muq-large-msd", 1024, _vec(1024), 9),
    )
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(
            "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count) VALUES (%s,%s,%s,%s,%s)",
            (a, "muq-large-msd", 1024, _vec(1024), 9),
        )


def test_ann_similarity_query_uses_inner_product(conn):
    # Normalized vectors → inner product ranks like cosine. Nearest of two
    # candidates to a [1,0,...] probe must be the matching direction.
    a = _artist(conn, "A")
    t = _track(conn, a, "t1")
    aligned = "[" + ",".join(["1"] + ["0"] * 1023) + "]"
    orthogonal = "[" + ",".join(["0", "1"] + ["0"] * 1022) + "]"
    conn.execute(
        "INSERT INTO clip_embedding (track_id, segment_start_s, segment_end_s, model, dim, embedding) "
        "VALUES (%s,0,30,'muq-large-msd',1024,%s), (%s,30,60,'muq-large-msd',1024,%s)",
        (t, aligned, t, orthogonal),
    )
    top = conn.execute(
        "SELECT segment_start_s FROM clip_embedding WHERE model='muq-large-msd' "
        "ORDER BY embedding::vector(1024) <#> %s::vector(1024) LIMIT 1",
        (aligned,),
    ).fetchone()[0]
    assert top == 0


def test_partial_hnsw_indexes_exist_for_default_model(conn):
    idx = {
        r[0]
        for r in conn.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename IN ('clip_embedding','artist_embedding')"
        ).fetchall()
    }
    assert "idx_clip_embedding_muq_ann" in idx
    assert "idx_artist_embedding_muq_ann" in idx
