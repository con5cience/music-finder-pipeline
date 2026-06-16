"""ADR-021 Tier A: the embed pass stashes the MuLan per-window + artist-mean
vectors so corpus re-analysis is a math pass, never a re-fetch. These tests pin
the contract artist_tag_pass relies on: rows land, the stored mean equals the
vector scoring actually uses, the write is idempotent-replace, and a persist
failure can never abort the (far more valuable) embed."""

from __future__ import annotations

import numpy as np

from pipeline.heads import TagHead, artist_tag_pass, persist_analysis_vectors
from pipeline.tags import TAG_MODEL
from pipeline.windows import WINDOW_VERSION


class _FakeScorer:
    """Truthy stand-in for MulanTagScorer — artist_tag_pass only needs a scorer
    whose score_vectors returns the tag list; the vectors come in pre-embedded."""

    def score_vectors(self, vecs, top_k=20):
        return [("zz-genre", 0.5)]


def _artist(conn) -> str:
    return str(
        conn.execute("INSERT INTO artist (display_name) VALUES ('Vec Fixture') RETURNING id")
        .fetchone()[0]
    )


def _read_mean(conn, artist_id) -> np.ndarray:
    txt = conn.execute(
        "SELECT embedding::text FROM artist_analysis_vector "
        "WHERE artist_id = %s AND kind = 'mean'",
        (artist_id,),
    ).fetchone()[0]
    return np.array([float(x) for x in txt.strip("[]").split(",")], dtype=np.float32)


def test_artist_tag_pass_stashes_window_and_mean_vectors(conn):
    a = _artist(conn)
    # two tracks contributing 2 + 3 windows → 5 window rows + 1 mean row
    v1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    v2 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0]], dtype=np.float32)

    artist_tag_pass(conn, [TagHead(_FakeScorer())], a, [v1, v2])

    counts = dict(
        conn.execute(
            "SELECT kind, count(*) FROM artist_analysis_vector "
            "WHERE artist_id = %s AND model = %s GROUP BY kind",
            (a, TAG_MODEL),
        ).fetchall()
    )
    assert counts == {"window": 5, "mean": 1}

    dim, wv = conn.execute(
        "SELECT dim, window_version FROM artist_analysis_vector "
        "WHERE artist_id = %s AND kind = 'mean'",
        (a,),
    ).fetchone()
    assert dim == 4 and wv == WINDOW_VERSION

    # the stored mean is EXACTLY the normalized mean of all windows — the same
    # vector score_vectors computes, so a future re-score reproduces today's tags.
    stacked = np.concatenate([v1, v2])
    expected = stacked.mean(axis=0)
    expected /= np.linalg.norm(expected) + 1e-9
    assert np.allclose(_read_mean(conn, a), expected, atol=1e-4)


def test_persist_is_idempotent_replace(conn):
    a = _artist(conn)
    big = np.eye(4, dtype=np.float32)  # 4 windows
    artist_tag_pass(conn, [TagHead(_FakeScorer())], a, [big])
    assert _window_count(conn, a) == 4

    # re-run with FEWER windows must drop the stale high-idx rows, not accumulate
    small = np.array([[1, 0, 0, 0]], dtype=np.float32)
    artist_tag_pass(conn, [TagHead(_FakeScorer())], a, [small])
    assert _window_count(conn, a) == 1


def test_persist_guards_degenerate_input(conn):
    a = _artist(conn)
    persist_analysis_vectors(conn, a, np.empty((0, 4), dtype=np.float32))  # no windows
    persist_analysis_vectors(conn, a, np.array([1.0, 2.0], dtype=np.float32))  # 1-D, not a matrix
    assert _window_count(conn, a) == 0


def _window_count(conn, artist_id) -> int:
    return conn.execute(
        "SELECT count(*) FROM artist_analysis_vector WHERE artist_id = %s AND kind = 'window'",
        (artist_id,),
    ).fetchone()[0]


def test_refresh_centering_computes_d_from_stored_vectors(conn):
    """ADR-020 P5: refresh_centering writes per-tag d_i = (vocab text emb) . (mean
    audio direction) from the stored MuLan vectors. With one stored vector [1,0],
    mu-hat=[1,0], so an aligned vocab row scores d=1 and an orthogonal one d=0."""
    import numpy as np
    from pipeline.tags import load_centering, refresh_centering

    a = conn.execute("INSERT INTO artist (display_name) VALUES ('rc') RETURNING id").fetchone()[0]
    conn.execute(
        "INSERT INTO artist_analysis_vector (artist_id, model, kind, idx, dim, embedding, window_version) "
        "VALUES (%s,'muq-mulan-large','mean',0,2,'[1,0]','peak-v1')", (a,),
    )

    class FakeScorer:
        vocabulary = ["t-aligned", "t-ortho"]
        _vocab_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        def _ensure(self):
            pass

    assert refresh_centering(conn, FakeScorer(), sample=10) == 2
    cen = load_centering(conn)
    assert abs(cen["t-aligned"] - 1.0) < 1e-5
    assert abs(cen["t-ortho"] - 0.0) < 1e-5
