"""Deezer→Bandcamp recovery: the audio CONFIDENCE SCORECARD. Name only nominates;
audio binds. Stacks multi-track agreement + margin/kNN so the genre-twin tail
(different artist, similar audio) is rejected, not bound."""

from __future__ import annotations

from pathlib import Path

from pipeline.deezer_bandcamp import _group_means, bandcamp_top_tracks, recover_artist_bandcamp

ALBUM = (Path(__file__).parent / "fixtures" / "bandcamp_album.html").read_bytes()


def _serve(body):
    return lambda url: (200, "text/html", body)


def _artist(conn, name="Recover Fixture") -> str:
    return conn.execute(
        "INSERT INTO artist (display_name) VALUES (%s) RETURNING id", (name,)
    ).fetchone()[0]


def _searcher(cands):
    return lambda conn, name: cands


def _embedder(mapping):  # subdomain -> list of per-track embeddings ([] = unfetchable)
    return lambda conn, sub: mapping.get(sub, [])


def _nearest(value):  # subdomain-agnostic nearest-OTHER-centroid cosine
    return lambda conn, vec, exclude: value


ANCHOR = [1.0, 0.0, 0.0, 0.0]
SAME = [1.0, 0.0, 0.0, 0.0]       # cosine 1.0 vs anchor
DIFFERENT = [0.0, 1.0, 0.0, 0.0]  # cosine 0.0 vs anchor


def _cand(name, sub):
    return {"name": name, "platform_id": sub, "popularity": 0}


def test_review_only_scores_and_never_binds(conn):
    a = _artist(conn)
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture", anchor=ANCHOR,
        searcher=_searcher([_cand("Recover Fixture", "zz-good")]),
        embedder=_embedder({"zz-good": [SAME, SAME, SAME]}),
        nearest_other_fn=_nearest(0.3),
    )  # auto_bind_threshold None → review-only
    assert v == "review"
    assert conn.execute(
        "SELECT count(*) FROM platform_identity WHERE artist_id=%s AND platform='bandcamp'", (a,)
    ).fetchone()[0] == 0
    ev = conn.execute("SELECT evidence FROM review_item WHERE subject_id=%s", (a,)).fetchone()[0]
    card = ev["candidates"][0]
    assert card["audio"]["n"] == 3 and card["audio"]["median"] > 0.99
    assert card["margin"] > 0.6  # 1.0 - 0.3


def test_no_name_match_is_none(conn):
    a = _artist(conn)
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture", anchor=ANCHOR,
        searcher=_searcher([_cand("Totally Different Band", "zz-x")]),
        embedder=_embedder({}), nearest_other_fn=_nearest(0.3),
    )
    assert v == "none"
    assert conn.execute("SELECT count(*) FROM review_item WHERE subject_id=%s", (a,)).fetchone()[0] == 0


def test_auto_bind_high_audio_high_margin_binds(conn):
    a = _artist(conn)
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture", anchor=ANCHOR, auto_bind_threshold=0.8,
        searcher=_searcher([_cand("Recover Fixture", "zz-good")]),
        embedder=_embedder({"zz-good": [SAME, SAME, SAME]}),
        nearest_other_fn=_nearest(0.30),  # margin 0.70 >= floor
    )
    assert v == "bound"
    tier, ev = conn.execute(
        "SELECT binding_tier, binding_evidence FROM platform_identity "
        "WHERE artist_id=%s AND platform='bandcamp'", (a,)
    ).fetchone()
    assert tier == "B"
    assert ev["method"] == "deezer_bandcamp_audio"
    assert ev["scorecard"]["audio"]["median"] > 0.99 and ev["scorecard"]["margin"] > 0.6


def test_auto_bind_rejects_homonym_low_audio(conn):
    a = _artist(conn)
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture", anchor=ANCHOR, auto_bind_threshold=0.8,
        searcher=_searcher([_cand("Recover Fixture", "zz-homonym")]),  # exact NAME
        embedder=_embedder({"zz-homonym": [DIFFERENT, DIFFERENT]}),     # wrong AUDIO
        nearest_other_fn=_nearest(0.0),
    )
    assert v == "review"
    assert conn.execute(
        "SELECT count(*) FROM platform_identity WHERE artist_id=%s AND platform='bandcamp'", (a,)
    ).fetchone()[0] == 0


def test_auto_bind_rejects_low_margin_genre_twin(conn):
    # high absolute audio, but it's ~as close to SOME OTHER artist → low margin →
    # the genre-generic match the single-cosine gate would have wrongly bound
    a = _artist(conn)
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture", anchor=ANCHOR, auto_bind_threshold=0.8,
        searcher=_searcher([_cand("Recover Fixture", "zz-twin")]),
        embedder=_embedder({"zz-twin": [SAME, SAME]}),  # median 1.0
        nearest_other_fn=_nearest(0.95),                # margin 0.05 < 0.10 floor
    )
    assert v == "review"
    assert conn.execute(
        "SELECT count(*) FROM platform_identity WHERE artist_id=%s AND platform='bandcamp'", (a,)
    ).fetchone()[0] == 0


def test_auto_bind_requires_multiple_tracks(conn):
    a = _artist(conn)
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture", anchor=ANCHOR, auto_bind_threshold=0.8, min_tracks=2,
        searcher=_searcher([_cand("Recover Fixture", "zz-onetrack")]),
        embedder=_embedder({"zz-onetrack": [SAME]}),  # only 1 track < min_tracks
        nearest_other_fn=_nearest(0.30),
    )
    assert v == "review"  # single-track coincidence never auto-binds
    assert conn.execute(
        "SELECT count(*) FROM platform_identity WHERE artist_id=%s AND platform='bandcamp'", (a,)
    ).fetchone()[0] == 0


def test_auto_bind_never_binds_typo(conn):
    a = _artist(conn, name="Burial")
    v = recover_artist_bandcamp(
        conn, str(a), "Burial", anchor=ANCHOR, auto_bind_threshold=0.8,
        searcher=_searcher([_cand("Buria", "zz-typo")]),  # edit-distance 1
        embedder=_embedder({"zz-typo": [SAME, SAME]}),
        nearest_other_fn=_nearest(0.30),
    )
    assert v == "review"
    assert conn.execute(
        "SELECT count(*) FROM platform_identity WHERE artist_id=%s AND platform='bandcamp'", (a,)
    ).fetchone()[0] == 0


def test_auto_bind_multiple_exact_goes_to_review(conn):
    a = _artist(conn)
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture", anchor=ANCHOR, auto_bind_threshold=0.8,
        searcher=_searcher([_cand("Recover Fixture", "zz-1"), _cand("recover fixture", "zz-2")]),
        embedder=_embedder({"zz-1": [SAME, SAME], "zz-2": [SAME, SAME]}),  # both eligible → ambiguous
        nearest_other_fn=_nearest(0.30),
    )
    assert v == "review"
    assert conn.execute(
        "SELECT count(*) FROM platform_identity WHERE artist_id=%s AND platform='bandcamp'", (a,)
    ).fetchone()[0] == 0


def test_no_anchor_skips(conn):
    a = _artist(conn)
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture",
        searcher=_searcher([_cand("Recover Fixture", "zz-good")]),
        embedder=_embedder({"zz-good": [SAME, SAME]}), nearest_other_fn=_nearest(0.3),
    )
    assert v == "no_anchor"
    assert conn.execute("SELECT count(*) FROM search_attempt WHERE artist_id=%s", (a,)).fetchone()[0] == 0


def test_bandcamp_top_tracks_parses_fixture(conn, tmp_path):
    tracks = bandcamp_top_tracks(conn, "zz-fixture", k=3, fetcher=_serve(ALBUM), cache_dir=tmp_path)
    assert 1 <= len(tracks) <= 3
    assert all(t.stream_url and t.duration_s for t in tracks)


def test_group_means_collapses_per_track():
    # 3 window-vectors: 2 for track A, 1 for track B → one mean vector each
    means = _group_means([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0]], [2, 1])
    assert means == [[1.0, 0.0], [0.0, 2.0]]


def test_already_searched_is_skipped(conn):
    a = _artist(conn)
    conn.execute(
        "INSERT INTO search_attempt (artist_id, platform, query, verdict, candidates) "
        "VALUES (%s,'bandcamp','x','none',0)", (a,)
    )
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture", anchor=ANCHOR,
        searcher=_searcher([_cand("Recover Fixture", "zz")]),
        embedder=_embedder({"zz": [SAME, SAME]}), nearest_other_fn=_nearest(0.3),
    )
    assert v == "skipped"
