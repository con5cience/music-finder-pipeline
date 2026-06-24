"""Deezer→Bandcamp recovery: the AUDIO gate. Name only nominates; audio binds.
Encodes the contamination lessons — name-alone never binds, homonyms are
rejected by audio, typos never auto-bind, ambiguity → review."""

from __future__ import annotations

from pipeline.deezer_bandcamp import recover_artist_bandcamp


def _artist(conn, name="Recover Fixture") -> str:
    # mbid NULL → name keys = just the display name (no mb_raw alias lookup)
    return conn.execute(
        "INSERT INTO artist (display_name) VALUES (%s) RETURNING id", (name,)
    ).fetchone()[0]


def _searcher(cands):
    return lambda conn, name: cands


def _embedder(mapping):  # subdomain -> embedding vector (None = unfetchable)
    return lambda conn, sub: mapping.get(sub)


ANCHOR = [1.0, 0.0, 0.0, 0.0]
SAME = [1.0, 0.0, 0.0, 0.0]       # cosine 1.0 vs anchor
DIFFERENT = [0.0, 1.0, 0.0, 0.0]  # cosine 0.0 vs anchor (homonym)


def test_review_only_scores_and_never_binds(conn):
    a = _artist(conn)
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture", anchor=ANCHOR,
        searcher=_searcher([{"name": "Recover Fixture", "platform_id": "zz-good", "popularity": 0}]),
        embedder=_embedder({"zz-good": SAME}),
    )  # auto_bind_threshold defaults None → review-only
    assert v == "review"
    assert conn.execute(
        "SELECT count(*) FROM platform_identity WHERE artist_id = %s AND platform = 'bandcamp'", (a,)
    ).fetchone()[0] == 0  # review-only NEVER binds, even at cosine 1.0
    ev = conn.execute("SELECT evidence FROM review_item WHERE subject_id = %s", (a,)).fetchone()[0]
    assert ev["candidates"][0]["subdomain"] == "zz-good"
    assert ev["candidates"][0]["audio_score"] > 0.99


def test_no_name_match_is_none(conn):
    a = _artist(conn)
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture", anchor=ANCHOR,
        searcher=_searcher([{"name": "Totally Different Band", "platform_id": "zz-x", "popularity": 0}]),
        embedder=_embedder({}),
    )
    assert v == "none"
    assert conn.execute("SELECT count(*) FROM review_item WHERE subject_id = %s", (a,)).fetchone()[0] == 0


def test_auto_bind_high_audio_exact_binds(conn):
    a = _artist(conn)
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture", anchor=ANCHOR, auto_bind_threshold=0.8,
        searcher=_searcher([{"name": "Recover Fixture", "platform_id": "zz-good", "popularity": 0}]),
        embedder=_embedder({"zz-good": SAME}),
    )
    assert v == "bound"
    tier, ev = conn.execute(
        "SELECT binding_tier, binding_evidence FROM platform_identity "
        "WHERE artist_id = %s AND platform = 'bandcamp'", (a,)
    ).fetchone()
    assert tier == "B"
    assert ev["method"] == "deezer_bandcamp_audio" and ev["audio_score"] > 0.99


def test_auto_bind_rejects_homonym_low_audio(conn):
    a = _artist(conn)
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture", anchor=ANCHOR, auto_bind_threshold=0.8,
        searcher=_searcher([{"name": "Recover Fixture", "platform_id": "zz-homonym", "popularity": 0}]),
        embedder=_embedder({"zz-homonym": DIFFERENT}),  # exact NAME, wrong AUDIO
    )
    assert v == "review"  # audio gate kept the homonym out
    assert conn.execute(
        "SELECT count(*) FROM platform_identity WHERE artist_id = %s AND platform = 'bandcamp'", (a,)
    ).fetchone()[0] == 0


def test_auto_bind_never_binds_typo_even_on_high_audio(conn):
    a = _artist(conn, name="Burial")
    v = recover_artist_bandcamp(
        conn, str(a), "Burial", anchor=ANCHOR, auto_bind_threshold=0.8,
        searcher=_searcher([{"name": "Buria", "platform_id": "zz-typo", "popularity": 0}]),  # edit-distance 1
        embedder=_embedder({"zz-typo": SAME}),
    )
    assert v == "review"  # typo-tier never auto-binds (the 85%-wrong class)
    assert conn.execute(
        "SELECT count(*) FROM platform_identity WHERE artist_id = %s AND platform = 'bandcamp'", (a,)
    ).fetchone()[0] == 0


def test_auto_bind_multiple_exact_goes_to_review(conn):
    a = _artist(conn)
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture", anchor=ANCHOR, auto_bind_threshold=0.8,
        searcher=_searcher([
            {"name": "Recover Fixture", "platform_id": "zz-1", "popularity": 0},
            {"name": "recover fixture", "platform_id": "zz-2", "popularity": 0},
        ]),
        embedder=_embedder({"zz-1": SAME, "zz-2": SAME}),  # both clear the bar → ambiguous
    )
    assert v == "review"
    assert conn.execute(
        "SELECT count(*) FROM platform_identity WHERE artist_id = %s AND platform = 'bandcamp'", (a,)
    ).fetchone()[0] == 0


def test_no_anchor_skips_without_binding(conn):
    a = _artist(conn)  # no artist_embedding row, no injected anchor
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture",
        searcher=_searcher([{"name": "Recover Fixture", "platform_id": "zz-good", "popularity": 0}]),
        embedder=_embedder({"zz-good": SAME}),
    )
    assert v == "no_anchor"
    assert conn.execute("SELECT count(*) FROM search_attempt WHERE artist_id = %s", (a,)).fetchone()[0] == 0


def test_already_searched_is_skipped(conn):
    a = _artist(conn)
    conn.execute(
        "INSERT INTO search_attempt (artist_id, platform, query, verdict, candidates) "
        "VALUES (%s, 'bandcamp', 'x', 'none', 0)", (a,)
    )
    v = recover_artist_bandcamp(
        conn, str(a), "Recover Fixture", anchor=ANCHOR,
        searcher=_searcher([{"name": "Recover Fixture", "platform_id": "zz", "popularity": 0}]),
        embedder=_embedder({"zz": SAME}),
    )
    assert v == "skipped"
