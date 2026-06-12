"""Behavioral AI-slop detector: catalog forensics, flag-only into the
publish/MB freezer. The threat model is slop FARMS (machine-generated
catalogs at scale), not one careful AI single — farms leave statistical
fingerprints: near-identical track durations, templated/numbered titles,
huge catalogs with no MB identity. Bias toward gray: a false AI verdict
on a human underground artist is the worst failure mode for this mission.
"""

from __future__ import annotations

import json

from pipeline.slop_detect import score_artist, scan_slop

MBID = "00000000-feed-4bad-9bad-0000000b1b1b"


def _artist_with_catalog(conn, name, durations, titles, *, mbid=None):
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES (%s, %s, 'soundcloud') RETURNING id",
        (name, mbid)).fetchone()[0]
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type, binding_tier) "
        "VALUES (%s, 'soundcloud', %s, 'artist', 'A')", (a, f"zz-slop-{name}"))
    for i, (d, t) in enumerate(zip(durations, titles)):
        conn.execute(
            "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
            "binding_tier, verification_status, binding_evidence) "
            "VALUES (%s,'soundcloud',%s,'/x.mp3',%s,'A','verified',%s)",
            (a, f"zz-slop-{name}-{i}", d, json.dumps({"title": t})))
    return a


def test_uniform_farm_catalog_scores_high(conn):
    # 12 tracks all ~180s, numbered template titles, no mbid: the farm shape
    a = _artist_with_catalog(
        conn, "farm",
        [180, 181, 179, 180, 182, 180, 178, 181, 180, 179, 180, 181],
        [f"Chill Lofi Beats Vol. {i}" for i in range(1, 13)])
    s = score_artist(conn, a)
    assert s["score"] >= 0.6
    assert s["duration_cv"] < 0.05


def test_organic_catalog_scores_low(conn):
    a = _artist_with_catalog(
        conn, "human",
        [142, 367, 95, 204, 251, 178, 489, 132, 305, 88],
        ["Winter Letters", "The Yard", "Coda", "Sleeper Hold", "Margaret",
         "Vapor Trails", "Knife Hits", "Dust Settles Pt. 2", "Hollow", "June"],
        mbid=MBID[:-4] + "0001")
    s = score_artist(conn, a)
    assert s["score"] < 0.4


def test_scan_flags_and_holds_from_publish(conn):
    from pipeline.publish import publishable_artists

    a = _artist_with_catalog(
        conn, "farm2",
        [199, 201, 200, 200, 202, 198, 200, 201, 199, 200, 200, 201, 199, 200, 202],
        [f"Sleep Music Part {i}" for i in range(15)])
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s, 'mock-model', 2, '[1,0]', 4, 1.0)", (a,))
    out = scan_slop(conn)
    assert out["flagged"] >= 1
    assert conn.execute(
        "SELECT count(*) FROM review_item WHERE subject_id=%s AND reason='ai_slop' AND status='pending'",
        (a,)).fetchone()[0] == 1
    assert [r for r in publishable_artists(conn, 1000) if r[0] == a] == []  # frozen
    # resolution releases; second scan never re-flags a resolved item
    conn.execute(
        "UPDATE review_item SET status='rejected', resolved_at=now() WHERE subject_id=%s AND reason='ai_slop'",
        (a,))
    assert [r for r in publishable_artists(conn, 1000) if r[0] == a] != []
    assert scan_slop(conn)["flagged"] == 0


def test_small_catalogs_never_flagged(conn):
    # 4 uniform tracks is an EP, not evidence
    a = _artist_with_catalog(conn, "ep", [180, 180, 181, 180], ["A", "B", "C", "D"])
    s = score_artist(conn, a)
    assert s["score"] == 0.0


def test_mb_queue_excludes_slop_flagged(conn):
    from pipeline.mb_submit import queue_eligible

    a = _artist_with_catalog(
        conn, "farm3", [239, 241, 240, 240, 242, 238, 240, 241, 239, 240, 240, 241],
        [f"Ambient Sessions {i}" for i in range(12)])
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s, 'mock-model', 2, '[1,0]', 4, 1.0)", (a,))
    conn.execute(
        "INSERT INTO bc_candidate (artist_id, platform_id, band_name, band_url, status) "
        "VALUES (%s, 'zz-farm3', 'farm3', 'https://zz-farm3.bandcamp.com', 'admitted')", (a,))
    scan_slop(conn)
    queue_eligible(conn, limit=100)
    assert conn.execute(
        "SELECT count(*) FROM mb_submission WHERE artist_id=%s", (a,)).fetchone()[0] == 0


def test_preview_constant_duration_is_not_evidence(conn):
    """854-artist false-positive wave (2026-06-12): deezer stores the
    CONSTANT preview length (30s) as duration_s, so every deezer artist
    had cv=0.0 by construction — classical ensembles with movement
    numbering got frozen as farms. Real durations live in
    binding_evidence.track_duration_s; degenerate single-valued duration
    sets are VOID evidence either way."""
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES "
        "('I Solisti Fixture', '00000000-feed-4bad-9bad-0000000b2b01', 'deezer') RETURNING id"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type, binding_tier) "
        "VALUES (%s, 'deezer', 'zz-slop-prev', 'artist', 'A')", (a,))
    real = [412, 188, 305, 247, 533, 164, 421, 376, 290, 198, 350, 275]
    for i, d in enumerate(real):
        conn.execute(
            "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
            "binding_tier, verification_status, binding_evidence) "
            "VALUES (%s,'deezer',%s,'/x.mp3',30,'A','verified',%s)",
            (a, f"zz-slop-prev-{i}", json.dumps({
                "title": f"Sonata No. {i + 1}", "track_duration_s": d})))
    s = score_artist(conn, a)
    assert s["score"] < 0.6  # movements + varied REAL durations = human
    assert s["duration_cv"] is None or s["duration_cv"] > 0.05


def test_degenerate_single_duration_value_is_void(conn):
    # even without evidence durations, a single distinct stored value can
    # never count as uniformity — it's a data artifact, not a measurement
    a = _artist_with_catalog(conn, "constonly", [30] * 12,
                             [f"Mix {i}" for i in range(12)])
    s = score_artist(conn, a)
    assert s["duration_cv"] is None


def test_logistic_probe_separates_synthetic_clouds():
    import numpy as np
    from pipeline.ai_probe import logistic_probe

    rng = np.random.default_rng(7)
    a = rng.normal(0.5, 1.0, (600, 32))   # shifted cloud
    b = rng.normal(-0.5, 1.0, (600, 32))
    out = logistic_probe(a, b)
    assert out["auc"] > 0.95

    # identical distributions: probe must NOT hallucinate separation
    c = rng.normal(0, 1, (600, 32))
    d = rng.normal(0, 1, (600, 32))
    out2 = logistic_probe(c, d)
    assert 0.35 < out2["auc"] < 0.65


def test_publish_evaluates_unscored_artists_in_cycle(conn):
    """THE CONTINUOUS GATE: a farm embedded between audits must be caught
    by the publish cycle itself — evaluation happens lazily at the choke
    point, in the same pass that would otherwise expose the artist."""
    from pipeline.publish import publishable_artists
    from pipeline.slop_detect import gate_unevaluated

    a = _artist_with_catalog(
        conn, "lategate",
        [199, 201, 200, 200, 202, 198, 200, 201, 199, 200, 200, 201],
        [f"Rain Sounds Part {i}" for i in range(12)])
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s, 'mock-model', 2, '[1,0]', 4, 1.0)", (a,))
    rows = publishable_artists(conn, 1000)
    assert any(r[0] == a for r in rows)  # would have published unflagged
    out = gate_unevaluated(conn, [r[0] for r in rows])
    assert out["flagged"] >= 1
    assert [r for r in publishable_artists(conn, 1000) if r[0] == a] == []  # caught
    # clean artists get an evaluation row and publish untouched
    assert conn.execute(
        "SELECT count(*) FROM slop_evaluation WHERE artist_id = %s", (a,)).fetchone()[0] == 1


def test_catalog_growth_triggers_reevaluation(conn):
    from pipeline.slop_detect import gate_unevaluated

    a = _artist_with_catalog(conn, "growth", [142, 367, 95, 204, 251, 178, 489, 132],
                             ["A1", "B2", "C3", "D4", "E5", "F6", "G7", "H8"])
    gate_unevaluated(conn, [a])
    first = conn.execute(
        "SELECT n_tracks, evaluated_at FROM slop_evaluation WHERE artist_id=%s", (a,)).fetchone()
    # catalog doubles with farm-shaped additions
    for i in range(12):
        conn.execute(
            "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
            "binding_tier, verification_status, binding_evidence) "
            "VALUES (%s,'soundcloud',%s,'/x.mp3',%s,'A','verified',%s)",
            (a, f"zz-slop-growth-x{i}", 200 + (i % 3), json.dumps({"title": f"Sleep Aid {i}"})))
    out = gate_unevaluated(conn, [a])
    assert out["evaluated"] == 1  # re-scored because the catalog grew
    second = conn.execute(
        "SELECT n_tracks FROM slop_evaluation WHERE artist_id=%s", (a,)).fetchone()
    assert second[0] > first[0]


def test_publish_incremental_gates_and_survives_all_flagged_batch(conn):
    """END-TO-END choke-point wiring (the prior test bypassed publish):
    a farm in the publishable batch must be flagged + dropped INSIDE
    publish_incremental — including the degenerate batch where EVERY row
    gets flagged (the empty-list tail must not crash the keyset advance)."""
    from pipeline.publish import publish_incremental

    conn.execute("""
        CREATE TEMP TABLE IF NOT EXISTS artists (
            id uuid PRIMARY KEY, mbid text UNIQUE, name text NOT NULL, slug text,
            tags jsonb, audio_embedding text, signal_ratio real, embedding_source text,
            perceptual jsonb, language text, location text,
            audio_embedding_updated timestamptz, created_at timestamptz,
            deezer_url text, bandcamp_url text, soundcloud_url text,
            youtube_url text, tidal_url text) ON COMMIT DROP""")
    a = _artist_with_catalog(
        conn, "pubfarm",
        [199, 201, 200, 200, 202, 198, 200, 201, 199, 200, 200, 201],
        [f"White Noise Loop {i}" for i in range(12)])
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio, computed_at) "
        "VALUES (%s, 'mock-model', 2, '[1,0]', 4, 1.0, now())", (a,))
    conn.execute("UPDATE publish_watermark SET last_run = now() - interval '1 hour' WHERE id='default'")
    # the farm is the ONLY changed artist: the whole batch gets flagged —
    # this is the rows[-1] empty-tail shape
    n = publish_incremental(conn, conn, limit=5)
    assert conn.execute(
        "SELECT count(*) FROM review_item WHERE subject_id=%s AND reason='ai_slop' AND status='pending'",
        (a,)).fetchone()[0] == 1
    assert conn.execute(
        "SELECT count(*) FROM artists WHERE id = %s", (a,)).fetchone()[0] == 0  # never published
