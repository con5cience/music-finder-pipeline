"""Cross-source coherence gate: the audio-native binding validator.

A wrong binding embeds a DIFFERENT musician's audio — per-source centroids
diverge where a correct binding's agree (measured bimodal: 0.7-0.95 same act,
<0.5 impostors, empty gap between). These tests pin: flagging below
threshold, idempotence, auto-heal on recovery, and the publish + MB-submit
exclusion gates.
"""

from __future__ import annotations

import json

from pipeline.coherence import artist_source_coherence, scan_coherence

MBID = "00000000-feed-4bad-9bad-00000000c0c0"


def _artist_with_sources(conn, name, mbid, vec_by_platform):
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES (%s, %s, 'deezer') RETURNING id",
        (name, mbid),
    ).fetchone()[0]
    for platform, vecs in vec_by_platform.items():
        t = conn.execute(
            "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
            "binding_tier, verification_status) VALUES (%s,%s,%s,'/x.wav',90,'A','verified') RETURNING id",
            (a, platform, f"zz-coh-{name}-{platform}"),
        ).fetchone()[0]
        for i, v in enumerate(vecs):
            conn.execute(
                "INSERT INTO clip_embedding (track_id, segment_start_s, segment_end_s, model, dim, embedding) "
                "VALUES (%s, %s, %s, 'mock-model', 2, %s)",
                (t, i * 30, i * 30 + 30, json.dumps(v)),
            )
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s, 'mock-model', 2, '[1,0]', 4, 1.0)", (a,),
    )
    return a


def test_coherent_sources_pass(conn):
    a = _artist_with_sources(conn, "Coherent", MBID[:-4] + "0001", {
        "deezer": [[1, 0], [0.99, 0.05]],
        "bandcamp": [[0.98, 0.1], [1, 0.02]],
    })
    c = artist_source_coherence(conn, a, model="mock-model")
    assert c["min_cosine"] > 0.9
    out = scan_coherence(conn, model="mock-model")
    assert out["flagged"] == 0


def test_impostor_source_flags_below_threshold(conn):
    a = _artist_with_sources(conn, "Impostor", MBID[:-4] + "0002", {
        "deezer": [[1, 0], [0.99, 0.05]],
        "bandcamp": [[0, 1], [0.05, 0.99]],  # orthogonal = different musician
    })
    out = scan_coherence(conn, model="mock-model")
    assert out["flagged"] == 1
    ev = conn.execute(
        "SELECT evidence FROM review_item WHERE subject_id = %s AND reason='source_coherence' "
        "AND status='pending'", (a,),
    ).fetchone()[0]
    assert ev["coherence"]["min_cosine"] < 0.6
    # idempotent: second scan files nothing new
    assert scan_coherence(conn, model="mock-model")["flagged"] == 0


def test_flag_autoheals_when_impostor_removed(conn):
    a = _artist_with_sources(conn, "Healer", MBID[:-4] + "0003", {
        "deezer": [[1, 0], [0.99, 0.05]],
        "bandcamp": [[0, 1], [0.05, 0.99]],
    })
    assert scan_coherence(conn, model="mock-model")["flagged"] == 1
    # remediation unbinds the impostor source's tracks…
    conn.execute(
        "DELETE FROM audio_track WHERE artist_id = %s AND platform = 'bandcamp'", (a,),
    )
    # …but a single remaining source is unjudgeable → flag stays (no false heal)
    assert scan_coherence(conn, model="mock-model")["healed"] == 0
    # a coherent second source appearing DOES heal
    t = conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
        "binding_tier, verification_status) VALUES (%s,'soundcloud','zz-coh-heal','/x.wav',90,'A','verified') "
        "RETURNING id", (a,),
    ).fetchone()[0]
    for i, v in enumerate([[1, 0.01], [0.99, 0.03]]):
        conn.execute(
            "INSERT INTO clip_embedding (track_id, segment_start_s, segment_end_s, model, dim, embedding) "
            "VALUES (%s, %s, %s, 'mock-model', 2, %s)", (t, i * 30, i * 30 + 30, json.dumps(v)),
        )
    out = scan_coherence(conn, model="mock-model")
    assert out["healed"] == 1


def test_publish_excludes_flagged_artists(conn):
    from pipeline.publish import publishable_artists

    a = _artist_with_sources(conn, "Held Back", MBID[:-4] + "0004", {
        "deezer": [[1, 0], [0.99, 0.05]],
        "bandcamp": [[0, 1], [0.05, 0.99]],
    })
    names = lambda: [r[2] for r in publishable_artists(conn, 1000) if r[0] == a]  # noqa: E731
    assert names() == ["Held Back"]  # publishable before the flag
    scan_coherence(conn, model="mock-model")
    assert names() == []  # held while the flag is open
    conn.execute(
        "UPDATE review_item SET status='rejected', resolved_at=now() "
        "WHERE subject_id=%s AND reason='source_coherence'", (a,),
    )
    assert names() == ["Held Back"]  # resolution releases the hold


def test_mb_queue_excludes_flagged_artists(conn):
    from pipeline.mb_submit import queue_eligible

    a = _artist_with_sources(conn, "MB Held", None, {
        "deezer": [[1, 0], [0.99, 0.05]],
        "bandcamp": [[0, 1], [0.05, 0.99]],
    })
    conn.execute(
        "INSERT INTO bc_candidate (artist_id, platform_id, band_name, band_url, status) "
        "VALUES (%s, 'zz-mbheld', 'MB Held', 'https://zz-mbheld.bandcamp.com', 'admitted')", (a,),
    )
    scan_coherence(conn, model="mock-model")
    queue_eligible(conn, limit=100)
    assert conn.execute(
        "SELECT count(*) FROM mb_submission WHERE artist_id = %s", (a,),
    ).fetchone()[0] == 0  # never queued while acoustically suspect


def test_binding_audit_reports_methods_and_flags(conn):
    from pipeline.binding_audit import audit

    a = _artist_with_sources(conn, "Audit Fix", MBID[:-4] + "0005", {
        "deezer": [[1, 0], [0.99, 0.05]],
        "bandcamp": [[0, 1], [0.05, 0.99]],
    })
    scan_coherence(conn, model="mock-model")
    out = audit(conn)
    assert any(r[0] == "source_coherence" for r in out["open_flags"])
    assert out["methods"]  # provenance distribution always present
