"""Staged embed pipeline: prep round-trip (manifest + clips + CPU analysis
ledgered), embed-from-stage (clips/centroid/heads/cleanup, retry-safe),
legacy fallback on missing manifest, stale GC."""

from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from pipeline.bench.mock import MockEmbedder
from pipeline.staging import clean_stale_stage, embed_staged, prep_artist, stage_root

MBID = "00000000-feed-4bad-9bad-000000000aac"
SR = 8000


def _artist(conn) -> str:
    return conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Stage Fixture', %s) RETURNING id", (MBID,)
    ).fetchone()[0]


def _bc_track(conn, a, tid, path, dur, ri):
    conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
        "binding_tier, binding_evidence, verification_status) "
        "VALUES (%s,'bandcamp',%s,%s,%s,'A',%s,'verified')",
        (a, tid, path, dur, json.dumps({"release_index": ri, "track_index": 0})),
    )


def _wav(tmp_path, name, secs):
    p = tmp_path / f"{name}.wav"
    rng = np.random.default_rng(7)
    sf.write(p, (rng.standard_normal(SR * secs) * 0.1).astype(np.float32), SR)
    return str(p)


def test_prep_then_embed_staged_roundtrip(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_STAGE_DIR", str(tmp_path / "stage"))
    a = _artist(conn)
    _bc_track(conn, a, "zz-st-1", _wav(tmp_path, "s1", 90), 90, 0)
    _bc_track(conn, a, "zz-st-2", _wav(tmp_path, "s2", 90), 90, 1)

    staged = prep_artist(conn, str(a), "bandcamp", "mock-model")
    assert staged == 2
    adir = stage_root() / str(a)
    manifest = json.loads((adir / "manifest.json").read_text())
    assert len(manifest["tracks"]) == 2
    assert all((adir / f).exists() for t in manifest["tracks"] for _s, _e, f in t["segs"])
    # CPU analysis ran during prep and is ledgered
    n = conn.execute(
        "SELECT count(*) FROM track_head_runs r JOIN audio_track t ON t.id = r.track_id "
        "WHERE t.artist_id = %s AND r.head = 'cpu_analysis'", (a,)).fetchone()[0]
    assert n == 2

    emb = MockEmbedder(dim=8, name="mock-model")
    k = embed_staged(conn, emb, str(a), "bandcamp", 0.67)
    assert k >= 4  # 2 tracks x >=2 windows of 90s audio
    src, clip_count = conn.execute(
        "SELECT a.embedding_source, ae.clip_count FROM artist a "
        "JOIN artist_embedding ae ON ae.artist_id = a.id WHERE a.id = %s", (a,)).fetchone()
    assert src == "bandcamp" and clip_count == k
    assert not adir.exists()  # stage cleaned on success

    # retry-safety: a second embed call (manifest gone) falls back cleanly
    # to the legacy path which finds nothing pending → converge metadata
    assert embed_staged(conn, emb, str(a), "bandcamp", 0.67) == 0


def test_embed_staged_fallback_without_manifest(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_STAGE_DIR", str(tmp_path / "stage-empty"))
    a = _artist(conn)
    _bc_track(conn, a, "zz-st-f1", _wav(tmp_path, "f1", 90), 90, 0)
    emb = MockEmbedder(dim=8, name="mock-model")
    # no prep ever ran → legacy single-pass path embeds directly
    k = embed_staged(conn, emb, str(a), "bandcamp", 0.33)
    assert k >= 2


def test_clean_stale_stage(tmp_path, monkeypatch):
    import os
    import time as _time

    monkeypatch.setenv("PIPELINE_STAGE_DIR", str(tmp_path / "stage-gc"))
    old = stage_root() / "old-artist"
    new = stage_root() / "new-artist"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    stale = _time.time() - 48 * 3600
    os.utime(old, (stale, stale))
    assert clean_stale_stage(24.0) == 1
    assert not old.exists() and new.exists()
