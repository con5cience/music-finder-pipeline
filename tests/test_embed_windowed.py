"""Windowed embedding for full-track sources: selection (3 tracks, newest
across distinct releases, ≥60s preferred) + RMS-peak clips with real segments.
Audio fixtures are real wav files written to tmp (the windower reads audio)."""

from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from pipeline.bench.mock import MockEmbedder
from pipeline.embed_job import embed_artist_clips

MBID = "00000000-feed-4bad-9bad-000000000bcb"
SR = 8000


def _artist(conn) -> str:
    return conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Windowed Fixture', %s) RETURNING id", (MBID,)
    ).fetchone()[0]


def _wav(tmp_path, name: str, duration_s: int) -> str:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(duration_s * SR).astype(np.float32) * 0.02
    mid = duration_s // 2
    x[mid * SR:(mid + 20) * SR] *= 20  # a loud hook in the middle
    p = tmp_path / f"{name}.wav"
    sf.write(p, x, SR)
    return str(p)


def _bc_track(conn, a, tid: str, path: str, dur: int, album: str, ri: int, ti: int = 0) -> None:
    conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
        "binding_tier, binding_evidence, verification_status) "
        "VALUES (%s, 'bandcamp', %s, %s, %s, 'A', %s, 'verified')",
        (a, tid, path, dur,
         json.dumps({"source": "bandcamp_tralbum", "album_path": album,
                     "release_index": ri, "track_index": ti})),
    )


def test_windowed_embed_selects_and_segments(conn, tmp_path):
    a = _artist(conn)
    # 3 releases, newest-first by release_index; release A has 2 tracks; one short skit
    _bc_track(conn, a, "zz-w-a1", _wav(tmp_path, "a1", 120), 120, "/album/newest", 0, 0)
    _bc_track(conn, a, "zz-w-a2", _wav(tmp_path, "a2", 120), 120, "/album/newest", 0, 1)
    _bc_track(conn, a, "zz-w-skit", _wav(tmp_path, "skit", 20), 20, "/album/mid", 1, 0)  # <60s: skipped
    _bc_track(conn, a, "zz-w-b1", _wav(tmp_path, "b1", 120), 120, "/album/mid", 1, 1)
    _bc_track(conn, a, "zz-w-c1", _wav(tmp_path, "c1", 120), 120, "/album/oldest", 2, 0)
    n = embed_artist_clips(conn, MockEmbedder(dim=8, name="mock-model"), a, source="bandcamp", signal_ratio=1.0)

    rows = conn.execute(
        "SELECT t.platform_track_id, ce.segment_start_s, ce.segment_end_s FROM clip_embedding ce "
        "JOIN audio_track t ON t.id = ce.track_id WHERE t.artist_id = %s ORDER BY 1, 2",
        (a,),
    ).fetchall()
    embedded_tracks = {r[0] for r in rows}
    # one track per distinct release, newest-first; the second newest-album
    # track and the skit are NOT embedded
    assert embedded_tracks == {"zz-w-a1", "zz-w-b1", "zz-w-c1"}
    assert n == len(rows)
    # windows are real segments: 30s long, not all anchored at 0
    assert all(e - s == 30 for _t, s, e in rows)
    assert any(s > 0 for _t, s, _e in rows)
    # each track contributes multiple windows (120s track: 3-4 fit)
    per_track = {t: sum(1 for r in rows if r[0] == t) for t in embedded_tracks}
    assert all(2 <= c <= 4 for c in per_track.values())


def test_windowed_rerun_is_idempotent(conn, tmp_path):
    a = _artist(conn)
    _bc_track(conn, a, "zz-w-r1", _wav(tmp_path, "r1", 90), 90, "/album/x", 0)
    emb = MockEmbedder(dim=8, name="mock-model")
    n1 = embed_artist_clips(conn, emb, a, source="bandcamp", signal_ratio=0.33)
    assert n1 >= 2
    n2 = embed_artist_clips(conn, emb, a, source="bandcamp", signal_ratio=0.33)
    assert n2 == 0  # track has clips for this model → not pending


def test_preview_platform_still_single_clip(conn, tmp_path):
    # regression: deezer path is untouched by windowing
    a = _artist(conn)
    conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
        "binding_tier, verification_status) VALUES (%s, 'deezer', 'zz-w-d1', '/audio/d1.mp3', 30, 'A', 'verified')",
        (a,),
    )
    n = embed_artist_clips(conn, MockEmbedder(dim=8, name="mock-model"), a, source="deezer", signal_ratio=0.1)
    assert n == 1
    seg = conn.execute(
        "SELECT ce.segment_start_s, ce.segment_end_s FROM clip_embedding ce "
        "JOIN audio_track t ON t.id = ce.track_id WHERE t.artist_id = %s",
        (a,),
    ).fetchone()
    assert seg == (0, 30)
