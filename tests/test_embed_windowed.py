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


def test_rerun_respects_track_budget(conn, tmp_path):
    # Review finding: re-runs embedded 3 MORE pending tracks each time. The
    # budget must count tracks ALREADY embedded for this (source, model).
    a = _artist(conn)
    for i in range(5):
        _bc_track(conn, a, f"zz-w-b{i}", _wav(tmp_path, f"b{i}", 90), 90, f"/album/r{i}", i)
    emb = MockEmbedder(dim=8, name="mock-model")
    n1 = embed_artist_clips(conn, emb, a, source="bandcamp", signal_ratio=1.0)
    assert n1 > 0
    tracks_after_1 = conn.execute(
        "SELECT count(DISTINCT ce.track_id) FROM clip_embedding ce "
        "JOIN audio_track t ON t.id = ce.track_id WHERE t.artist_id = %s", (a,)
    ).fetchone()[0]
    assert tracks_after_1 == 3  # the budget
    n2 = embed_artist_clips(conn, emb, a, source="bandcamp", signal_ratio=1.0)
    assert n2 == 0  # budget already spent — re-run embeds NOTHING more
    tracks_after_2 = conn.execute(
        "SELECT count(DISTINCT ce.track_id) FROM clip_embedding ce "
        "JOIN audio_track t ON t.id = ce.track_id WHERE t.artist_id = %s", (a,)
    ).fetchone()[0]
    assert tracks_after_2 == 3


def test_windowed_rerun_is_idempotent(conn, tmp_path):
    a = _artist(conn)
    _bc_track(conn, a, "zz-w-r1", _wav(tmp_path, "r1", 90), 90, "/album/x", 0)
    emb = MockEmbedder(dim=8, name="mock-model")
    n1 = embed_artist_clips(conn, emb, a, source="bandcamp", signal_ratio=0.33)
    assert n1 >= 2
    n2 = embed_artist_clips(conn, emb, a, source="bandcamp", signal_ratio=0.33)
    assert n2 == 0  # track has clips for this model → not pending


def test_analysis_and_tag_heads_run_in_embed_pass(conn, tmp_path):
    # decode-once integration: embedding an artist also writes track_analysis
    # (CPU heads) and track_tag_scores (via the injected scorer).
    class FakeScorer:
        def embed_clips(self, artist_id, clip_paths):
            assert clip_paths  # window files exist at scoring time
            import numpy as np
            return np.ones((len(clip_paths), 4), dtype=np.float32)

        def score_vectors(self, vecs, top_k=20):
            return [("zz-fake-genre", 0.42)]

    a = _artist(conn)
    _bc_track(conn, a, "zz-w-h1", _wav(tmp_path, "h1", 90), 90, "/album/x", 0)
    from pipeline.heads import CpuAnalysisHead, TagHead

    n = embed_artist_clips(
        conn, MockEmbedder(dim=8, name="mock-model"), a,
        source="bandcamp", signal_ratio=0.33,
        heads=[CpuAnalysisHead(), TagHead(FakeScorer())],
    )
    assert n >= 2
    analysis = conn.execute(
        "SELECT ta.integrity, ta.tempo_bpm IS NOT NULL, ta.fingerprint IS NOT NULL "
        "FROM track_analysis ta JOIN audio_track t ON t.id = ta.track_id WHERE t.artist_id = %s",
        (a,),
    ).fetchone()
    assert analysis is not None and analysis[0] == "ok" and analysis[1] and analysis[2]
    tags = conn.execute(
        "SELECT tag, score FROM track_tag_scores tts JOIN audio_track t ON t.id = tts.track_id "
        "WHERE t.artist_id = %s",
        (a,),
    ).fetchall()
    assert tags == [("zz-fake-genre", 0.42)]


def test_preview_platform_still_single_clip(conn, tmp_path):
    # regression: deezer path is untouched by windowing
    a = _artist(conn)
    conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
        "binding_tier, verification_status) VALUES (%s, 'deezer', 'zz-w-d1', '/audio/d1.mp3', 30, 'A', 'verified')",
        (a,),
    )
    n = embed_artist_clips(
        conn, MockEmbedder(dim=8, name="mock-model"), a,
        source="deezer", signal_ratio=0.1,  # no heads (default): mechanics only
    )
    assert n == 1
    seg = conn.execute(
        "SELECT ce.segment_start_s, ce.segment_end_s FROM clip_embedding ce "
        "JOIN audio_track t ON t.id = ce.track_id WHERE t.artist_id = %s",
        (a,),
    ).fetchone()
    assert seg == (0, 30)


def test_preview_platform_respects_track_budget(conn, tmp_path):
    # Observability catch: SC (non-windowed) embedded ALL ~50 stored previews
    # (49 clips vs deezer's 12) — the budget only capped windowed platforms.
    # Preview sources cap at PREVIEW_TRACKS_CAP newest-by-walk-order, and
    # re-runs must not creep past it (same mechanics as the windowed budget).
    import json as _json

    from pipeline.embed_job import PREVIEW_TRACKS_CAP

    a = _artist(conn)
    for i in range(PREVIEW_TRACKS_CAP + 8):
        conn.execute(
            "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
            "binding_tier, binding_evidence, verification_status) "
            "VALUES (%s, 'soundcloud', %s, %s, 30, 'A', %s, 'verified')",
            (a, f"zz-pcap-{i}", f"/audio/pcap-{i}.mp3",
             _json.dumps({"release_index": i, "track_index": i})),
        )
    emb = MockEmbedder(dim=8, name="mock-model")
    n1 = embed_artist_clips(conn, emb, a, source="soundcloud", signal_ratio=2.0)
    assert n1 == PREVIEW_TRACKS_CAP
    n2 = embed_artist_clips(conn, emb, a, source="soundcloud", signal_ratio=2.0)
    assert n2 == 0  # no creep on re-run
    newest = conn.execute(
        "SELECT (t.binding_evidence->>'track_index')::int FROM clip_embedding ce "
        "JOIN audio_track t ON t.id = ce.track_id WHERE t.artist_id = %s "
        "ORDER BY 1 DESC LIMIT 1", (a,),
    ).fetchone()[0]
    assert newest == PREVIEW_TRACKS_CAP - 1  # newest-first selection, not random


def test_fetch_audio_yt_scheme(monkeypatch, tmp_path):
    """The 2026-06-11 gate: yt:<id> routes through the governed extractor;
    the kill switch raises AudioFetchError (track stays pending, never a
    poisoned verdict)."""
    from pipeline import embed_job
    from pipeline.embed_job import AudioFetchError, fetch_audio

    calls = {}

    class FakeYDL:
        def __init__(self, opts):
            calls["opts"] = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            calls["url"] = url
            # the FFmpegExtractAudio postprocessor leaves only the wav
            (tmp_path / "yt-vid123.wav").write_bytes(b"x")
            return {"ext": "m4a"}

    import sys
    import types

    fake = types.SimpleNamespace(YoutubeDL=FakeYDL, utils=types.SimpleNamespace(DownloadError=Exception))
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)
    monkeypatch.setattr(embed_job, "_YT_MIN_INTERVAL", 0.0)  # no politeness sleep in tests
    out = fetch_audio("yt:vid123", tmp_path)
    assert out.endswith("yt-vid123.wav")  # m4a is undecodable; wav is the contract
    assert calls["url"] == "https://www.youtube.com/watch?v=vid123"
    assert calls["opts"]["noplaylist"] is True

    monkeypatch.setenv("PIPELINE_YT_EXTRACTION", "0")
    import pytest

    with pytest.raises(AudioFetchError, match="disabled"):
        fetch_audio("yt:vid999", tmp_path)


def test_yt_audio_format_caps_bitrate():
    """Fetch-size reduction (task: YouTube fetch size): the format selector
    prefers a <=cap-kbps audio stream (the embedder downsamples to ~24kHz mono,
    so bestaudio's 128-160kbps is over-spec) while keeping a hard bestaudio
    fallback so extraction never fails on formats with no abr tag."""
    from pipeline.embed_job import _yt_audio_format

    assert _yt_audio_format(80) == (
        "bestaudio[abr<=80][ext=m4a]/bestaudio[abr<=80]/bestaudio[ext=m4a]/bestaudio"
    )
    assert "abr<=64" in _yt_audio_format(64)        # cap is honored
    assert _yt_audio_format(50).endswith("/bestaudio")  # unconstrained final fallback


def test_fetch_audio_yt_passes_capped_format(monkeypatch, tmp_path):
    """_fetch_youtube hands yt-dlp the bitrate-capped format (the actual
    fetch-size lever), driven by the env-overridable _YT_MAX_ABR knob."""
    import sys
    import types

    from pipeline import embed_job
    from pipeline.embed_job import fetch_audio

    calls = {}

    class FakeYDL:
        def __init__(self, opts):
            calls["opts"] = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            (tmp_path / "yt-vidX.wav").write_bytes(b"x")
            return {"ext": "m4a"}

    fake = types.SimpleNamespace(YoutubeDL=FakeYDL, utils=types.SimpleNamespace(DownloadError=Exception))
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)
    monkeypatch.setattr(embed_job, "_YT_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(embed_job, "_YT_MAX_ABR", 64)
    fetch_audio("yt:vidX", tmp_path)
    assert calls["opts"]["format"] == (
        "bestaudio[abr<=64][ext=m4a]/bestaudio[abr<=64]/bestaudio[ext=m4a]/bestaudio"
    )


def test_youtube_floor_open():
    from pipeline.cascade import choose_source
    from pipeline.queues import PLATFORMS

    assert PLATFORMS["youtube"].floor == 4
    # last resort: yt only wins when everything above fails its floor
    assert choose_source({"deezer": 12, "youtube": 9})[0] == "deezer"
    choice = choose_source({"deezer": 1, "bandcamp": 0, "soundcloud": 2, "youtube": 6})
    assert choice[0] == "youtube" and choice[1] >= 1
