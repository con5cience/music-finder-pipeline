"""embed_job: registry-driven embed-and-store with model stamping (slice 2).

Uses the DB `conn` fixture + the torch-free MockEmbedder; audio "fetching" is a
no-op for local paths, and MockEmbedder hashes clip ids rather than reading
audio, so these tests exercise the full store path without model weights.
"""

from __future__ import annotations

import math
from pathlib import Path

from pipeline.bench.mock import MockEmbedder
from pipeline.embed_job import _audio_ext, embed_artist_clips, pending_tracks


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
        "SELECT ce.track_id, ce.segment_start_s, ce.segment_end_s, ce.model, ce.dim, vector_dims(ce.embedding) "
        "FROM clip_embedding ce JOIN audio_track t ON t.id = ce.track_id "
        "WHERE t.artist_id = %s ORDER BY ce.segment_end_s DESC",
        (a,),
    ).fetchall()
    assert rows == [(t1, 0, 30, "mock-model", 8, 8), (t2, 0, 25, "mock-model", 8, 8)]


def test_embed_is_idempotent_per_model(conn):
    a = _artist(conn, "A")
    _track(conn, a, "t1", "/audio/t1.mp3")
    assert embed_artist_clips(conn, _embedder(), a) == 1
    assert embed_artist_clips(conn, _embedder(), a) == 0  # nothing pending second time
    n = conn.execute(
        "SELECT count(*) FROM clip_embedding ce JOIN audio_track t ON t.id = ce.track_id WHERE t.artist_id = %s",
        (a,),
    ).fetchone()[0]
    assert n == 1


def test_second_model_is_additive(conn):
    # The ADR-016 swap story: a different model re-embeds the same clips.
    a = _artist(conn, "A")
    _track(conn, a, "t1", "/audio/t1.mp3")
    embed_artist_clips(conn, _embedder(), a)
    embed_artist_clips(conn, MockEmbedder(dim=8, name="mock-model-v2"), a)
    models = {
        r[0]
        for r in conn.execute(
            "SELECT ce.model FROM clip_embedding ce JOIN audio_track t ON t.id = ce.track_id "
            "WHERE t.artist_id = %s",
            (a,),
        ).fetchall()
    }
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


def test_audio_sniffer_names_for_content():
    # libsndfile's mp3 detection is extension-gated: the suffix must match
    # the sniffed content, never the URL tail.
    assert _audio_ext(b"ID3\x04\x00\x00rest") == ".mp3"
    assert _audio_ext(bytes([0xFF, 0xFB, 0x92, 0x64])) == ".mp3"  # raw MPEG sync
    assert _audio_ext(b"RIFF....WAVE") == ".wav"
    assert _audio_ext(b"OggS\x00") == ".ogg"
    assert _audio_ext(b"fLaC\x00") == ".flac"


def test_audio_sniffer_rejects_error_bodies():
    assert _audio_ext(b"<!DOCTYPE html><html>...") is None  # CDN sad page
    assert _audio_ext(b'{"error": "expired"}') is None
    assert _audio_ext(b"") is None


def test_expired_url_refreshes_and_embeds(conn):
    # Signed URL 403s → refresher provides a fresh URL → track embeds.
    from pipeline.embed_job import AudioFetchError

    a = _artist(conn, "A")
    _track(conn, a, "t1", "/expired/t1.mp3")

    def fake_fetch(url, workdir):
        if "/expired/" in url:
            raise AudioFetchError("audio fetch HTTP 403")
        return url

    refreshed = []

    def refresher(conn_, platform, ptid):
        refreshed.append((platform, ptid))
        return "/fresh/t1.mp3"

    n = embed_artist_clips(conn, _embedder(), a, fetch=fake_fetch, refresher=refresher)
    assert n == 1
    assert refreshed == [("deezer", "t1")]


def test_urlerror_becomes_audiofetcherror():
    # Review finding: only HTTPError was converted; URLError/timeouts escaped
    # the per-track isolation. 127.0.0.1:1 refuses instantly → URLError.
    import pytest

    from pipeline.embed_job import AudioFetchError, fetch_audio

    with pytest.raises(AudioFetchError):
        fetch_audio("http://127.0.0.1:1/nope.mp3", Path("/tmp"))


def test_all_selected_tracks_failing_raises_not_silent_success(conn):
    # Review finding: total fetch failure returned 0 → workflow completed
    # status='embedded' with no centroid. Must raise so Temporal retries and
    # failure is VISIBLE; tracks stay pending either way.
    import pytest

    from pipeline.embed_job import AudioFetchError

    a = _artist(conn, "A")
    _track(conn, a, "t1", "/expired/t1.mp3")
    _track(conn, a, "t2", "/expired/t2.mp3")

    def dead_fetch(url, workdir):
        raise AudioFetchError("audio fetch HTTP 403")

    with pytest.raises(AudioFetchError, match="all 2 selected"):
        embed_artist_clips(conn, _embedder(), a, fetch=dead_fetch, refresher=lambda *a_: None)
    assert len(pending_tracks(conn, a, "mock-model")) == 2  # still pending


def test_unrefreshable_track_is_skipped_not_poisonous(conn):
    # One dead track must not block the artist's other tracks (the calibration
    # stall: a single 403 retried the whole batch forever).
    from pipeline.embed_job import AudioFetchError

    a = _artist(conn, "A")
    _track(conn, a, "dead", "/expired/dead.mp3")
    t_ok = _track(conn, a, "ok", "/audio/ok.mp3")

    def fake_fetch(url, workdir):
        if "/expired/" in url:
            raise AudioFetchError("audio fetch HTTP 403")
        return url

    n = embed_artist_clips(conn, _embedder(), a, fetch=fake_fetch, refresher=lambda *args: None)
    assert n == 1  # the healthy track embedded; the dead one skipped
    rows = conn.execute(
        "SELECT ce.track_id FROM clip_embedding ce JOIN audio_track t ON t.id = ce.track_id "
        "WHERE t.artist_id = %s",
        (a,),
    ).fetchall()
    assert [r[0] for r in rows] == [t_ok]
    # the dead track stays pending for a later pass (no clip row, not rejected)
    assert len(pending_tracks(conn, a, "mock-model")) == 1


def test_yt_fetch_transcodes_to_wav(tmp_path, monkeypatch):
    """yt-dlp must request the FFmpegExtractAudio→wav postprocessor and the
    fetch must return the .wav path. The 2026-06-11 lesson: libsndfile cannot
    decode m4a AT ALL, so a raw bestaudio[ext=m4a] download burned a governed
    8-15s fetch and then failed prep with 'Format not recognised' — for every
    single youtube artist."""
    import yt_dlp

    from pipeline.embed_job import fetch_audio

    monkeypatch.setattr("pipeline.embed_job._YT_MIN_INTERVAL", 0.0)
    captured = {}

    class FakeYDL:
        def __init__(self, opts):
            captured.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            # the postprocessor leaves ONLY the transcoded wav behind
            (tmp_path / "yt-vid123.wav").write_bytes(b"RIFFfake")
            return {"ext": "m4a"}

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    path = fetch_audio("yt:vid123", tmp_path)
    assert path.endswith("yt-vid123.wav")
    assert Path(path).exists()
    pps = captured.get("postprocessors") or []
    assert any(p.get("key") == "FFmpegExtractAudio" and p.get("preferredcodec") == "wav" for p in pps)
