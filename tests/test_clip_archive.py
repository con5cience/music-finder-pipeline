"""ADR-021 Tier B v2: the embed pass keeps a COMPRESSED copy of the embedded
window clips (Opus) so a future model swap / re-window re-analyzes locally
instead of re-fetching. Pins: each window is encoded under <artist>/<track>/, the
ledger records the track dir + total bytes, the write is idempotent-replace, a
min-free-disk cap skips cleanly, and BOTH failure modes (encode error, DB error)
are contained by the savepoint so archiving can never abort the embed.

Host has no ffmpeg, so _encode_opus is mocked here; the real ffmpeg encode +
re-decode is verified against the worker image separately."""

from __future__ import annotations

from pathlib import Path

import pipeline.embed_job as ej
from pipeline.embed_job import _try_archive, archive_window_clips


def _fake_encode(monkeypatch, size: int = 5000):
    def enc(src_wav: str, dst_ogg: str) -> None:
        Path(dst_ogg).write_bytes(b"\x00" * size)  # stand in for a real Opus file
    monkeypatch.setattr(ej, "_encode_opus", enc)
    # neutralize the disk cap by default — tmp_path lives on a small tmpfs whose
    # free space is below the 200GB prod default (the cap test re-raises it).
    monkeypatch.setattr(ej, "ARCHIVE_MIN_FREE_BYTES", 0)


def _artist_track(conn) -> tuple[str, str]:
    a = conn.execute(
        "INSERT INTO artist (display_name) VALUES ('Archive Fixture') RETURNING id"
    ).fetchone()[0]
    t = conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
        "binding_tier, verification_status) VALUES (%s,'bandcamp','zz-arc-1','/x.mp3',180,'A','verified') "
        "RETURNING id",
        (a,),
    ).fetchone()[0]
    return str(a), str(t)


def _segs(tmp_path, spans=((0, 30), (60, 90), (120, 150))) -> list[tuple[int, int, str]]:
    """Staged window WAVs (dummy bytes — the encoder is mocked)."""
    out = []
    for i, (s, e) in enumerate(spans):
        p = tmp_path / f"w{i}.wav"
        p.write_bytes(b"RIFFdummywav")
        out.append((s, e, str(p)))
    return out


def test_encodes_each_window_and_records_track_dir(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_ARCHIVE_DIR", str(tmp_path / "arc"))
    _fake_encode(monkeypatch, size=5000)
    a, t = _artist_track(conn)

    archive_window_clips(conn, a, t, "bandcamp", _segs(tmp_path))

    d = tmp_path / "arc" / a / t
    assert sorted(p.name for p in d.glob("*.ogg")) == ["0_30.ogg", "120_150.ogg", "60_90.ogg"]
    row = conn.execute(
        "SELECT rel_path, bytes, platform FROM audio_clip_archive WHERE track_id=%s", (t,)
    ).fetchone()
    assert row == (f"{a}/{t}", 15000, "bandcamp")  # 3 windows x 5000 bytes


def test_idempotent_replace_drops_stale_windows(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_ARCHIVE_DIR", str(tmp_path / "arc"))
    _fake_encode(monkeypatch, size=5000)
    a, t = _artist_track(conn)

    archive_window_clips(conn, a, t, "bandcamp", _segs(tmp_path))  # 3 windows
    archive_window_clips(conn, a, t, "bandcamp", _segs(tmp_path, spans=((0, 30),)))  # re-prep: 1 window

    d = tmp_path / "arc" / a / t
    assert [p.name for p in d.glob("*.ogg")] == ["0_30.ogg"]  # stale 60_90/120_150 gone
    assert conn.execute(
        "SELECT bytes FROM audio_clip_archive WHERE track_id=%s", (t,)
    ).fetchone()[0] == 5000


def test_cap_skips_when_disk_below_min_free(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_ARCHIVE_DIR", str(tmp_path / "arc"))
    _fake_encode(monkeypatch)
    monkeypatch.setattr(ej, "ARCHIVE_MIN_FREE_BYTES", 10 ** 18)  # require an exabyte free -> always capped
    a, t = _artist_track(conn)

    archive_window_clips(conn, a, t, "bandcamp", _segs(tmp_path))

    assert conn.execute("SELECT count(*) FROM audio_clip_archive WHERE track_id=%s", (t,)).fetchone()[0] == 0
    assert not (tmp_path / "arc" / a / t).exists()  # nothing written


def test_empty_segs_is_noop(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_ARCHIVE_DIR", str(tmp_path / "arc"))
    a, t = _artist_track(conn)
    archive_window_clips(conn, a, t, "bandcamp", [])
    assert conn.execute("SELECT count(*) FROM audio_clip_archive WHERE track_id=%s", (t,)).fetchone()[0] == 0


def test_try_archive_swallows_encode_failure(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_ARCHIVE_DIR", str(tmp_path / "arc"))

    def boom(src_wav, dst_ogg):
        raise RuntimeError("ffmpeg exited non-zero")
    monkeypatch.setattr(ej, "_encode_opus", boom)
    a, t = _artist_track(conn)

    _try_archive(conn, a, t, "bandcamp", _segs(tmp_path))  # must not raise

    assert conn.execute("SELECT count(*) FROM audio_clip_archive WHERE track_id=%s", (t,)).fetchone()[0] == 0
    assert conn.execute("SELECT 1").fetchone()[0] == 1  # transaction still usable


def test_try_archive_contains_db_error(conn, tmp_path, monkeypatch):
    # Encode succeeds but the track_id has no audio_track row -> FK violation; the
    # savepoint must contain it so the outer (embed) transaction survives.
    monkeypatch.setenv("PIPELINE_ARCHIVE_DIR", str(tmp_path / "arc"))
    _fake_encode(monkeypatch)
    a = conn.execute("INSERT INTO artist (display_name) VALUES ('Arc DB') RETURNING id").fetchone()[0]
    _try_archive(conn, str(a), "00000000-0000-4000-8000-0000000000ff", "bandcamp", _segs(tmp_path))
    assert conn.execute("SELECT 1").fetchone()[0] == 1
