"""ADR-021 Tier B: the embed pass keeps the compressed source clip so a future
model swap / windowing change re-analyzes locally instead of re-fetching. Pins:
the clip lands on the archive volume + in the ledger, the write is
idempotent-upsert, truncated clips are skipped, and BOTH failure modes (missing
file, DB error) are contained by the savepoint so archiving can never abort the
embed."""

from __future__ import annotations

from pipeline.embed_job import _try_archive, archive_source_clip


def _artist_track(conn) -> tuple[str, str]:
    a = conn.execute(
        "INSERT INTO artist (display_name) VALUES ('Archive Fixture') RETURNING id"
    ).fetchone()[0]
    t = conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
        "binding_tier, verification_status) VALUES (%s,'bandcamp','zz-arc-1','/x.mp3',90,'A','verified') "
        "RETURNING id",
        (a,),
    ).fetchone()[0]
    return str(a), str(t)


def _src(tmp_path, name="clip.mp3", size=4096) -> str:
    p = tmp_path / name
    p.write_bytes(b"\x00" * size)
    return str(p)


def test_archive_copies_and_records(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_ARCHIVE_DIR", str(tmp_path / "arc"))
    a, t = _artist_track(conn)
    archive_source_clip(conn, a, t, "bandcamp", _src(tmp_path))

    row = conn.execute(
        "SELECT artist_id::text, platform, rel_path, bytes FROM audio_clip_archive WHERE track_id=%s",
        (t,),
    ).fetchone()
    assert row[0] == a and row[1] == "bandcamp" and row[3] == 4096
    assert row[2] == f"{a}/{t}.mp3"
    assert (tmp_path / "arc" / row[2]).exists()


def test_archive_is_idempotent_upsert(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_ARCHIVE_DIR", str(tmp_path / "arc"))
    a, t = _artist_track(conn)
    archive_source_clip(conn, a, t, "bandcamp", _src(tmp_path, "a.mp3", 2048))
    archive_source_clip(conn, a, t, "bandcamp", _src(tmp_path, "a.mp3", 8192))  # re-fetch, bigger
    assert conn.execute(
        "SELECT count(*), max(bytes) FROM audio_clip_archive WHERE track_id=%s", (t,)
    ).fetchone() == (1, 8192)


def test_archive_skips_truncated(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_ARCHIVE_DIR", str(tmp_path / "arc"))
    a, t = _artist_track(conn)
    archive_source_clip(conn, a, t, "bandcamp", _src(tmp_path, "tiny.mp3", 512))  # <= 1024
    assert conn.execute(
        "SELECT count(*) FROM audio_clip_archive WHERE track_id=%s", (t,)
    ).fetchone()[0] == 0


def test_try_archive_swallows_missing_file(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_ARCHIVE_DIR", str(tmp_path / "arc"))
    a, t = _artist_track(conn)
    _try_archive(conn, a, t, "bandcamp", str(tmp_path / "nope.mp3"))  # no such file
    assert conn.execute(
        "SELECT count(*) FROM audio_clip_archive WHERE track_id=%s", (t,)
    ).fetchone()[0] == 0
    assert conn.execute("SELECT 1").fetchone()[0] == 1  # txn still usable


def test_try_archive_contains_db_error(conn, tmp_path, monkeypatch):
    # A real file, but a track_id with no audio_track row → the INSERT hits the
    # FK. The savepoint must contain it so the outer transaction survives.
    monkeypatch.setenv("PIPELINE_ARCHIVE_DIR", str(tmp_path / "arc"))
    a = conn.execute("INSERT INTO artist (display_name) VALUES ('Arc DB') RETURNING id").fetchone()[0]
    _try_archive(conn, str(a), "00000000-0000-4000-8000-0000000000ff", "bandcamp", _src(tmp_path))
    assert conn.execute("SELECT 1").fetchone()[0] == 1  # not aborted by the FK violation
