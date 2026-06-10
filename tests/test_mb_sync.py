"""mb-sync acquisition half: md5 fail-closed verify, member extraction
(flattened, core+derived split), already-applied skip. Network mocked."""

from __future__ import annotations

import hashlib
import tarfile

import pytest

from pipeline.mb_sync import CORE_TABLES, DERIVED_TABLES, already_applied, extract_tables, verify_md5


def test_verify_md5_pass_and_fail(tmp_path):
    a = tmp_path / "mbdump.tar.bz2"
    a.write_bytes(b"dump-bytes")
    good = hashlib.md5(b"dump-bytes").hexdigest()
    sums = tmp_path / "MD5SUMS"
    sums.write_text(f"{good}  mbdump.tar.bz2\n")
    verify_md5(a, sums)  # passes silently
    sums.write_text(f"{'0'*32}  mbdump.tar.bz2\n")
    with pytest.raises(RuntimeError, match="md5 mismatch"):
        verify_md5(a, sums)
    sums.write_text("deadbeef  other.tar.bz2\n")
    with pytest.raises(RuntimeError, match="not present"):
        verify_md5(a, sums)


def test_extract_tables_flattens_members(tmp_path):
    src = tmp_path / "payload"
    (src / "mbdump").mkdir(parents=True)
    for t in ("artist", "url", "ignored_table"):
        (src / "mbdump" / t).write_text(f"{t}-data\n")
    archive = tmp_path / "mbdump.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        tar.add(src / "mbdump", arcname="mbdump")
    out = tmp_path / "out"
    out.mkdir()
    extract_tables(archive, ["artist", "url"], out)
    assert (out / "artist").read_text() == "artist-data\n"
    assert (out / "url").exists() and not (out / "ignored_table").exists()


def test_table_split_matches_refresh_contract():
    from pipeline.mb_refresh import REFRESH_TABLES

    assert set(CORE_TABLES) | set(DERIVED_TABLES) == set(REFRESH_TABLES)
    # the bootstrap-era lesson, pinned: tags live in the DERIVED archive
    assert "artist_tag" in DERIVED_TABLES and "tag" in DERIVED_TABLES


def test_already_applied_keyed_on_serial_and_applied(conn):
    assert already_applied(conn, "20990101-000000") is False
    conn.execute("INSERT INTO mb_refresh_run (gates, serial) VALUES ('{}', '20990101-000000')")
    assert already_applied(conn, "20990101-000000") is False  # dry-run doesn't count
    conn.execute(
        "INSERT INTO mb_refresh_run (gates, serial, applied_at) VALUES ('{}', '20990101-000000', now())"
    )
    assert already_applied(conn, "20990101-000000") is True
