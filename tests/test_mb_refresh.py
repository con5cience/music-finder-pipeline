"""ADR-018 refresh: shadow load, fail-closed gates, derive-diff adds/renames,
merge paths (simple repoint, keep-embedded, both-embedded → review), swap.
Hermetic: fixture-scale dumps via the bootstrap test helpers."""

from __future__ import annotations

import pytest

from pipeline.mb_refresh import REFRESH_TABLES, diff_and_apply, prepare_shadow, run_refresh, sanity_gates
from test_mb_bootstrap import build_fixture_dump


def _write_redirect(dump_dir, rows):
    with open(dump_dir / "artist_gid_redirect", "w") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")


def test_refresh_loads_nonempty_redirect_file(conn, tmp_path):
    # review finding: _assert_layout KeyError'd on artist_gid_redirect because
    # it ignored the passed tables dict — only EMPTY fixture files dodged it
    d = build_fixture_dump(tmp_path)
    _write_redirect(d, [["00000000-feed-4bad-9bad-00000000aa41", "1", "2026-01-01 00:00:00+00"]])
    report = run_refresh(conn, d, apply=False, serial="20990101-000001")
    assert report["gates"]["ok"] is True  # load survived the non-empty redirect


def test_gate_failure_gets_own_ledger_row_not_applied(conn, refresh_dump):
    # review finding: gate failure used to stamp its serial onto the PREVIOUS
    # (possibly applied) row via UPDATE max(id) — fail-open on already_applied
    conn.execute("INSERT INTO mb_raw.artist (id, gid, name, sort_name) "
                 "SELECT 99100000 + n, gen_random_uuid(), 'Filler', 'Filler' FROM generate_series(1, 50) n")

    report = run_refresh(conn, refresh_dump, apply=True, serial="20990101-000002")
    assert "aborted" in report
    serial, applied = conn.execute(
        "SELECT serial, applied_at FROM mb_refresh_run ORDER BY id DESC LIMIT 1").fetchone()
    assert serial == "20990101-000002" and applied is None
    from pipeline.mb_sync import already_applied
    assert already_applied(conn, "20990101-000002") is False


@pytest.fixture
def refresh_dump(tmp_path):
    d = build_fixture_dump(tmp_path)
    _write_redirect(d, [])
    return d


def test_gates_abort_on_truncated_dump(conn, refresh_dump):
    # live mb_raw holds the REAL corpus (543k artists); the fixture dump has 4
    # → gates must abort with live state untouched. (In the test DB mb_raw is
    # fixture-scale too, so simulate by shrinking the next side.)
    from pipeline.mb_bootstrap import load_mbdump

    prepare_shadow(conn)
    load_mbdump(conn, refresh_dump, schema="mb_raw_next", tables=REFRESH_TABLES)
    conn.execute("DELETE FROM mb_raw_next.artist")  # truncated artist table
    conn.execute("INSERT INTO mb_raw.artist (id, gid, name, sort_name) VALUES "
                 "(99000020, '00000000-feed-4bad-9bad-00000000aa20', 'Gatekeeper', 'Gatekeeper')")
    gates = sanity_gates(conn)
    assert gates["ok"] is False


def test_refresh_dry_run_reports_without_applying(conn, refresh_dump):
    before = conn.execute("SELECT count(*) FROM artist").fetchone()[0]
    report = run_refresh(conn, refresh_dump, apply=False)
    assert report["gates"]["ok"] is True
    assert "adds" in report and report.get("new_identities") is None  # nothing applied
    assert conn.execute("SELECT count(*) FROM artist").fetchone()[0] == before
    run_row = conn.execute(
        "SELECT applied_at FROM mb_refresh_run ORDER BY id DESC LIMIT 1").fetchone()
    assert run_row[0] is None  # ledgered as dry-run


def test_refresh_apply_adds_and_swaps(conn, refresh_dump):
    report = run_refresh(conn, refresh_dump, apply=True)
    assert report["gates"]["ok"] is True
    assert report["new_identities"] >= 0
    # swap happened: mb_raw is the new generation, old kept one cycle
    assert conn.execute(
        "SELECT count(*) FROM information_schema.schemata WHERE schema_name = 'mb_raw_old'"
    ).fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM mb_raw.artist").fetchone()[0] > 0


def test_merge_both_embedded_goes_to_review(conn, refresh_dump):
    # two LOCAL artists, both embedded; the new dump merges old→new mbid
    old = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES "
        "('Merge Old', '00000000-feed-4bad-9bad-00000000aa31', 'deezer') RETURNING id").fetchone()[0]
    conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES "
        "('Merge New', '00000000-feed-4bad-9bad-00000000aa32', 'bandcamp')")
    from pipeline.mb_bootstrap import load_mbdump

    prepare_shadow(conn)
    load_mbdump(conn, refresh_dump, schema="mb_raw_next", tables=REFRESH_TABLES)
    conn.execute(
        "INSERT INTO mb_raw_next.artist (id, gid, name, sort_name) VALUES "
        "(99000032, '00000000-feed-4bad-9bad-00000000aa32', 'Merge New', 'Merge New')")
    conn.execute(
        "INSERT INTO mb_raw_next.artist_gid_redirect (gid, new_id) VALUES "
        "('00000000-feed-4bad-9bad-00000000aa31', 99000032)")
    report = diff_and_apply(conn, apply=True)
    assert report["reviews"] == 1
    kind, ev = conn.execute(
        "SELECT kind, evidence FROM review_item WHERE subject_id = %s", (old,)).fetchone()
    assert kind == "source_binding"
    assert ev["mb_merge"]["new_mbid"] == "00000000-feed-4bad-9bad-00000000aa32"
    # both rows survive — never auto-pick between two centroids
    assert conn.execute(
        "SELECT count(*) FROM artist WHERE mbid IN "
        "('00000000-feed-4bad-9bad-00000000aa31','00000000-feed-4bad-9bad-00000000aa32')"
    ).fetchone()[0] == 2
