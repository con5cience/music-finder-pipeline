"""MB bootstrap (slice 3a): mbdump COPY load + Tier-A identity derivation.

Fixtures are synthetic mbdump files in the VERIFIED column layout (parsed from
musicbrainz-server CreateTables.sql, cross-checked against the real dump).
PG COPY text format: tab-separated, \\N for NULL.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.mb_bootstrap import PLATFORM_PATTERNS, derive_identities, load_mbdump

# MB row ids. GIDs and platform ids are FULLY SYNTHETIC (no real MBIDs, no real
# platform ids): the test DB doubles as the factory DB holding real bootstrap
# data, and fixture rows must never collide with real rows on any unique key.
A_BURIAL, A_HOMEPAGE_ONLY, A_ENDED_REL, A_MULTI = 1, 2, 3, 4
GID = {
    A_BURIAL: "00000000-feed-4bad-9bad-000000000001",
    A_HOMEPAGE_ONLY: "00000000-feed-4bad-9bad-000000000002",
    A_ENDED_REL: "00000000-feed-4bad-9bad-000000000003",
    A_MULTI: "00000000-feed-4bad-9bad-000000000004",
}


def _w(d: Path, name: str, rows: list[list[str]]) -> None:
    (d / name).write_text("".join("\t".join(r) + "\n" for r in rows))


@pytest.fixture
def dump_dir(tmp_path: Path) -> Path:
    d = tmp_path / "mbdump"
    d.mkdir()
    # artist: id gid name sort_name b_y b_m b_d e_y e_m e_d type area gender
    #         comment edits_pending last_updated ended begin_area end_area (19)
    def artist_row(aid: int, name: str, comment: str = "") -> list[str]:
        nine = ["\\N"] * 9
        return [str(aid), GID[aid], name, name, *nine, comment, "0", "2026-01-01 00:00:00+00", "f", "\\N", "\\N"]

    _w(d, "artist", [
        artist_row(A_BURIAL, "Burial", "Will Bevan"),
        artist_row(A_HOMEPAGE_ONLY, "Nowhere Man"),
        artist_row(A_ENDED_REL, "Gone Cat"),
        artist_row(A_MULTI, "Everywhere Band"),
    ])
    # url: id gid url edits_pending last_updated (5)
    _w(d, "url", [
        ["10", "aaaaaaa1-0000-0000-0000-000000000001", "https://www.deezer.com/artist/990000000001", "0", "\\N"],
        ["11", "aaaaaaa1-0000-0000-0000-000000000002", "https://zz-test-fixture.bandcamp.com", "0", "\\N"],
        ["12", "aaaaaaa1-0000-0000-0000-000000000003", "https://www.example.com/nowhere", "0", "\\N"],
        ["13", "aaaaaaa1-0000-0000-0000-000000000004", "https://soundcloud.com/zz-test-gonecat", "0", "\\N"],
        ["14", "aaaaaaa1-0000-0000-0000-000000000005", "https://soundcloud.com/zz-test-everywhere", "0", "\\N"],
        ["15", "aaaaaaa1-0000-0000-0000-000000000006",
         "https://www.youtube.com/channel/UCzztestfixture00000000", "0", "\\N"],
        ["16", "aaaaaaa1-0000-0000-0000-000000000007",
         "https://tidal.com/browse/artist/990000000002", "0", "\\N"],
        ["17", "aaaaaaa1-0000-0000-0000-000000000008",
         "https://music.apple.com/us/artist/zz-test/990000000003", "0", "\\N"],
        ["18", "aaaaaaa1-0000-0000-0000-000000000009",
         "https://www.qobuz.com/jp-ja/interpreter/-/990000000004", "0", "\\N"],
    ])
    # link_type: id parent child_order gid e0 e1 name description link_phrase
    #            reverse long last_updated is_deprecated has_dates c0 c1 (16)
    def link_type_row(ltid: int, name: str) -> list[str]:
        gid = f"bbbbbbb1-0000-0000-0000-{ltid:012d}"
        return [str(ltid), "\\N", "0", gid, "artist", "url", name, "d", "p", "rp", "lp", "\\N", "f", "f", "0", "0"]

    _w(d, "link_type", [
        link_type_row(100, "free streaming"),
        link_type_row(101, "bandcamp"),
        link_type_row(102, "official homepage"),
        link_type_row(103, "soundcloud"),
        link_type_row(104, "youtube"),
        link_type_row(105, "streaming"),
    ])
    # link: id link_type b_y b_m b_d e_y e_m e_d attribute_count created ended (11)
    six = ["\\N"] * 6
    _w(d, "link", [
        ["200", "100", *six, "0", "\\N", "f"],  # free streaming, active
        ["201", "101", *six, "0", "\\N", "f"],  # bandcamp, active
        ["202", "102", *six, "0", "\\N", "f"],  # homepage, active
        ["203", "103", *six, "0", "\\N", "t"],  # soundcloud, ENDED
        ["204", "103", *six, "0", "\\N", "f"],  # soundcloud, active
        ["205", "104", *six, "0", "\\N", "f"],  # youtube, active
        ["206", "105", *six, "0", "\\N", "f"],  # streaming (tidal), active
        ["207", "105", *six, "0", "\\N", "f"],  # streaming (apple), active
        ["208", "105", *six, "0", "\\N", "f"],  # streaming (qobuz), active
    ])
    # l_artist_url: id link entity0(artist) entity1(url) edits_pending
    #               last_updated link_order e0_credit e1_credit (9)
    _w(d, "l_artist_url", [
        ["300", "200", str(A_BURIAL), "10", "0", "\\N", "0", "", ""],
        ["301", "201", str(A_BURIAL), "11", "0", "\\N", "0", "", ""],
        ["302", "202", str(A_HOMEPAGE_ONLY), "12", "0", "\\N", "0", "", ""],
        ["303", "203", str(A_ENDED_REL), "13", "0", "\\N", "0", "", ""],  # ended → skip
        ["304", "204", str(A_MULTI), "14", "0", "\\N", "0", "", ""],
        ["305", "205", str(A_MULTI), "15", "0", "\\N", "0", "", ""],
        ["306", "206", str(A_MULTI), "16", "0", "\\N", "0", "", ""],
        ["307", "207", str(A_BURIAL), "17", "0", "\\N", "0", "", ""],
        ["308", "208", str(A_BURIAL), "18", "0", "\\N", "0", "", ""],
    ])
    # artist_tag: artist tag count last_updated (4)  /  tag: id name ref_count (3)
    _w(d, "tag", [["400", "dubstep", "1"], ["401", "electronic", "1"]])
    _w(d, "artist_tag", [[str(A_BURIAL), "400", "5", "\\N"], [str(A_BURIAL), "401", "3", "\\N"]])
    # artist_alias: id artist name locale edits_pending last_updated type
    #               sort_name b_y b_m b_d e_y e_m e_d primary_for_locale ended (16)
    alias = "William Emmanuel Bevan"
    _w(d, "artist_alias", [
        ["500", str(A_BURIAL), alias, "\\N", "0", "\\N", "\\N", alias, *["\\N"] * 6, "f", "f"],
    ])
    # genre: id gid name comment edits_pending last_updated (6)
    _w(d, "genre", [
        ["600", "ccccccc1-0000-0000-0000-000000000001", "synth-punk", "", "0", "\\N"],
        ["601", "ccccccc1-0000-0000-0000-000000000002", "dubstep", "", "0", "\\N"],
    ])
    # genre_alias: id genre name locale edits_pending last_updated type sort_name
    #              b_y b_m b_d e_y e_m e_d primary_for_locale ended (16)
    _w(d, "genre_alias", [
        ["700", "600", "synth punk", "en", "0", "\\N", "1", "synth punk", *["\\N"] * 6, "f", "f"],
        ["701", "600", "synthpunk", "en", "0", "\\N", "1", "synthpunk", *["\\N"] * 6, "f", "f"],
    ])
    return d


@pytest.fixture
def loaded(conn, dump_dir):
    load_mbdump(conn, dump_dir)
    return conn


def test_load_row_counts(loaded):
    for table, n in [("artist", 4), ("url", 9), ("link_type", 6), ("link", 9),
                     ("l_artist_url", 9), ("tag", 2), ("artist_tag", 2), ("artist_alias", 1),
                     ("genre", 2), ("genre_alias", 2)]:
        assert loaded.execute(f"SELECT count(*) FROM mb_raw.{table}").fetchone()[0] == n, table


def test_load_is_idempotent_refresh(loaded, dump_dir):
    load_mbdump(loaded, dump_dir)  # refresh = truncate + reload, not append
    assert loaded.execute("SELECT count(*) FROM mb_raw.artist").fetchone()[0] == 4


# The dev/test DB may also hold REAL bootstrap data (it's the factory DB), so
# every assertion below scopes to the fixture's own MBIDs — never global counts.
_FIXTURE_GIDS = list(GID.values())


def _fixture_identities(conn) -> set[tuple[str, str]]:
    return {
        (p, pid)
        for p, pid in conn.execute(
            "SELECT pi.platform, pi.platform_id FROM platform_identity pi "
            "JOIN artist a ON a.id = pi.artist_id WHERE a.mbid = ANY(%s::uuid[])",
            (_FIXTURE_GIDS,),
        ).fetchall()
    }


def test_derive_creates_artists_with_mbid(loaded):
    derive_identities(loaded)
    rows = dict(
        loaded.execute(
            "SELECT mbid::text, display_name FROM artist WHERE mbid = ANY(%s::uuid[])", (_FIXTURE_GIDS,)
        ).fetchall()
    )
    assert rows[GID[A_BURIAL]] == "Burial"
    assert rows[GID[A_MULTI]] == "Everywhere Band"
    # homepage-only and ended-rel-only artists are NOT derived
    assert GID[A_HOMEPAGE_ONLY] not in rows
    assert GID[A_ENDED_REL] not in rows


def test_derive_platform_id_extraction(loaded):
    derive_identities(loaded)
    got = _fixture_identities(loaded)
    assert ("deezer", "990000000001") in got      # numeric from /artist/<id>
    assert ("bandcamp", "zz-test-fixture") in got  # subdomain
    assert ("soundcloud", "zz-test-everywhere") in got  # permalink
    assert ("youtube", "UCzztestfixture00000000") in got  # channel id
    assert ("tidal", "990000000002") in got       # numeric
    assert ("apple_music", "990000000003") in got  # numeric
    assert ("qobuz", "990000000004") in got       # trailing numeric, slug is "-"
    # the ENDED soundcloud rel must not appear
    assert ("soundcloud", "zz-test-gonecat") not in got


def test_derive_identity_provenance(loaded):
    derive_identities(loaded)
    rows = loaded.execute(
        "SELECT pi.page_type, pi.vanity_url FROM platform_identity pi "
        "JOIN artist a ON a.id = pi.artist_id WHERE a.mbid = ANY(%s::uuid[])",
        (_FIXTURE_GIDS,),
    ).fetchall()
    assert {r[0] for r in rows} == {"artist"}
    assert "https://zz-test-fixture.bandcamp.com" in {r[1] for r in rows}


def test_derive_multi_platform_artist_one_row(loaded):
    derive_identities(loaded)
    n_art = loaded.execute("SELECT count(*) FROM artist WHERE mbid=%s::uuid", (GID[A_MULTI],)).fetchone()[0]
    n_ids = loaded.execute(
        "SELECT count(*) FROM platform_identity pi JOIN artist a ON a.id=pi.artist_id WHERE a.mbid=%s::uuid",
        (GID[A_MULTI],),
    ).fetchone()[0]
    assert (n_art, n_ids) == (1, 3)  # soundcloud + youtube + tidal


def test_derive_is_idempotent(loaded):
    derive_identities(loaded)
    derive_identities(loaded)
    n_artists = loaded.execute(
        "SELECT count(*) FROM artist WHERE mbid = ANY(%s::uuid[])", (_FIXTURE_GIDS,)
    ).fetchone()[0]
    assert n_artists == 2  # A_BURIAL + A_MULTI; homepage-only and ended-rel never derive
    before = _fixture_identities(loaded)
    derive_identities(loaded)
    assert _fixture_identities(loaded) == before


def test_platform_patterns_cover_locked_platforms():
    assert {"deezer", "bandcamp", "soundcloud", "youtube", "tidal", "apple_music", "qobuz"} <= set(PLATFORM_PATTERNS)
