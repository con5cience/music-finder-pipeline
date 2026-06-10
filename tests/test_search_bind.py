"""3d B-tier search binding: normalization, the exact-unique policy, the
homonym→Tier-C path, the no-match ledger, and never-research idempotency.
Hazards encoded from live probes: 'Tomasito' = 3 bandcamp accounts with one
normalized name; popularity must NEVER auto-pick among homonyms."""

from __future__ import annotations

from pipeline.search_bind import (
    artist_name_keys,
    bind_artist_on_platform,
    normalize_name,
    unbound_artists,
)

MBID = "00000000-feed-4bad-9bad-000000000888"


def _artist(conn, name="Search Fixture", mbid=MBID) -> str:
    return conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES (%s, %s) RETURNING id", (name, mbid)
    ).fetchone()[0]


def _searcher(candidates):
    return lambda conn, name: candidates


def test_normalize_name_diacritics_case_punct():
    assert normalize_name("Delić") == "delic"
    assert normalize_name("KAN3KI") == "kan3ki"
    assert normalize_name("Synth-Pop!") == normalize_name("synth pop")
    assert normalize_name("Abdon Mech") == "abdonmech"


def test_exact_unique_binds_tier_b(conn):
    a = _artist(conn)
    v = bind_artist_on_platform(
        conn, str(a), "Search Fixture", "deezer",
        searcher=_searcher([
            {"name": "Search Fixture", "platform_id": "990001", "popularity": 5},
            {"name": "Completely Other", "platform_id": "990002", "popularity": 99999},
        ]),
    )
    assert v == "bound"
    tier, ev, status = conn.execute(
        "SELECT binding_tier, binding_evidence, scan_status FROM platform_identity "
        "WHERE artist_id = %s AND platform = 'deezer'", (a,),
    ).fetchone()
    assert tier == "B"
    assert ev["method"] == "search_exact_unique"
    assert status == "pending"  # the cascade picks B identities up like any other


def test_homonyms_go_to_review_never_popularity(conn):
    # the Tomasito hazard: multiple exact matches; the popular one must NOT win
    a = _artist(conn)
    v = bind_artist_on_platform(
        conn, str(a), "Search Fixture", "bandcamp",
        searcher=_searcher([
            {"name": "Search Fixture", "platform_id": "zz-sf-1", "popularity": 100000},
            {"name": "search fixture", "platform_id": "zz-sf-2", "popularity": 3},
        ]),
    )
    assert v == "review"
    assert conn.execute(
        "SELECT count(*) FROM platform_identity WHERE artist_id = %s AND platform = 'bandcamp'", (a,)
    ).fetchone()[0] == 0  # NOTHING bound
    kind, ev = conn.execute(
        "SELECT kind, evidence FROM review_item WHERE subject_id = %s", (a,)
    ).fetchone()
    assert kind == "source_binding"
    assert len(ev["candidates"]) == 2  # admin sees both


def test_no_match_records_none_and_never_researches(conn):
    a = _artist(conn)
    calls = []

    def counting_searcher(conn_, name):
        calls.append(name)
        return [{"name": "Unrelated Act", "platform_id": "zz-x", "popularity": 1}]

    v1 = bind_artist_on_platform(conn, str(a), "Search Fixture", "deezer", searcher=counting_searcher)
    v2 = bind_artist_on_platform(conn, str(a), "Search Fixture", "deezer", searcher=counting_searcher)
    assert (v1, v2) == ("none", "skipped")
    assert len(calls) == 1  # the ledger prevents the second search


def test_alias_match_binds(conn):
    # MB aliases participate: artist known as 'Burial' with alias 'William
    # Emmanuel Bevan' must match a candidate under the alias.
    a = _artist(conn, name="ZZ Alias Fixture", mbid="00000000-feed-4bad-9bad-000000000889")
    conn.execute(
        "INSERT INTO mb_raw.artist (id, gid, name, sort_name) VALUES "
        "(99000010, '00000000-feed-4bad-9bad-000000000889', 'ZZ Alias Fixture', 'ZZ Alias Fixture')"
    )
    conn.execute(
        "INSERT INTO mb_raw.artist_alias (id, artist, name, sort_name) VALUES "
        "(99000011, 99000010, 'The Zed Zed Project', 'Zed Zed Project, The')"
    )
    keys = artist_name_keys(conn, str(a), "ZZ Alias Fixture")
    assert "thezedzedproject" in keys
    v = bind_artist_on_platform(
        conn, str(a), "ZZ Alias Fixture", "soundcloud",
        searcher=_searcher([{"name": "The Zed Zed Project", "platform_id": "zz-alias-sc", "popularity": 2}]),
    )
    assert v == "bound"


def test_unbound_artists_excludes_searched_and_bound(conn):
    a = _artist(conn)
    ids = {r[0] for r in unbound_artists(conn, 10_000_000)}
    assert a in ids
    for p in ("deezer", "bandcamp", "soundcloud"):
        conn.execute(
            "INSERT INTO search_attempt (artist_id, platform, query, verdict) "
            "VALUES (%s, %s, 'x', 'none')", (a, p),
        )
    ids = {r[0] for r in unbound_artists(conn, 10_000_000)}
    assert a not in ids  # fully searched → out of the queue


def test_review_poller_applies_approved_decision(conn):
    from pipeline.review_poller import apply_approved_bindings

    a = _artist(conn, name="ZZ Poller Fixture", mbid="00000000-feed-4bad-9bad-000000000890")
    conn.execute(
        "INSERT INTO review_item (kind, subject_type, subject_id, reason, evidence, status) "
        "VALUES ('source_binding', 'artist', %s, '2 candidates', %s, 'approved')",
        (a, '{"platform": "bandcamp", "candidates": [], '
            '"decision": {"platform": "bandcamp", "platform_id": "zz-chosen"}}'),
    )
    assert apply_approved_bindings(conn) == 1
    tier, ev = conn.execute(
        "SELECT binding_tier, binding_evidence FROM platform_identity "
        "WHERE artist_id = %s AND platform = 'bandcamp'", (a,),
    ).fetchone()
    assert tier == "C"
    assert ev["method"] == "admin_review"
    # idempotent: resolved_at stamped → second pass applies nothing
    assert apply_approved_bindings(conn) == 0


def test_review_poller_skips_pending_and_rejected(conn):
    from pipeline.review_poller import apply_approved_bindings

    a = _artist(conn, name="ZZ Poller2", mbid="00000000-feed-4bad-9bad-000000000891")
    for status in ("pending", "rejected"):
        conn.execute(
            "INSERT INTO review_item (kind, subject_type, subject_id, reason, evidence, status) "
            "VALUES ('source_binding', 'artist', %s, 'x', '{}', %s)", (a, status),
        )
    # rejected rows without decisions: poller only touches APPROVED
    assert apply_approved_bindings(conn) == 0
    assert conn.execute(
        "SELECT count(*) FROM platform_identity WHERE artist_id = %s", (a,)
    ).fetchone()[0] == 0
