"""Tag auto-classifier (genre-only policy). Heuristics are pure; run_classify is
exercised against the migrated test DB via the `conn` fixture. The LLM lane is
monkeypatched (no network)."""

from __future__ import annotations

import pipeline.classify_tags as ct
from pipeline.classify_tags import heuristic_category, run_classify


def test_heuristic_category_blocks_clear_non_genres():
    assert heuristic_category("united kingdom") == "location"
    assert heuristic_category("oakland") == "location"
    assert heuristic_category("piano") == "instrument"
    assert heuristic_category("female vocals") == "meta"
    assert heuristic_category("atmospheric") == "meta"
    assert heuristic_category("sett records") == "label"
    assert heuristic_category("2019") == "meta"  # numeric/year


def test_heuristic_category_leaves_real_genres_undecided():
    # real genres / long-tail subgenres must NOT be heuristically blocked —
    # they're decided by MB-vocab or the LLM.
    for g in ("post-punk", "doom", "dungeon synth", "coldwave", "shoegaze", "ambient"):
        assert heuristic_category(g) is None


def test_heuristic_category_blocks_eras_moods_and_city_states():
    # eras (bare decades + decade-qualified compounds + word decades)
    for era in ("00s", "90s", "90's", "1990s", "2000s", "90's hip hop", "80s pop", "eighties", "nineties"):
        assert heuristic_category(era) == "era", era
    # moods
    for mood in ("depressing", "erotic", "exotic", "easy", "sexy", "nostalgic"):
        assert heuristic_category(mood) == "mood", mood
    # "<city> ST" + a bare foreign city we added
    assert heuristic_category("denver co") == "location"
    assert heuristic_category("austin tx") == "location"
    assert heuristic_category("toulouse") == "location"


def test_new_heuristics_dont_false_positive_on_genres():
    # decade-looking-but-genre, state-code-suffix-but-genre, mood-prefix-but-genre
    for g in ("8-bit", "8 bit", "100 gecs", "2 step", "easy listening", "chicago house",
              "detroit techno", "uk garage", "hip hop"):
        assert heuristic_category(g) is None, g


def test_dry_run_writes_nothing_but_reports_buckets(conn):
    conn.execute(
        "INSERT INTO mb_raw.genre (id, gid, name) "
        "VALUES (972003,'00000000-0000-4000-8000-0000000c1a03','zz-realgenre')"
    )
    _seed_bc_tags(conn, ["zz-realgenre", "denver co", "00s"])
    before_ap = conn.execute("SELECT count(*) FROM tag_approved").fetchone()[0]
    before_bl = conn.execute("SELECT count(*) FROM tag_manual_blocklist").fetchone()[0]

    samples: dict[str, list[str]] = {}
    counts = run_classify(conn, use_llm=False, dry_run=True, samples=samples)

    # nothing written
    assert conn.execute("SELECT count(*) FROM tag_approved").fetchone()[0] == before_ap
    assert conn.execute("SELECT count(*) FROM tag_manual_blocklist").fetchone()[0] == before_bl
    # but the buckets are reported + sampled
    assert counts["genre_mb"] == 1 and counts["block_heuristic"] == 2
    assert "zz-realgenre" in samples.get("genre", [])
    assert "denver co" in samples.get("location", []) and "00s" in samples.get("era", [])


def _seed_bc_tags(conn, tags: list[str]) -> None:
    aid = conn.execute(
        "INSERT INTO artist (display_name, embedding_source) VALUES ('ClsTest','bandcamp') RETURNING id"
    ).fetchone()[0]
    arr = "{" + ",".join('"' + t + '"' for t in tags) + "}"
    conn.execute(
        "INSERT INTO bc_candidate (platform_id, band_name, band_url, tags, status, artist_id) "
        "VALUES ('cls-1','ClsTest','https://x',%s,'admitted',%s)",
        (arr, aid),
    )
    # plain REFRESH (non-CONCURRENT) is transaction-safe and sees this tx's rows
    conn.execute("REFRESH MATERIALIZED VIEW tag_review_freq")


def test_run_classify_mb_vocab_and_heuristics(conn):
    conn.execute(
        "INSERT INTO mb_raw.genre (id, gid, name) "
        "VALUES (972001,'00000000-0000-4000-8000-0000000c1a01','zz-realgenre')"
    )
    _seed_bc_tags(conn, ["zz-realgenre", "united kingdom", "zz-unknownsub"])

    counts = run_classify(conn, use_llm=False)

    approved = {r[0] for r in conn.execute("SELECT tag FROM tag_approved").fetchall()}
    blocked = {r[0]: r[1] for r in conn.execute("SELECT tag, category FROM tag_manual_blocklist").fetchall()}
    assert "zz-realgenre" in approved  # MB-vocab genre kept
    assert blocked.get("united kingdom") == "location"  # heuristic non-genre blocked
    assert "zz-unknownsub" not in approved and "zz-unknownsub" not in blocked  # residual undecided
    assert counts["genre_mb"] == 1 and counts["block_heuristic"] == 1 and counts["undecided"] == 1
    # source provenance is 'auto'
    src = conn.execute("SELECT source FROM tag_approved WHERE tag='zz-realgenre'").fetchone()[0]
    assert src == "auto"


def test_run_classify_llm_residual(conn, monkeypatch):
    # both tags are residual (not MB vocab, not heuristic) — the LLM decides.
    monkeypatch.setattr(ct, "llm_classify", lambda tags, **_: {"zz-unknownsub": "genre", "zz-junk": "nongenre"})
    _seed_bc_tags(conn, ["zz-unknownsub", "zz-junk"])

    counts = run_classify(conn, use_llm=True)

    approved = {r[0] for r in conn.execute("SELECT tag FROM tag_approved").fetchall()}
    blocked = {r[0] for r in conn.execute("SELECT tag FROM tag_manual_blocklist").fetchall()}
    assert "zz-unknownsub" in approved and "zz-junk" in blocked
    assert counts["genre_llm"] == 1 and counts["block_llm"] == 1


def test_run_classify_never_overwrites_human(conn):
    # a human block must survive a classify run even if MB vocab says genre.
    conn.execute(
        "INSERT INTO mb_raw.genre (id, gid, name) "
        "VALUES (972002,'00000000-0000-4000-8000-0000000c1a02','zz-human')"
    )
    conn.execute("INSERT INTO tag_manual_blocklist (tag, reason, source) VALUES ('zz-human','nope','human')")
    _seed_bc_tags(conn, ["zz-human"])

    run_classify(conn, use_llm=False)

    # still blocked, still human — not moved to approved
    row = conn.execute("SELECT source FROM tag_manual_blocklist WHERE tag='zz-human'").fetchone()
    assert row is not None and row[0] == "human"
    assert conn.execute("SELECT count(*) FROM tag_approved WHERE tag='zz-human'").fetchone()[0] == 0
