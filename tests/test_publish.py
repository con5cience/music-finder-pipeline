"""Publish: factory → serving upsert. Hermetic — a TEMP `artists` table
(serving-schema subset incl. the mbid unique that ON CONFLICT targets) lives
inside the test transaction, so the same conn plays both factory and app."""

from __future__ import annotations

import json

from pipeline.publish import artist_tags, artist_urls, publish_artists, resolve_slug, slug_base

MBID = "00000000-feed-4bad-9bad-000000000aab"


def _serving_schema(conn):
    conn.execute(
        """
        CREATE TEMP TABLE artists (
            id uuid PRIMARY KEY,
            mbid text UNIQUE,
            name text NOT NULL,
            slug text,
            tags jsonb,
            audio_embedding text,
            signal_ratio real,
            embedding_source text,
            perceptual jsonb,
            language text,
            location text,
            audio_embedding_updated timestamptz,
            created_at timestamptz,
            deezer_url text, bandcamp_url text, soundcloud_url text,
            youtube_url text, tidal_url text
        ) ON COMMIT DROP
        """
    )


def _factory_artist(conn) -> str:
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) "
        "VALUES ('Püblish Fixture', %s, 'bandcamp') RETURNING id", (MBID,)
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type) VALUES "
        "(%s, 'bandcamp', 'zz-pub-band', 'artist'), (%s, 'deezer', '990077', 'artist')",
        (a, a),
    )
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s, 'mock-model', 2, '[0.6,0.8]', 4, 0.67)", (a,),
    )
    t = conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
        "binding_tier, verification_status) VALUES (%s,'bandcamp','zz-pub-t1','/x.wav',90,'A','verified') "
        "RETURNING id", (a,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO track_tag_scores (track_id, tag, score, model) VALUES "
        "(%s, 'zz-pub-genre', 0.9, 'muq-mulan-large'), (%s, 'zz-pub-weak', 0.1, 'muq-mulan-large')",
        (t, t),
    )
    return a


def test_slug_base_safe():
    assert slug_base("Püblish Fixture!") == "publish-fixture"
    assert slug_base("") == "artist"


def test_resolve_slug_clean_first_suffix_on_collision(conn):
    _serving_schema(conn)
    # first holder gets the clean slug
    assert resolve_slug(conn, "Porches", "00000000-feed-4bad-9bad-0000000000b1") == "porches"
    conn.execute("INSERT INTO artists (id, mbid, name, slug) VALUES "
                 "(gen_random_uuid(), '00000000-feed-4bad-9bad-0000000000b1', 'Porches', 'porches')")
    # different artist, same name → -2 (the app's 031 convention)
    assert resolve_slug(conn, "Porches", "00000000-feed-4bad-9bad-0000000000b2") == "porches-2"
    # the original keeps its slug on re-publish (idempotent)
    assert resolve_slug(conn, "Porches", "00000000-feed-4bad-9bad-0000000000b1") == "porches"


def test_urls_derived_from_identities(conn):
    a = _factory_artist(conn)
    urls = artist_urls(conn, a)
    assert urls["bandcamp_url"] == "https://zz-pub-band.bandcamp.com"
    assert urls["deezer_url"] == "https://www.deezer.com/artist/990077"


def test_publish_upserts_and_is_idempotent(conn):
    a = _factory_artist(conn)
    _serving_schema(conn)
    assert publish_artists(conn, conn, limit=10_000_000) >= 1
    row = conn.execute("SELECT name, slug, tags, audio_embedding, bandcamp_url, "
                       "signal_ratio, embedding_source FROM artists WHERE mbid = %s", (MBID,)).fetchone()
    assert row[0] == "Püblish Fixture"
    assert row[1] == "publish-fixture"
    tags = row[2] if isinstance(row[2], dict) else json.loads(row[2])
    assert "zz-pub-genre" in tags and all(isinstance(v, int) and v >= 1 for v in tags.values())
    assert row[3] == "[0.6,0.8]"
    assert row[4] == "https://zz-pub-band.bandcamp.com"
    assert abs(row[5] - 0.67) < 1e-6        # signal_ratio carried (9b)
    assert row[6] == "bandcamp"             # embedding_source carried (9b)

    # idempotent re-publish: same row updated, not duplicated
    conn.execute("UPDATE artist SET display_name = 'Renamed Fixture' WHERE id = %s", (a,))
    publish_artists(conn, conn, limit=10_000_000)
    n, name = conn.execute(
        "SELECT count(*), max(name) FROM artists WHERE mbid = %s", (MBID,)
    ).fetchone()
    assert (n, name) == (1, "Renamed Fixture")


def test_tags_use_calibrated_positive_z_only(conn):
    a = _factory_artist(conn)
    tags = artist_tags(conn, a)
    assert "zz-pub-genre" in tags  # 0.9 is far above the corpus mean → positive z
    assert "zz-pub-weak" not in tags  # 0.1 is far below → negative z, excluded


def test_artist_tags_z_score_against_artist_moments_not_track(conn):
    """ADR-020 P1 units fix: the PRIMARY (artist_tag_scores) path must z-score
    against the ARTIST-source moments (model+'#artist'), not the track moments.
    Same tag, LOW track mean (0.49) but HIGH artist mean (0.80): a 0.50 score is
    above the track mean (would publish under the old bug) but below the artist
    mean → must be EXCLUDED."""
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) "
        "VALUES ('Cal Units Fixture', %s, 'bandcamp') RETURNING id",
        ("00000000-feed-4bad-9bad-0000000ca101",),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO artist_tag_scores (artist_id, tag, score, model) "
        "VALUES (%s, 'cal-units', 0.50, 'muq-mulan-large')",
        (a,),
    )
    conn.execute(
        "INSERT INTO tag_calibration (tag, model, mean, stddev, n) VALUES "
        "('cal-units', 'muq-mulan-large', 0.49, 0.05, 100), "
        "('cal-units', 'muq-mulan-large#artist', 0.80, 0.05, 100)"
    )
    assert "cal-units" not in artist_tags(conn, a)  # judged against the artist mean (0.80)


def test_refresh_calibration_writes_both_track_and_artist_moments(conn):
    """ADR-020 P1: refresh writes track-source moments (bare model) AND
    artist-source moments (model+'#artist') from their respective tables,
    distinct when the distributions differ."""
    from pipeline.tag_calibration import ARTIST_SUFFIX, refresh_calibration

    a1 = conn.execute("INSERT INTO artist (display_name) VALUES ('Cal A1') RETURNING id").fetchone()[0]
    a2 = conn.execute("INSERT INTO artist (display_name) VALUES ('Cal A2') RETURNING id").fetchone()[0]
    conn.execute(
        "INSERT INTO artist_tag_scores (artist_id, tag, score, model) VALUES "
        "(%s,'zz-cal',0.70,'muq-mulan-large'),(%s,'zz-cal',0.70,'muq-mulan-large')",
        (a1, a2),
    )
    t1 = conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, binding_tier, verification_status) "
        "VALUES (%s,'deezer','zz-cal-t1','A','verified') RETURNING id",
        (a1,),
    ).fetchone()[0]
    t2 = conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, binding_tier, verification_status) "
        "VALUES (%s,'deezer','zz-cal-t2','A','verified') RETURNING id",
        (a1,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO track_tag_scores (track_id, tag, score, model) VALUES "
        "(%s,'zz-cal',0.30,'muq-mulan-large'),(%s,'zz-cal',0.30,'muq-mulan-large')",
        (t1, t2),
    )
    refresh_calibration(conn)
    track = conn.execute(
        "SELECT mean FROM tag_calibration WHERE tag='zz-cal' AND model='muq-mulan-large'"
    ).fetchone()
    artist = conn.execute(
        "SELECT mean FROM tag_calibration WHERE tag='zz-cal' AND model=%s",
        ("muq-mulan-large" + ARTIST_SUFFIX,),
    ).fetchone()
    assert track and abs(track[0] - 0.30) < 1e-3
    assert artist and abs(artist[0] - 0.70) < 1e-3  # distinct artist-source mean — the units fix


def test_publish_mbid_null_then_loop_close(conn):
    """ADR-019's whole identity lifecycle in one test: provisional publish
    (app row id = factory id), then the mbid attaches (sync), then
    re-publish must UPDATE the same row — not duplicate it (critical
    review finding)."""
    _serving_schema(conn)
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) "
        "VALUES ('Loop Closer', NULL, 'bandcamp') RETURNING id").fetchone()[0]
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s, 'mock-model', 2, '[1,0]', 3, 0.5)", (a,))
    assert publish_artists(conn, conn, limit=10) >= 1
    row1 = conn.execute("SELECT id, slug, mbid FROM artists WHERE name = 'Loop Closer'").fetchone()
    assert str(row1[0]) == str(a) and row1[1] == 'loop-closer' and row1[2] is None
    # the MB loop closes: sync attaches an mbid to the FACTORY row
    new_mbid = '00000000-feed-4bad-9bad-000000000fc9'
    conn.execute("UPDATE artist SET mbid = %s WHERE id = %s", (new_mbid, a))
    assert publish_artists(conn, conn, limit=10) >= 1
    rows = conn.execute("SELECT id, slug, mbid FROM artists WHERE name = 'Loop Closer'").fetchall()
    assert len(rows) == 1, "loop close must not duplicate the app row"
    assert str(rows[0][0]) == str(a)          # same row, same id
    assert rows[0][1] == 'loop-closer'        # slug stable
    assert rows[0][2] == new_mbid             # identity upgraded in place


def test_incremental_publish_watermark_and_bans(conn):
    """Hourly-sync semantics, deterministic by construction: timestamps are
    set EXPLICITLY relative to a hand-placed watermark (no clock races),
    and assertions name WHO publishes."""
    from pipeline.publish import publishable_artists

    _serving_schema(conn)

    def names(since):
        return sorted(r[2] for r in publishable_artists(conn, 100, since=since))

    a1 = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES "
        "('Inc One', '00000000-feed-4bad-9bad-000000000aa1', 'deezer') RETURNING id").fetchone()[0]
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio, computed_at) "
        "VALUES (%s, 'mock-model', 2, '[1,0]', 3, 1.0, '2026-01-01')", (a1,))
    a2 = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES "
        "('Inc Two', '00000000-feed-4bad-9bad-000000000aa2', 'deezer') RETURNING id").fetchone()[0]
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio, computed_at) "
        "VALUES (%s, 'mock-model', 2, '[0,1]', 3, 1.0, '2026-02-01')", (a2,))

    assert names(since="2025-12-01") == ["Inc One", "Inc Two"]  # both new
    assert names(since="2026-01-15") == ["Inc Two"]             # watermark passed a1
    assert names(since="2026-03-01") == []                      # nothing newer

    # a1 changes after the watermark → re-eligible…
    conn.execute("UPDATE artist_embedding SET computed_at = '2026-03-15' WHERE artist_id = %s", (a1,))
    assert names(since="2026-03-01") == ["Inc One"]
    # …unless banned: excluded by artist_id AND by mbid, forever
    conn.execute(
        "INSERT INTO ban_ledger (artist_id, mbid, display_name) VALUES (%s, %s, 'Inc One')",
        (a1, "00000000-feed-4bad-9bad-000000000aa1"))
    assert names(since="2026-03-01") == []
    assert names(since="2025-12-01") == ["Inc Two"]  # full publish also excludes

    # the prune half: a serving row for a banned artist is deleted on sync
    from pipeline.publish import publish_incremental

    conn.execute(
        "INSERT INTO artists (id, mbid, name, slug) VALUES (%s, %s, 'Inc One', 'inc-one')",
        (a1, "00000000-feed-4bad-9bad-000000000aa1"))
    conn.execute("UPDATE publish_watermark SET last_run = '2026-03-01' WHERE id = 'default'")
    publish_incremental(conn, conn)
    assert conn.execute("SELECT count(*) FROM artists WHERE name = 'Inc One'").fetchone()[0] == 0


def test_incremental_drain_advances_keyset(conn):
    """Second review catch: drain-until-empty must ADVANCE — with
    changed-set > limit, iterations page by keyset instead of re-publishing
    the first window forever."""
    from pipeline.publish import publish_incremental

    _serving_schema(conn)
    conn.execute("UPDATE publish_watermark SET last_run = 'epoch' WHERE id = 'default'")
    for i in range(5):
        a = conn.execute(
            "INSERT INTO artist (display_name, mbid, embedding_source) VALUES "
            f"('Drain {i}', '00000000-feed-4bad-9bad-00000000dd{i:02d}', 'deezer') RETURNING id").fetchone()[0]
        conn.execute(
            "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
            "VALUES (%s, 'mock-model', 2, '[1,0]', 3, 1.0)", (a,))
    n = publish_incremental(conn, conn, limit=2)  # 5 changed, window of 2
    assert n == 5  # 3 keyset pages, no infinite loop, full drain
    assert conn.execute("SELECT count(*) FROM artists WHERE name LIKE 'Drain %'").fetchone()[0] == 5


def test_incremental_prunes_rows_whose_embedding_was_reset(conn):
    """The poisoned-well gap (2026-06-12): publish only UPSERTS embedded
    artists and deletes only banned ones — an artist whose embedding is
    RESET (centroid pollution remediation, NaN purges) silently drops out
    of the publishable set while its stale serving row + vector live
    forever. The hourly sync now prunes serving rows whose factory artist
    has no embedding. Rows factory doesn't know are NEVER touched."""
    from pipeline.publish import publish_incremental

    _serving_schema(conn)
    mb_reset = "00000000-feed-4bad-9bad-000000000ab1"
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES "
        "('Reset Case', %s, 'deezer') RETURNING id", (mb_reset,)).fetchone()[0]
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s, 'mock-model', 2, '[1,0]', 3, 1.0)", (a,))
    conn.execute(
        "INSERT INTO artists (id, mbid, name, slug) VALUES (%s, %s, 'Reset Case', 'reset-case')",
        (a, mb_reset))
    # a row factory has never heard of — must survive every prune
    conn.execute(
        "INSERT INTO artists (id, mbid, name, slug) VALUES "
        "('00000000-feed-4bad-9bad-0000000fffff', NULL, 'Foreign Row', 'foreign-row')")
    conn.execute("UPDATE publish_watermark SET last_run = now() WHERE id = 'default'")

    publish_incremental(conn, conn)
    assert conn.execute("SELECT count(*) FROM artists WHERE name='Reset Case'").fetchone()[0] == 1

    # the remediation: embedding reset → next sync removes the stale row
    conn.execute("DELETE FROM artist_embedding WHERE artist_id = %s", (a,))
    conn.execute("UPDATE artist SET embedding_source = NULL WHERE id = %s", (a,))
    publish_incremental(conn, conn)
    assert conn.execute("SELECT count(*) FROM artists WHERE name='Reset Case'").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM artists WHERE name='Foreign Row'").fetchone()[0] == 1


def test_prune_matches_provisional_by_id_when_mbid_null(conn):
    """A provisional (mbid-NULL, ADR-019) artist's serving row is keyed by the
    factory id, so the prune must catch it via the id branch — the OR-free
    rewrite splits id-match from mbid-match, and a mbid-NULL row can only match
    by id."""
    from pipeline.publish import publish_incremental

    _serving_schema(conn)
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES "
        "('Provisional Reset', NULL, 'bandcamp') RETURNING id").fetchone()[0]
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s, 'mock-model', 2, '[1,0]', 3, 1.0)", (a,))
    conn.execute(
        "INSERT INTO artists (id, mbid, name, slug) VALUES (%s, NULL, 'Provisional Reset', 'provisional-reset')",
        (a,))
    conn.execute("UPDATE publish_watermark SET last_run = now() WHERE id = 'default'")

    publish_incremental(conn, conn)
    assert conn.execute("SELECT count(*) FROM artists WHERE name='Provisional Reset'").fetchone()[0] == 1

    # embedding reset → pruned via the id branch (mbid is NULL, so mbid branch can't match)
    conn.execute("DELETE FROM artist_embedding WHERE artist_id = %s", (a,))
    conn.execute("UPDATE artist SET embedding_source = NULL WHERE id = %s", (a,))
    publish_incremental(conn, conn)
    assert conn.execute("SELECT count(*) FROM artists WHERE name='Provisional Reset'").fetchone()[0] == 0
