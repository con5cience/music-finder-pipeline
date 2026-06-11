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
    """Hourly-sync mode: only changed-since-watermark artists publish; a
    banned artist is excluded AND pruned from the serving side."""
    from pipeline.publish import publish_incremental

    _serving_schema(conn)
    conn.execute("UPDATE publish_watermark SET last_run = 'epoch' WHERE id = 'default'")  # hermetic
    a1 = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES "
        "('Inc One', '00000000-feed-4bad-9bad-000000000aa1', 'deezer') RETURNING id").fetchone()[0]
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s, 'mock-model', 2, '[1,0]', 3, 1.0)", (a1,))
    assert publish_incremental(conn, conn) == 1     # first run: everything is new
    assert publish_incremental(conn, conn) == 0     # nothing changed since watermark
    # a new artist arrives → only IT publishes
    a2 = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES "
        "('Inc Two', '00000000-feed-4bad-9bad-000000000aa2', 'deezer') RETURNING id").fetchone()[0]
    conn.execute(  # clock_timestamp: in-txn now() predates any wall-clock watermark
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio, computed_at) "
        "VALUES (%s, 'mock-model', 2, '[0,1]', 3, 1.0, clock_timestamp())", (a2,))
    assert publish_incremental(conn, conn) == 1
    # ban a1 → next sync prunes the serving row and never re-publishes
    conn.execute(
        "INSERT INTO ban_ledger (artist_id, mbid, display_name) VALUES (%s, %s, 'Inc One')",
        (a1, "00000000-feed-4bad-9bad-000000000aa1"))
    conn.execute("UPDATE artist_embedding SET computed_at = clock_timestamp() WHERE artist_id = %s", (a1,))
    assert publish_incremental(conn, conn) == 0     # banned: excluded despite the change
    assert conn.execute("SELECT count(*) FROM artists WHERE name = 'Inc One'").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM artists WHERE name = 'Inc Two'").fetchone()[0] == 1
