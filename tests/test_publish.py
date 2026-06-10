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
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count) "
        "VALUES (%s, 'mock-model', 2, '[0.6,0.8]', 4)", (a,),
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
    row = conn.execute("SELECT name, slug, tags, audio_embedding, bandcamp_url "
                       "FROM artists WHERE mbid = %s", (MBID,)).fetchone()
    assert row[0] == "Püblish Fixture"
    assert row[1] == "publish-fixture"
    tags = row[2] if isinstance(row[2], dict) else json.loads(row[2])
    assert "zz-pub-genre" in tags and all(isinstance(v, int) and v >= 1 for v in tags.values())
    assert row[3] == "[0.6,0.8]"
    assert row[4] == "https://zz-pub-band.bandcamp.com"

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
