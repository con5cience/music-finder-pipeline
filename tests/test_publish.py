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
            tag_vector text,
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


def _embedded_artist(conn, i: int) -> str:
    """A minimal publishable artist (distinct mbid-NULL identity, has an
    embedding) — _factory_artist reuses one fixed MBID, so it can't make many."""
    a = conn.execute(
        "INSERT INTO artist (display_name, embedding_source) VALUES (%s, 'bandcamp') RETURNING id",
        (f"Rep Fixture {i}",),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s, 'mock-model', 2, '[0.6,0.8]', 4, 0.67)", (a,),
    )
    return str(a)


def test_republish_all_upserts_every_artist_in_batches(conn):
    from pipeline.publish import republish_all
    _serving_schema(conn)
    for i in range(5):
        _embedded_artist(conn, i)
    seen = []
    # commit_each=False keeps the rolled-back fixture isolated; we still exercise
    # the keyset loop + per-batch publish.
    n = republish_all(conn, conn, batch=2, commit_each=False, progress=lambda t, _a: seen.append(t))
    assert n == 5
    assert conn.execute("SELECT count(*) FROM artists").fetchone()[0] == 5
    assert seen == [2, 4, 5]  # batches of 2,2,1; cumulative progress


def test_republish_all_idempotent(conn):
    from pipeline.publish import republish_all
    _serving_schema(conn)
    for i in range(3):
        _embedded_artist(conn, i)
    republish_all(conn, conn, batch=10, commit_each=False)
    republish_all(conn, conn, batch=10, commit_each=False)  # re-run must not duplicate
    assert conn.execute("SELECT count(*) FROM artists").fetchone()[0] == 3


def test_republish_all_resumes_from_after(conn):
    from pipeline.publish import publishable_artists, republish_all
    _serving_schema(conn)
    for i in range(4):
        _embedded_artist(conn, i)
    page1 = publishable_artists(conn, 2, since=None)  # first 2 in keyset (a.id) order
    n = republish_all(conn, conn, batch=10, commit_each=False, start_after=page1[-1][0])
    assert n == 2  # only the artists with id > the 2nd id
    assert conn.execute("SELECT count(*) FROM artists").fetchone()[0] == 2


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
    # 3-tier (ADR-022), never empty: this fixture has no MB genres and no
    # bc_candidate tags, so it falls to the AUDIO tier — track-aggregation
    # fallback over its track_tag_scores. zz-pub-genre (0.9) is kept; zz-pub-weak
    # (0.1) is below the corpus mean and dropped. MB precedence + Bandcamp tier
    # are covered by test_publish_prefers_mb_genres_over_audio / *_bandcamp_*.
    tags = row[2] if isinstance(row[2], dict) else (json.loads(row[2]) if row[2] else {})
    assert "zz-pub-genre" in tags
    assert "zz-pub-weak" not in tags
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
    # filler corpus so N (distinct tagged artists) > df → the P2 idf is positive
    for i in range(2):
        fi = conn.execute("INSERT INTO artist (display_name) VALUES (%s) RETURNING id", (f"cu-f{i}",)).fetchone()[0]
        conn.execute(
            "INSERT INTO artist_tag_scores (artist_id, tag, score, model) VALUES (%s,'cu-filler',0.4,'muq-mulan-large')",
            (fi,),
        )
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
        "('cal-units', 'muq-mulan-large', 0.49, 0.05, 1), "
        "('cal-units', 'muq-mulan-large#artist', 0.80, 0.05, 1)"
    )
    assert "cal-units" not in artist_tags(conn, a)  # judged against the artist mean (0.80) → z<0


def test_artist_tags_idf_and_gate_suppress_a_magnet(conn):
    """ADR-020 P2+P3: with EQUAL raw z, the over-assigned tag (low idf=ln(N/df)) is
    demoted AND falls below the per-artist relative gate → dropped, while the rare
    well-matched tag is kept."""
    f = [
        conn.execute("INSERT INTO artist (display_name) VALUES (%s) RETURNING id", (f"idf-f{i}",)).fetchone()[0]
        for i in range(3)
    ]
    conn.execute(
        "INSERT INTO artist_tag_scores (artist_id, tag, score, model) VALUES "
        "(%s,'magnet-tag',0.5,'muq-mulan-large'),(%s,'magnet-tag',0.5,'muq-mulan-large'),"
        "(%s,'neutral-tag',0.5,'muq-mulan-large')",
        (f[0], f[1], f[2]),
    )
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) "
        "VALUES ('IDF Fixture', %s, 'bandcamp') RETURNING id",
        ("00000000-feed-4bad-9bad-0000000ca103",),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO artist_tag_scores (artist_id, tag, score, model) VALUES "
        "(%s,'magnet-tag',0.70,'muq-mulan-large'),(%s,'rare-tag',0.70,'muq-mulan-large')",
        (a, a),
    )
    # equal mean/sd → equal raw z; df differs (magnet on 3 of 4 artists, rare on 1)
    conn.execute(
        "INSERT INTO tag_calibration (tag, model, mean, stddev, n) VALUES "
        "('magnet-tag','muq-mulan-large#artist',0.45,0.10,3),"
        "('rare-tag','muq-mulan-large#artist',0.45,0.10,1)"
    )
    tags = artist_tags(conn, a)
    assert "rare-tag" in tags  # rare, well-matched → kept
    assert "magnet-tag" not in tags  # demoted by idf, then gated out by P3


def test_artist_tags_batch_matches_per_artist(conn):
    """The set-based fast path must be byte-identical to the per-artist function
    across all three cases: primary (z/idf/gate kept), all-gated (-> {}), and the
    no-scores fallback (track aggregation)."""
    from pipeline.publish import artist_tags, artist_tags_batch

    # df: magnet on 2 fillers (so it's over-assigned), rare on 1
    fillers = [
        conn.execute("INSERT INTO artist (display_name) VALUES (%s) RETURNING id", (f"eqf{i}",)).fetchone()[0]
        for i in range(2)
    ]
    conn.execute(
        "INSERT INTO artist_tag_scores (artist_id, tag, score, model) VALUES "
        "(%s,'magnet-tag',0.5,'muq-mulan-large'),(%s,'magnet-tag',0.5,'muq-mulan-large')",
        (fillers[0], fillers[1]),
    )
    # A: magnet (gated) + rare (kept)
    A = conn.execute("INSERT INTO artist (display_name) VALUES ('eqA') RETURNING id").fetchone()[0]
    conn.execute(
        "INSERT INTO artist_tag_scores (artist_id, tag, score, model) VALUES "
        "(%s,'magnet-tag',0.70,'muq-mulan-large'),(%s,'rare-tag',0.70,'muq-mulan-large')", (A, A),
    )
    # B: a single BELOW-mean tag → z<0 → z_adj<0 → dropped → {}
    B = conn.execute("INSERT INTO artist (display_name) VALUES ('eqB') RETURNING id").fetchone()[0]
    conn.execute(
        "INSERT INTO artist_tag_scores (artist_id, tag, score, model) VALUES "
        "(%s,'magnet-tag',0.30,'muq-mulan-large')", (B,),
    )
    # C: NO artist_tag_scores → must hit the per-artist track-aggregation fallback
    C = conn.execute("INSERT INTO artist (display_name) VALUES ('eqC') RETURNING id").fetchone()[0]
    tc = conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, binding_tier, verification_status) "
        "VALUES (%s,'deezer','eq-c-t1','A','verified') RETURNING id", (C,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO track_tag_scores (track_id, tag, score, model) VALUES (%s,'fallback-tag',0.8,'muq-mulan-large')",
        (tc,),
    )
    # a low filler track score so the corpus track-mean sits below C's tag —
    # otherwise a lone track_tag_score equals the mean (z=0) and gets gated.
    tf = conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, binding_tier, verification_status) "
        "VALUES (%s,'deezer','eq-f-t1','A','verified') RETURNING id", (fillers[0],),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO track_tag_scores (track_id, tag, score, model) VALUES (%s,'low-tag',0.1,'muq-mulan-large')",
        (tf,),
    )
    # D: a SECOND no-artist_tag_scores artist → exercises the batched fallback
    # partitioning across >1 artist (must not bleed C's tag into D).
    D = conn.execute("INSERT INTO artist (display_name) VALUES ('eqD') RETURNING id").fetchone()[0]
    td = conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, binding_tier, verification_status) "
        "VALUES (%s,'deezer','eq-d-t1','A','verified') RETURNING id", (D,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO track_tag_scores (track_id, tag, score, model) VALUES (%s,'d-tag',0.7,'muq-mulan-large')",
        (td,),
    )
    conn.execute(
        "INSERT INTO tag_calibration (tag, model, mean, stddev, n) VALUES "
        "('magnet-tag','muq-mulan-large#artist',0.45,0.10,3),"
        "('rare-tag','muq-mulan-large#artist',0.45,0.10,1)"
    )
    g = conn.execute(
        "SELECT avg(score), greatest(stddev(score),1e-6), count(DISTINCT artist_id) FROM artist_tag_scores"
    ).fetchone()

    batch = artist_tags_batch(conn, [A, B, C, D], g)
    for aid in (A, B, C, D):
        assert batch.get(str(aid), {}) == artist_tags(conn, aid, g_moments=g), f"batch != per-artist for {aid}"
    # and the substance held: A keeps rare not magnet, B empty, C+D fell back to
    # their OWN track tags (no cross-artist bleed)
    assert "rare-tag" in batch[str(A)] and "magnet-tag" not in batch[str(A)]
    assert batch.get(str(B), {}) == {}
    assert "fallback-tag" in batch[str(C)] and "d-tag" not in batch[str(C)]
    assert "d-tag" in batch[str(D)] and "fallback-tag" not in batch[str(D)]


def test_artist_urls_batch_matches_per_artist(conn):
    from pipeline.publish import artist_urls, artist_urls_batch
    a = _factory_artist(conn)  # has bandcamp + deezer platform_identity
    assert artist_urls_batch(conn, [a]).get(str(a), {}) == artist_urls(conn, a)
    assert artist_urls(conn, a)  # non-empty, so this actually proves equivalence


def test_centered_demotes_dominant_direction_tags(conn):
    """ADR-020 P5: a higher-raw-score tag that is aligned with the dominant audio
    direction (high d) is demoted below a lower-raw-score niche tag (low d), and
    gated out — the core of the centering fix."""
    from pipeline.publish import artist_tags_batch
    a = conn.execute("INSERT INTO artist (display_name) VALUES ('cen-a') RETURNING id").fetchone()[0]
    conn.execute(
        "INSERT INTO artist_tag_scores (artist_id, tag, score, model) VALUES "
        "(%s,'dominant-tag',0.55,'muq-mulan-large'),(%s,'niche-tag',0.50,'muq-mulan-large')", (a, a),
    )
    conn.execute(
        "INSERT INTO tag_centering (tag, model, d, mu_version, n_sample) VALUES "
        "('dominant-tag','muq-mulan-large',0.80,'v1',100),"
        "('niche-tag','muq-mulan-large',0.05,'v1',100)"
    )
    # tag_centering present -> artist_tags_batch dispatches to the centered ranking
    tags = artist_tags_batch(conn, [a], (0.4, 0.1, 4)).get(str(a), {})
    assert "niche-tag" in tags          # lower raw score, but survives centering
    assert "dominant-tag" not in tags   # higher raw score, but demoted + gated by centering


# (removed test_publish_uses_centering_when_tag_centering_present — ADR-022:
# publish is MB-only now, audio/centering is no longer in the publish path.)


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


def test_mb_genres_batch_editorial_genres(conn):
    """MB editorial genres: direct genre kept, alias merged to canonical,
    non-genre tag dropped, votes summed, all from mb_raw."""
    from pipeline.publish import mb_genres_batch
    conn.execute(
        "INSERT INTO mb_raw.genre (id, gid, name) VALUES "
        "(970001,'00000000-feed-4bad-9bad-00000000e001','zz-metal'),"
        "(970002,'00000000-feed-4bad-9bad-00000000e002','zz-wave')"
    )
    conn.execute(
        "INSERT INTO mb_raw.genre_alias (id, genre, name, sort_name) VALUES "
        "(970101, 970002, 'zz wave', 'zz wave')"
    )
    conn.execute(
        "INSERT INTO mb_raw.tag (id, name, ref_count) VALUES "
        "(980001,'zz-metal',1),(980002,'zz wave',1),(980003,'canadian',1)"
    )
    mbid = '00000000-feed-4bad-9bad-00000000a701'
    conn.execute(
        "INSERT INTO mb_raw.artist (id, gid, name, sort_name) VALUES (990701, %s, 'MBG', 'MBG')",
        (mbid,),
    )
    conn.execute(
        "INSERT INTO mb_raw.artist_tag (artist, tag, count) VALUES "
        "(990701,980001,5),(990701,980002,3),(990701,980003,9)"
    )
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('MBG', %s) RETURNING id", (mbid,)
    ).fetchone()[0]
    tags = mb_genres_batch(conn, [a]).get(str(a), {})
    assert 'zz-metal' in tags          # direct canonical genre
    assert 'zz-wave' in tags           # alias 'zz wave' merged to canonical
    assert 'canadian' not in tags      # non-genre dropped despite high vote count


def test_publish_prefers_mb_genres_over_audio(conn):
    """MB genres win when present; audio tags only fill in where MB has none."""
    from pipeline.publish import publish_rows, publishable_artists
    _serving_schema(conn)
    conn.execute(
        "INSERT INTO mb_raw.genre (id, gid, name) VALUES "
        "(971001,'00000000-feed-4bad-9bad-00000000e101','zz-mbgenre')"
    )
    conn.execute("INSERT INTO mb_raw.tag (id, name, ref_count) VALUES (981001,'zz-mbgenre',1)")
    mbid = '00000000-feed-4bad-9bad-00000000a801'
    conn.execute(
        "INSERT INTO mb_raw.artist (id, gid, name, sort_name) VALUES (991001, %s, 'HasMB', 'HasMB')",
        (mbid,),
    )
    conn.execute("INSERT INTO mb_raw.artist_tag (artist, tag, count) VALUES (991001,981001,4)")
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES ('HasMB', %s, 'bandcamp') RETURNING id",
        (mbid,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s,'mock-model',2,'[0.6,0.8]',4,0.67)", (a,),
    )
    # an audio tag that should be SUPPRESSED in favor of the MB genre
    conn.execute(
        "INSERT INTO artist_tag_scores (artist_id, tag, score, model) VALUES (%s,'zz-audiotag',0.9,'muq-mulan-large')",
        (a,),
    )
    publish_rows(conn, conn, publishable_artists(conn, 10))
    # MB-keyed artists get a fresh serving id (ON CONFLICT mbid) — query by mbid
    tags = conn.execute("SELECT tags FROM artists WHERE mbid=%s", (mbid,)).fetchone()[0]
    assert 'zz-mbgenre' in tags and 'zz-audiotag' not in tags


def test_audio_tier_knn_borrows_neighbor_genres(conn):
    """ADR-025 audio tier: an artist with no MB/Bandcamp tags borrows the MB
    genres of its nearest-SOUNDING MB-labeled anchor, not a distant one."""
    from pipeline.publish import _artist_tags_knn, load_anchor_genres

    def vec(i):  # 1024-dim unit vector pointing along axis i
        return "[" + ",".join("1" if k == i else "0" for k in range(1024)) + "]"

    # two MB-labeled anchors with different genres + orthogonal embeddings
    conn.execute(
        "INSERT INTO mb_raw.genre (id, gid, name) VALUES "
        "(972001,'00000000-feed-4bad-9bad-00000000e201','zz-knn-near'),"
        "(972002,'00000000-feed-4bad-9bad-00000000e202','zz-knn-far')"
    )
    conn.execute("INSERT INTO mb_raw.tag (id, name, ref_count) VALUES (982001,'zz-knn-near',1),(982002,'zz-knn-far',1)")
    near_mb = '00000000-feed-4bad-9bad-00000000a901'
    far_mb = '00000000-feed-4bad-9bad-00000000a902'
    conn.execute("INSERT INTO mb_raw.artist (id, gid, name, sort_name) VALUES (992001,%s,'NEAR','NEAR'),(992002,%s,'FAR','FAR')", (near_mb, far_mb))
    conn.execute("INSERT INTO mb_raw.artist_tag (artist, tag, count) VALUES (992001,982001,5),(992002,982002,5)")
    near = conn.execute("INSERT INTO artist (display_name, mbid, embedding_source) VALUES ('NEAR',%s,'bandcamp') RETURNING id", (near_mb,)).fetchone()[0]
    far = conn.execute("INSERT INTO artist (display_name, mbid, embedding_source) VALUES ('FAR',%s,'bandcamp') RETURNING id", (far_mb,)).fetchone()[0]
    target = conn.execute("INSERT INTO artist (display_name, embedding_source) VALUES ('TARGET','bandcamp') RETURNING id").fetchone()[0]
    for aid, v in ((near, vec(0)), (far, vec(1)), (target, vec(0))):  # target aligns with NEAR (axis 0)
        conn.execute("INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) VALUES (%s,'muq-large-msd',1024,%s,4,0.9)", (aid, v))

    anchors = load_anchor_genres(conn)
    assert str(near) in anchors and str(target) not in anchors  # only MB-labeled artists are anchors
    g = conn.execute("SELECT avg(score),greatest(stddev(score),1e-6),count(DISTINCT artist_id) FROM artist_tag_scores").fetchone()
    res = _artist_tags_knn(conn, [target], g, anchors).get(str(target), {})
    assert 'zz-knn-near' in res       # borrowed from the sonically-nearest anchor
    assert 'zz-knn-far' not in res    # the orthogonal anchor is gated out


def test_publish_uses_bandcamp_tags_when_no_mb_genres(conn):
    """Middle tier: artist with no MB genres but human Bandcamp tags gets those
    tags (verbatim) — NOT the audio tier — until MB later clobbers them."""
    from pipeline.publish import publish_rows, publishable_artists
    _serving_schema(conn)
    a = conn.execute(
        "INSERT INTO artist (display_name, embedding_source) VALUES ('BCOnly', 'bandcamp') RETURNING id"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s,'mock-model',2,'[0.6,0.8]',4,0.67)", (a,),
    )
    conn.execute(
        "INSERT INTO bc_candidate (platform_id, band_name, band_url, tags, status, artist_id) "
        "VALUES ('zz-bc-only','BCOnly','https://x','{\"zz-shoegaze\",\"zz-dream pop\"}','admitted',%s)",
        (a,),
    )
    # BC tier is allowlist-gated to tag_approved (genre-only policy) — approve them.
    conn.execute("INSERT INTO tag_approved (tag, source) VALUES ('zz-shoegaze','human'),('zz-dream pop','human')")
    # an audio tag that must be IGNORED because the Bandcamp tier wins
    conn.execute(
        "INSERT INTO artist_tag_scores (artist_id, tag, score, model) VALUES (%s,'zz-audiotag',0.9,'muq-mulan-large')",
        (a,),
    )
    publish_rows(conn, conn, publishable_artists(conn, 10))
    tags = conn.execute("SELECT tags FROM artists WHERE id=%s", (a,)).fetchone()[0]
    assert 'zz-shoegaze' in tags and 'zz-dream pop' in tags
    assert 'zz-audiotag' not in tags


def test_merge_human_tiers_unions_mb_and_bandcamp():
    """Pure: MB + Bandcamp tiers UNION; a tag in BOTH is corroborated (weights
    add → it leads); MB-only and BC-only tags are all kept. Empty tier is a no-op."""
    from pipeline.publish import merge_human_tiers

    out = merge_human_tiers({'goth': 3, 'shoegaze': 2}, {'goth': 1, 'post-punk': 1, 'ethereal': 1})
    assert out['goth'] == 4  # 3 (MB) + 1 (BC overlap) — corroborated, leads
    assert out['shoegaze'] == 2  # MB-only kept
    assert out['post-punk'] == 1 and out['ethereal'] == 1  # BC-only kept
    assert merge_human_tiers({}, {'a': 1}) == {'a': 1}
    assert merge_human_tiers({'a': 2}, {}) == {'a': 2}


def test_merge_human_tiers_caps_at_tag_k():
    from pipeline.publish import TAG_K, merge_human_tiers

    out = merge_human_tiers({'lead': 99}, {f'tag{i}': 1 for i in range(TAG_K + 5)})
    assert len(out) == TAG_K
    assert out['lead'] == 99  # highest weight survives the cap


def test_publish_unions_mb_genres_and_bandcamp_tags(conn):
    """Regression (autumn-us): an artist with BOTH MB genres AND Bandcamp human
    tags publishes the UNION, not just the MB tier (the old cascade dropped the
    Bandcamp folksonomy)."""
    from pipeline.publish import publish_rows, publishable_artists

    _serving_schema(conn)
    conn.execute(
        "INSERT INTO mb_raw.genre (id, gid, name) VALUES "
        "(971002,'00000000-feed-4bad-9bad-00000000e102','zz-mbgenre2')"
    )
    conn.execute("INSERT INTO mb_raw.tag (id, name, ref_count) VALUES (981002,'zz-mbgenre2',1)")
    mbid = '00000000-feed-4bad-9bad-00000000a802'
    conn.execute("INSERT INTO mb_raw.artist (id, gid, name, sort_name) VALUES (991002, %s, 'Both', 'Both')", (mbid,))
    conn.execute("INSERT INTO mb_raw.artist_tag (artist, tag, count) VALUES (991002,981002,4)")
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES ('Both', %s, 'bandcamp') RETURNING id",
        (mbid,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s,'mock-model',2,'[0.6,0.8]',4,0.67)", (a,),
    )
    conn.execute(
        "INSERT INTO bc_candidate (platform_id, band_name, band_url, tags, status, artist_id) "
        "VALUES ('zz-both','Both','https://x','{\"zz-bctag1\",\"zz-bctag2\"}','admitted',%s)",
        (a,),
    )
    # BC tier is allowlist-gated — approve the two Bandcamp tags so the union holds.
    conn.execute("INSERT INTO tag_approved (tag, source) VALUES ('zz-bctag1','human'),('zz-bctag2','human')")
    publish_rows(conn, conn, publishable_artists(conn, 10))
    tags = conn.execute("SELECT tags FROM artists WHERE mbid=%s", (mbid,)).fetchone()[0]
    assert 'zz-mbgenre2' in tags  # MB tier kept
    assert 'zz-bctag1' in tags and 'zz-bctag2' in tags  # Bandcamp tier ALSO kept (the fix)


def test_bandcamp_tier_allowlist_drops_unapproved_and_falls_through_to_audio(conn):
    """Genre-only allowlist: a Bandcamp tag NOT in tag_approved is dropped. An
    artist whose BC tags are ALL unapproved must fall through to the audio tier
    (not publish an empty Bandcamp tier). MB editorial tier is never gated."""
    from pipeline.publish import publish_rows, publishable_artists

    _serving_schema(conn)
    # (1) BC artist with one APPROVED + one UNDECIDED tag → only approved survives.
    a1 = conn.execute(
        "INSERT INTO artist (display_name, embedding_source) VALUES ('Mixed', 'bandcamp') RETURNING id"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
        "VALUES (%s,'mock-model',2,'[0.6,0.8]',4,0.67)", (a1,),
    )
    conn.execute(
        "INSERT INTO bc_candidate (platform_id, band_name, band_url, tags, status, artist_id) "
        "VALUES ('zz-mixed','Mixed','https://x','{\"zz-goodgenre\",\"zz-junktag\"}','admitted',%s)",
        (a1,),
    )
    conn.execute("INSERT INTO tag_approved (tag, source) VALUES ('zz-goodgenre','human')")
    # (2) BC artist whose ONLY tag is unapproved → BC tier empty → audio takes over.
    # Audio tier (ADR-025 kNN) borrows the MB genre of the sonically-nearest
    # anchor, so set up a real 1024-dim anchor (EMB_MODEL) for a2 to align with.
    def vec(i):  # 1024-dim unit vector along axis i
        return "[" + ",".join("1" if k == i else "0" for k in range(1024)) + "]"

    conn.execute(
        "INSERT INTO mb_raw.genre (id, gid, name) VALUES "
        "(973001,'00000000-feed-4bad-9bad-00000000e301','zz-audioborrow')"
    )
    conn.execute("INSERT INTO mb_raw.tag (id, name, ref_count) VALUES (983001,'zz-audioborrow',1)")
    anc_mb = '00000000-feed-4bad-9bad-00000000ab01'
    conn.execute("INSERT INTO mb_raw.artist (id, gid, name, sort_name) VALUES (993001,%s,'ANC','ANC')", (anc_mb,))
    conn.execute("INSERT INTO mb_raw.artist_tag (artist, tag, count) VALUES (993001,983001,5)")
    anc = conn.execute(
        "INSERT INTO artist (display_name, mbid, embedding_source) VALUES ('ANC',%s,'bandcamp') RETURNING id", (anc_mb,)
    ).fetchone()[0]
    a2 = conn.execute(
        "INSERT INTO artist (display_name, embedding_source) VALUES ('AllJunk', 'bandcamp') RETURNING id"
    ).fetchone()[0]
    for aid, v in ((anc, vec(0)), (a2, vec(0))):  # a2 aligns sonically with the anchor
        conn.execute(
            "INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio) "
            "VALUES (%s,%s,1024,%s,4,0.9)", (aid, 'muq-large-msd', v),
        )
    conn.execute(
        "INSERT INTO bc_candidate (platform_id, band_name, band_url, tags, status, artist_id) "
        "VALUES ('zz-alljunk','AllJunk','https://x','{\"zz-onlyjunk\"}','admitted',%s)",
        (a2,),
    )
    publish_rows(conn, conn, publishable_artists(conn, 10))
    t1 = conn.execute("SELECT tags FROM artists WHERE id=%s", (a1,)).fetchone()[0]
    assert 'zz-goodgenre' in t1 and 'zz-junktag' not in t1  # unapproved BC tag dropped
    t2 = conn.execute("SELECT tags FROM artists WHERE id=%s", (a2,)).fetchone()[0]
    assert 'zz-onlyjunk' not in t2  # all-unapproved BC tier discarded
    assert 'zz-audioborrow' in t2  # fell through to the audio tier (non-empty)
