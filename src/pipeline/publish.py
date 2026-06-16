"""Publish: factory → serving DB (the app's artists table). Local app DB now,
cloud sync later (locked decision).

Publishes EMBEDDED artists only (embedding_source stamped — the cascade's
quality gate). Upsert by mbid; idempotent re-publish refreshes everything we
own. We write ONLY pipeline-owned fields: identity (name/slug/mbid), platform
URLs derived from identities, tags (calibrated z-scores: top K with z > 0,
weight = round(z)+1 so the app's integer-weight convention holds), and the
MuQ centroid (app migration 047 made the column untyped). App-owned fields
(top-tracks caches, image_url, user-facing state) are never touched.

Run:  uv run poe publish -- --limit 1000
Env:  APP_DATABASE_URL (no default — publishing is deliberate)
"""

from __future__ import annotations

import re
import unicodedata

from psycopg import Connection

from pipeline.tag_calibration import ARTIST_SUFFIX

TAG_K = 10

_URL_BUILDERS = {
    "deezer": lambda pid: f"https://www.deezer.com/artist/{pid}",
    "bandcamp": lambda pid: f"https://{pid}.bandcamp.com",
    "soundcloud": lambda pid: f"https://soundcloud.com/{pid}",
    "youtube": lambda pid: f"https://www.youtube.com/channel/{pid}",
    "tidal": lambda pid: f"https://tidal.com/browse/artist/{pid}",
}
_URL_COLUMNS = {p: f"{p}_url" for p in _URL_BUILDERS}


def slug_base(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", folded.casefold()).strip("-") or "artist"


def resolve_slug(app: Connection, name: str, key: str) -> str:
    """The app's own slug convention (its migration 031 indexes exactly this):
    clean base for the first holder, base-2/-3/... on homonym collisions.
    Idempotent: a row that already holds a slug in the family keeps it."""
    base = slug_base(name)
    mine = app.execute(
        "SELECT slug FROM artists WHERE (mbid = %s OR id::text = %s) "
        "AND (slug = %s OR slug ~ ('^' || %s || '-[0-9]+$'))",
        (key, key, base, base),
    ).fetchone()
    if mine:
        return mine[0]  # stable across re-publishes
    taken = {
        r[0] for r in app.execute(
            "SELECT slug FROM artists WHERE (slug = %s OR slug ~ ('^' || %s || '-[0-9]+$')) "
            "AND coalesce(mbid, '') != %s AND id::text != %s",
            (base, base, key, key),
        ).fetchall()
    }
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def publishable_artists(conn: Connection, limit: int, since=None, after_id=None) -> list[tuple]:
    """All embedded artists, or — incremental mode — only those whose
    embedding or artist tags changed since the watermark. Banned artists
    never publish (ban_ledger, the do-not-rediscover law)."""
    since_sql = """
          AND (ae.computed_at >= %(since)s
               OR EXISTS (SELECT 1 FROM artist_tag_scores ats
                          WHERE ats.artist_id = a.id AND ats.computed_at >= %(since)s))
    """ if since is not None else ""
    after_sql = "AND a.id > %(after_id)s" if after_id is not None else ""
    return conn.execute(
        f"""
        SELECT a.id, a.mbid::text, a.display_name, a.embedding_source,
               ae.embedding::text, ae.model, ae.signal_ratio
        FROM artist a
        JOIN artist_embedding ae ON ae.artist_id = a.id
        WHERE a.embedding_source IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM ban_ledger b WHERE b.artist_id = a.id
                          OR (a.mbid IS NOT NULL AND b.mbid = a.mbid))
          -- integrity freezer (2026-06-12): open acoustic-disagreement or
          -- AI-slop flags hold the artist from serving until a human looks
          AND NOT EXISTS (SELECT 1 FROM review_item ri WHERE ri.subject_id = a.id
                          AND ri.reason IN ('source_coherence', 'ai_slop')
                          AND ri.status = 'pending')
          {since_sql}
          {after_sql}
        ORDER BY a.id LIMIT %(limit)s
        """,
        {"limit": limit, "since": since, "after_id": after_id},
    ).fetchall()


def publish_incremental(factory: Connection, app: Connection, limit: int = 100000) -> int:
    """Hourly-sync mode: advance the watermark FIRST (no lost-update window
    — anything changing mid-run lands next hour), publish changed artists,
    and prune app rows for newly banned ones."""
    wm = factory.execute("SELECT last_run FROM publish_watermark WHERE id = 'default'").fetchone()[0]
    # clock_timestamp: wall clock, not txn-frozen now()
    factory.execute("UPDATE publish_watermark SET last_run = clock_timestamp() WHERE id = 'default'")
    # drain-until-empty via KEYSET pagination (second review catch: without
    # advancing after_id, the same first window re-published forever)
    n = 0
    after = None
    while True:
        rows = publishable_artists(factory, limit, since=wm, after_id=after)
        if not rows:
            break
        # continuous slop gate: evaluate first-time/grown catalogs in the
        # same cycle; freshly flagged artists drop out before publishing
        from pipeline.slop_detect import gate_unevaluated

        gated = gate_unevaluated(factory, [r[0] for r in rows])
        # keyset advance comes from the PRE-filter batch: an all-flagged
        # batch leaves rows empty, and rows[-1] would crash the hourly sync
        # (caught by the end-to-end choke test before it ever ran live)
        after = rows[-1][0]
        full_batch = len(rows) == limit
        if gated["flagged"]:
            flagged_now = {r[0] for r in factory.execute(
                """SELECT subject_id FROM review_item WHERE reason='ai_slop'
                   AND status='pending' AND subject_id = ANY(%s)""",
                ([str(r[0]) for r in rows],)).fetchall()}
            rows = [r for r in rows if r[0] not in flagged_now]
        publish_rows(factory, app, rows)
        n += len(rows)
        if not full_batch:
            break
    for (aid, mbid) in factory.execute(
        "SELECT artist_id, mbid::text FROM ban_ledger WHERE banned_at >= %s", (wm,)
    ).fetchall():
        app.execute("DELETE FROM artists WHERE id = %s OR (mbid IS NOT NULL AND mbid = %s)",
                    (str(aid) if aid else "00000000-0000-0000-0000-000000000000", mbid))
    prune_lost_embeddings(factory, app)
    return n


def prune_lost_embeddings(factory: Connection, app: Connection, batch: int = 10000) -> int:
    """Remove serving rows whose factory artist no longer has an embedding.

    The poisoned-well gap (2026-06-12): publish only upserts embedded artists
    and deletes banned ones — a RESET embedding (binding remediation, NaN
    purges) silently stranded its stale row + vector on the serving side
    forever. Conservative by construction: only rows whose artist factory
    POSITIVELY KNOWS (matched by mbid or id) and which have zero embeddings
    are deleted; anything factory can't match is left alone."""
    pruned = 0
    last = None
    while True:
        rows = app.execute(
            "SELECT id::text, mbid FROM artists"
            + (" WHERE id > %s" if last is not None else "")
            + " ORDER BY id LIMIT %s",
            ((last, batch) if last is not None else (batch,)),
        ).fetchall()
        if not rows:
            break
        ids = [r[0] for r in rows]
        mbids = [r[1] for r in rows]
        # Index-friendly anti-join. The old single query joined on
        # (a.mbid = x.mbid::uuid OR a.id::text = x.app_id): the OR across two
        # columns plus the a.id::text cast defeated every index and seq-scanned
        # the whole artist table per batch — ~30 min at 100k+ published, which
        # held the publish transaction's watermark lock long enough that hourly
        # runs overlapped and never committed (the 2026-06-15 sync wedge). Split
        # into two index-driven branches (artist_pkey on id, idx_artist_mbid on
        # mbid) UNIONed by app_id — identical semantics, no seq scan.
        victims = [r[0] for r in factory.execute(
            """
            SELECT x.app_id::text
            FROM unnest(%(ids)s::uuid[]) AS x(app_id)
            JOIN artist a ON a.id = x.app_id
            WHERE NOT EXISTS (SELECT 1 FROM artist_embedding e WHERE e.artist_id = a.id)
            UNION
            SELECT x.app_id::text
            FROM unnest(%(ids)s::uuid[], %(mbids)s::uuid[]) AS x(app_id, mbid)
            JOIN artist a ON a.mbid = x.mbid
            WHERE x.mbid IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM artist_embedding e WHERE e.artist_id = a.id)
            """,
            {"ids": ids, "mbids": mbids},
        ).fetchall()]
        if victims:
            pruned += app.execute(
                "DELETE FROM artists WHERE id = ANY(%s::uuid[])", (victims,)
            ).rowcount
        last = ids[-1]
        if len(rows) < batch:
            break
    return pruned


def artist_language(conn: Connection, artist_id) -> str | None:
    """Majority ASR language over the artist's tracks (wave-3, sparse
    coverage by design — None until the artist has ≥2 agreeing tracks)."""
    row = conn.execute(
        """
        SELECT language, count(*) FROM track_language tl
        JOIN audio_track t ON t.id = tl.track_id
        WHERE t.artist_id = %s AND tl.confidence >= 0.6
        GROUP BY language ORDER BY 2 DESC LIMIT 1
        """,
        (artist_id,),
    ).fetchone()
    return row[0] if row and row[1] >= 2 else None


def artist_location(conn: Connection, artist_id) -> str | None:
    """Discovered artists carry their Bandcamp profile location (the MB
    area hint) — published so the product can show WHERE the underground is."""
    row = conn.execute(
        "SELECT location FROM bc_candidate WHERE artist_id = %s AND location IS NOT NULL",
        (artist_id,),
    ).fetchone()
    return row[0] if row else None


def artist_perceptual(conn: Connection, artist_id) -> dict | None:
    """Wave-2 axis means over the artist's tracks (the scorer's wₚ input)."""
    row = conn.execute(
        """
        SELECT avg(danceability), avg(valence), avg(arousal),
               avg(speechiness), avg(liveness), avg(vocalness)
        FROM track_perceptual tp JOIN audio_track t ON t.id = tp.track_id
        WHERE t.artist_id = %s
        """,
        (artist_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    keys = ("danceability", "valence", "arousal", "speechiness", "liveness", "vocalness")
    return {k: round(float(v), 4) for k, v in zip(keys, row, strict=True)}


def artist_urls(conn: Connection, artist_id) -> dict[str, str]:
    out: dict[str, str] = {}
    for platform, pid in conn.execute(
        "SELECT platform, platform_id FROM platform_identity WHERE artist_id = %s", (artist_id,)
    ).fetchall():
        builder = _URL_BUILDERS.get(platform)
        if builder and platform not in out:
            out[_URL_COLUMNS[platform]] = builder(pid)
    return out


def artist_tags(conn: Connection, artist_id, g_moments=None) -> dict[str, int]:
    """Calibrated artist-level tags. PRIMARY: artist_tag_scores (scored from
    the artist-mean MuLan vector at embed time — full resolution, no
    per-track-truncation pathology), z-ranked against per-tag corpus moments.
    FALLBACK until the v2 sweep covers an artist: the per-track aggregation
    (coverage-weighted, known-noisy on preview sources)."""
    if g_moments is None:
        g_moments = conn.execute(
            "SELECT avg(score), greatest(stddev(score), 1e-6) FROM artist_tag_scores"
        ).fetchone()
    gmean, gsd = (g_moments[0] or 0.0), (g_moments[1] or 1e-6)
    primary = conn.execute(
        """
        SELECT ats.tag,
               (ats.score - coalesce(tc.mean, %s)) / coalesce(tc.stddev, %s) AS z
        FROM artist_tag_scores ats
        -- ADR-020 P1: join the ARTIST-source moments (model||'#artist'), not the
        -- track moments, so z-scoring matches the distribution being scored.
        LEFT JOIN tag_calibration tc ON tc.tag = ats.tag AND tc.model = ats.model || %s
        WHERE ats.artist_id = %s AND ats.score != 'NaN'::real  -- NaN armor (pg NaN-equality law)
        ORDER BY z DESC LIMIT %s
        """,
        (gmean, gsd, ARTIST_SUFFIX, artist_id, TAG_K),
    ).fetchall()
    if primary:
        return {tag: max(1, round(float(z)) + 1) for tag, z in primary if float(z) > 0}
    rows = conn.execute(
        """
        WITH g AS (SELECT avg(score) gmean, greatest(stddev(score), 1e-6) gsd
                   FROM track_tag_scores),
        z AS (
            SELECT tts.tag,
                   avg((tts.score - coalesce(tc.mean, g.gmean))
                       / coalesce(tc.stddev, g.gsd)) AS mz,
                   count(*) AS cnt
            FROM track_tag_scores tts
            JOIN audio_track t ON t.id = tts.track_id
            CROSS JOIN g
            LEFT JOIN tag_calibration tc ON tc.tag = tts.tag AND tc.model = tts.model
            WHERE t.artist_id = %s
              AND tts.score != 'NaN'::real  -- pg law: NaN = NaN is TRUE; x=x can't detect it
            GROUP BY tts.tag
        )
        SELECT tag, mz, mz * sqrt(cnt) AS ranked
        FROM z WHERE mz > 0 ORDER BY ranked DESC LIMIT %s
        """,
        (artist_id, TAG_K),
    ).fetchall()
    return {tag: max(1, round(float(z)) + 1) for tag, z, _r in rows}


def publish_artists(factory: Connection, app: Connection, limit: int = 1000, since=None) -> int:
    """Upsert embedded artists into the serving DB. Returns artists published."""
    return publish_rows(factory, app, publishable_artists(factory, limit, since))


def publish_rows(factory: Connection, app: Connection, rows: list[tuple]) -> int:
    import json

    # global tag moments ONCE per batch (review finding: the per-artist CTE
    # full-scanned artist_tag_scores for EVERY artist — infeasible at 451k)
    g_artist = factory.execute(
        "SELECT avg(score), greatest(stddev(score), 1e-6) FROM artist_tag_scores"
    ).fetchone()
    published = 0
    for aid, mbid, name, source, embedding, _model, ratio in rows:
        urls = artist_urls(factory, aid)
        tags = artist_tags(factory, aid, g_moments=g_artist)
        perceptual = artist_perceptual(factory, aid)
        language = artist_language(factory, aid)
        location = artist_location(factory, aid)
        url_cols = "".join(f", {c} = %s" for c in urls)
        # Identity key (ADR-019): mbid when MB knows them; otherwise the
        # FACTORY artist id IS the app row id — provisional identity that an
        # accepted MB submission upgrades in place (mbid attaches via sync).
        if mbid:
            # ADR-019 loop close (review finding, critical): an artist first
            # published mbid-NULL has an app row keyed by factory id with
            # mbid NULL — ON CONFLICT (mbid) would MISS it and insert a
            # duplicate. Claim the provisional row first; then the conflict
            # fires and DO UPDATE refreshes in place (id + slug stable).
            app.execute(
                "UPDATE artists SET mbid = %s WHERE id = %s AND mbid IS NULL",
                (mbid, str(aid)),
            )
            conflict = "ON CONFLICT (mbid)"
            id_value, key = "gen_random_uuid()", mbid
        else:
            conflict = "ON CONFLICT (id)"
            id_value, key = "%s", str(aid)
        app.execute(
            f"""
            INSERT INTO artists (id, mbid, name, slug, tags, audio_embedding,
                                 signal_ratio, embedding_source, perceptual,
                                 language, location,
                                 audio_embedding_updated, created_at
                                 {"".join("," + c for c in urls)})
            VALUES ({id_value}, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), now()
                    {", %s" * len(urls)})
            {conflict} DO UPDATE SET
                name = EXCLUDED.name, slug = EXCLUDED.slug, tags = EXCLUDED.tags,
                audio_embedding = EXCLUDED.audio_embedding,
                signal_ratio = EXCLUDED.signal_ratio,
                embedding_source = EXCLUDED.embedding_source,
                perceptual = EXCLUDED.perceptual,
                language = coalesce(EXCLUDED.language, artists.language),
                location = coalesce(EXCLUDED.location, artists.location),
                audio_embedding_updated = now()
                {url_cols}
            """,
            ((*(() if mbid else (str(aid),)), mbid, name, resolve_slug(app, name, key),
              json.dumps(tags), embedding,
              ratio, source, json.dumps(perceptual) if perceptual else None,
              language, location,
              *urls.values(), *urls.values())),
        )
        published += 1
    return published


def main() -> None:
    import argparse
    import os

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="publish embedded artists to the serving DB")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--incremental", action="store_true", help="watermark mode (the hourly sync)")
    args = ap.parse_args()
    app_dsn = os.environ.get("APP_DATABASE_URL")
    if not app_dsn:
        raise SystemExit("APP_DATABASE_URL not set — publishing is deliberate, no default")
    with psycopg.connect(Settings().database_url) as factory, psycopg.connect(app_dsn) as app:
        if args.incremental:
            n = publish_incremental(factory, app, args.limit)
            # COMMIT ORDER LAW (review finding, high): app rows FIRST, then
            # the watermark. A crash between them re-publishes idempotent
            # upserts next run; the reverse order silently never publishes
            # the lost window again.
            app.commit()
            factory.commit()
        else:
            n = publish_artists(factory, app, args.limit)
            app.commit()
    print(f"published={n}", flush=True)


if __name__ == "__main__":
    main()
