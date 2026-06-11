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


def publishable_artists(conn: Connection, limit: int) -> list[tuple]:
    return conn.execute(
        """
        SELECT a.id, a.mbid::text, a.display_name, a.embedding_source,
               ae.embedding::text, ae.model, ae.signal_ratio
        FROM artist a
        JOIN artist_embedding ae ON ae.artist_id = a.id
        WHERE a.embedding_source IS NOT NULL
        ORDER BY a.id LIMIT %s
        """,
        (limit,),
    ).fetchall()


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


def artist_tags(conn: Connection, artist_id) -> dict[str, int]:
    """Calibrated artist-level tags. PRIMARY: artist_tag_scores (scored from
    the artist-mean MuLan vector at embed time — full resolution, no
    per-track-truncation pathology), z-ranked against per-tag corpus moments.
    FALLBACK until the v2 sweep covers an artist: the per-track aggregation
    (coverage-weighted, known-noisy on preview sources)."""
    primary = conn.execute(
        """
        WITH g AS (SELECT avg(score) gmean, greatest(stddev(score), 1e-6) gsd
                   FROM artist_tag_scores)
        SELECT ats.tag,
               (ats.score - coalesce(tc.mean, g.gmean)) / coalesce(tc.stddev, g.gsd) AS z
        FROM artist_tag_scores ats
        CROSS JOIN g
        LEFT JOIN tag_calibration tc ON tc.tag = ats.tag AND tc.model = ats.model
        WHERE ats.artist_id = %s AND ats.score != 'NaN'::real  -- NaN armor (pg NaN-equality law)
        ORDER BY z DESC LIMIT %s
        """,
        (artist_id, TAG_K),
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
            WHERE t.artist_id = %s AND tts.score != 'NaN'::real  -- NaN armor (pg law: NaN = NaN is TRUE; x=x can't detect it)
            GROUP BY tts.tag
        )
        SELECT tag, mz, mz * sqrt(cnt) AS ranked
        FROM z WHERE mz > 0 ORDER BY ranked DESC LIMIT %s
        """,
        (artist_id, TAG_K),
    ).fetchall()
    return {tag: max(1, round(float(z)) + 1) for tag, z, _r in rows}


def publish_artists(factory: Connection, app: Connection, limit: int = 1000) -> int:
    """Upsert embedded artists into the serving DB. Returns artists published."""
    import json

    published = 0
    for aid, mbid, name, source, embedding, _model, ratio in publishable_artists(factory, limit):
        urls = artist_urls(factory, aid)
        tags = artist_tags(factory, aid)
        perceptual = artist_perceptual(factory, aid)
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
                                 audio_embedding_updated, created_at
                                 {"".join("," + c for c in urls)})
            VALUES ({id_value}, %s, %s, %s, %s, %s, %s, %s, %s, now(), now()
                    {", %s" * len(urls)})
            {conflict} DO UPDATE SET
                name = EXCLUDED.name, slug = EXCLUDED.slug, tags = EXCLUDED.tags,
                audio_embedding = EXCLUDED.audio_embedding,
                signal_ratio = EXCLUDED.signal_ratio,
                embedding_source = EXCLUDED.embedding_source,
                perceptual = EXCLUDED.perceptual,
                audio_embedding_updated = now()
                {url_cols}
            """,
            ((*(() if mbid else (str(aid),)), mbid, name, resolve_slug(app, name, key),
              json.dumps(tags), embedding,
              ratio, source, json.dumps(perceptual) if perceptual else None,
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
    args = ap.parse_args()
    app_dsn = os.environ.get("APP_DATABASE_URL")
    if not app_dsn:
        raise SystemExit("APP_DATABASE_URL not set — publishing is deliberate, no default")
    with psycopg.connect(Settings().database_url) as factory, psycopg.connect(app_dsn) as app:
        n = publish_artists(factory, app, args.limit)
        app.commit()
    print(f"published={n}", flush=True)


if __name__ == "__main__":
    main()
