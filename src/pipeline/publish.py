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


def resolve_slug(app: Connection, name: str, mbid: str) -> str:
    """The app's own slug convention (its migration 031 indexes exactly this):
    clean base for the first holder, base-2/-3/... on homonym collisions.
    Idempotent: a row that already holds a slug in the family keeps it."""
    base = slug_base(name)
    mine = app.execute(
        "SELECT slug FROM artists WHERE mbid = %s AND (slug = %s OR slug ~ ('^' || %s || '-[0-9]+$'))",
        (mbid, base, base),
    ).fetchone()
    if mine:
        return mine[0]  # stable across re-publishes
    taken = {
        r[0] for r in app.execute(
            "SELECT slug FROM artists WHERE (slug = %s OR slug ~ ('^' || %s || '-[0-9]+$')) AND mbid != %s",
            (base, base, mbid),
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
               ae.embedding::text, ae.model
        FROM artist a
        JOIN artist_embedding ae ON ae.artist_id = a.id
        WHERE a.embedding_source IS NOT NULL AND a.mbid IS NOT NULL
        ORDER BY a.id LIMIT %s
        """,
        (limit,),
    ).fetchall()


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
    """Calibrated artist-level tags: mean z across the artist's tracks,
    top K with z > 0, integer weights (app convention)."""
    rows = conn.execute(
        """
        WITH g AS (SELECT avg(score) gmean, greatest(stddev(score), 1e-6) gsd
                   FROM track_tag_scores),
        z AS (
            SELECT tts.tag,
                   avg((tts.score - coalesce(tc.mean, g.gmean))
                       / coalesce(tc.stddev, g.gsd)) AS mz
            FROM track_tag_scores tts
            JOIN audio_track t ON t.id = tts.track_id
            CROSS JOIN g
            LEFT JOIN tag_calibration tc ON tc.tag = tts.tag AND tc.model = tts.model
            WHERE t.artist_id = %s
            GROUP BY tts.tag
        )
        SELECT tag, mz FROM z WHERE mz > 0 ORDER BY mz DESC LIMIT %s
        """,
        (artist_id, TAG_K),
    ).fetchall()
    return {tag: max(1, round(float(z)) + 1) for tag, z in rows}


def publish_artists(factory: Connection, app: Connection, limit: int = 1000) -> int:
    """Upsert embedded artists into the serving DB. Returns artists published."""
    import json

    published = 0
    for aid, mbid, name, _source, embedding, _model in publishable_artists(factory, limit):
        urls = artist_urls(factory, aid)
        tags = artist_tags(factory, aid)
        url_cols = "".join(f", {c} = %s" for c in urls)
        app.execute(
            f"""
            INSERT INTO artists (id, mbid, name, slug, tags, audio_embedding,
                                 audio_embedding_updated, created_at
                                 {"".join("," + c for c in urls)})
            VALUES (gen_random_uuid(), %s, %s, %s, %s, %s, now(), now()
                    {", %s" * len(urls)})
            ON CONFLICT (mbid) DO UPDATE SET
                name = EXCLUDED.name, slug = EXCLUDED.slug, tags = EXCLUDED.tags,
                audio_embedding = EXCLUDED.audio_embedding,
                audio_embedding_updated = now()
                {url_cols}
            """,
            (mbid, name, resolve_slug(app, name, mbid), json.dumps(tags), embedding,
             *urls.values(), *urls.values()),
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
