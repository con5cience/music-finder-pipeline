"""MB bootstrap (ADR-017 §9): load mbdump tables, derive Tier-A identities.

`load_mbdump` streams the extracted fullexport table files (PG COPY text
format: tab-separated, \\N nulls) into the `mb_raw` mirror schema —
truncate-and-reload, so a re-run with a newer dump IS the refresh path.
`derive_identities` turns artist→url relationships into `artist` +
`platform_identity` rows by URL host pattern (more robust than MB's link-type
taxonomy), skipping ended relationships. Both are idempotent.

Run:  uv run python -m pipeline.mb_bootstrap --dir ~/g/db-backups/mbdump-extract/mbdump
"""

from __future__ import annotations

from pathlib import Path

from psycopg import Connection

# table → expected column count (verified against CreateTables.sql + the
# 20260606 dump). A mismatch on load means upstream schema drift: fail fast.
EXPECTED_COLS = {
    "artist": 19,
    "artist_alias": 16,
    "artist_tag": 4,
    "tag": 3,
    "url": 5,
    "l_artist_url": 9,
    "link": 11,
    "link_type": 16,
    "genre": 6,        # canonical tag vocabulary (Wave-1 analysis heads)
    "genre_alias": 16,  # editor-curated merges: "synth punk" → "synth-punk"
}

# platform → (host match regex, platform_id capture regex), applied to url.url.
# The capture group is the stable per-platform identifier we key identity on.
PLATFORM_PATTERNS: dict[str, tuple[str, str]] = {
    "deezer": (r"deezer\.com/(?:[a-z]{2}/)?artist/[0-9]+", r"deezer\.com/(?:[a-z]{2}/)?artist/([0-9]+)"),
    "bandcamp": (r"^https?://[a-z0-9][a-z0-9-]*\.bandcamp\.com", r"^https?://([a-z0-9][a-z0-9-]*)\.bandcamp\.com"),
    "soundcloud": (r"soundcloud\.com/[a-zA-Z0-9_-]+", r"soundcloud\.com/([a-zA-Z0-9_-]+)"),
    "youtube": (r"youtube\.com/channel/[A-Za-z0-9_-]+", r"youtube\.com/channel/([A-Za-z0-9_-]+)"),
    "tidal": (r"tidal\.com/(?:browse/)?artist/[0-9]+", r"tidal\.com/(?:browse/)?artist/([0-9]+)"),
    "apple_music": (r"music\.apple\.com/[a-z]{2}/artist/", r"music\.apple\.com/[a-z]{2}/artist/(?:[^/]+/)?([0-9]+)"),
    # qobuz artist URLs: …/{locale}/interpreter/{slug}/{numeric} (slug is often
    # literally "-") or open.qobuz.com/artist/{numeric}. The trailing numeric is
    # the stable id — capturing the slug collides half the corpus on "-".
    "qobuz": (r"qobuz\.com/", r"qobuz\.com/(?:[a-z]{2}-[a-z]{2}/interpreter/[^/]+|artist)/([0-9]+)"),
}

_CHUNK = 1 << 20  # 1 MiB


def _assert_layout(path: Path, table: str, expected: dict[str, int] | None = None) -> None:
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    if not first:  # empty table file is legal
        return
    n = first.count("\t") + 1
    want = (expected or EXPECTED_COLS)[table]
    if n != want:
        raise RuntimeError(
            f"mbdump/{table}: {n} columns, expected {want} — upstream MB schema drift; "
            f"re-verify column layout before loading"
        )


def load_mbdump(
    conn: Connection, dump_dir: Path | str, *, schema: str = "mb_raw",
    tables: dict[str, int] | None = None,
) -> dict[str, int]:
    """Truncate-and-reload every `schema` table from `dump_dir` (schema param:
    ADR-018 shadow loads into mb_raw_next). Returns row counts."""
    dump_dir = Path(dump_dir)
    counts: dict[str, int] = {}
    for table in (tables or EXPECTED_COLS):
        path = dump_dir / table
        if not path.exists():
            raise FileNotFoundError(f"missing mbdump table file: {path}")
        _assert_layout(path, table, tables or EXPECTED_COLS)
        conn.execute(f"TRUNCATE {schema}.{table}")
        with conn.cursor() as cur, cur.copy(f"COPY {schema}.{table} FROM STDIN") as copy, open(path, "rb") as f:
            while chunk := f.read(_CHUNK):
                copy.write(chunk)
        counts[table] = conn.execute(f"SELECT count(*) FROM {schema}.{table}").fetchone()[0]
    return counts


def _patterns_values_sql() -> tuple[str, list[str]]:
    rows = ", ".join(["(%s, %s, %s)"] * len(PLATFORM_PATTERNS))
    params: list[str] = []
    for platform, (host_re, id_re) in PLATFORM_PATTERNS.items():
        params.extend([platform, host_re, id_re])
    return rows, params


def matched_artist_url_cte(schema: str) -> tuple[str, list[str]]:
    """`WITH patterns, matched AS (...)` selecting MB artists whose url matches
    one of OUR platform patterns — exactly the set derive_identities mints into
    `artist`/`platform_identity`.

    Shared so the ADR-018 refresh "adds" preview counts the same population the
    apply actually lands; counting MB artists with *any* url over-reported it by
    ~360x (con5cience/music-finder-pipeline#1)."""
    rows, params = _patterns_values_sql()
    cte = f"""
        WITH patterns (platform, host_re, id_re) AS (VALUES {rows}),
        matched AS (
            SELECT a.gid AS mbid, a.name, p.platform,
                   substring(u.url FROM p.id_re) AS platform_id, u.url
            FROM {schema}.l_artist_url lau
            JOIN {schema}.link l ON l.id = lau.link AND NOT l.ended
            JOIN {schema}.url u ON u.id = lau.entity1
            JOIN {schema}.artist a ON a.id = lau.entity0
            JOIN patterns p ON u.url ~ p.host_re
            WHERE substring(u.url FROM p.id_re) IS NOT NULL
        )
    """
    return cte, params


def derive_identities(conn: Connection, *, schema: str = "mb_raw") -> dict[str, int]:
    """Derive artist + platform_identity rows from active artist→url rels.

    Idempotent: conflicts on artist.mbid / (platform, platform_id) are skipped
    — which makes this the ADR-018 refresh diff-apply too (run against
    mb_raw_next: only NEW artists/identities land). Returns per-platform
    identity counts (cumulative, post-derive).
    """
    matched_cte, params = matched_artist_url_cte(schema)
    # Two statements: an INSERT's rows aren't visible to later joins in the
    # same statement, and identities must join the artists they just created.
    conn.execute(
        matched_cte + """
        INSERT INTO artist (display_name, mbid)
        SELECT DISTINCT ON (mbid) name, mbid FROM matched
        ON CONFLICT (mbid) WHERE mbid IS NOT NULL DO NOTHING
        """,
        params,
    )
    conn.execute(
        matched_cte + """
        INSERT INTO platform_identity (artist_id, platform, platform_id, vanity_url, page_type)
        SELECT DISTINCT ON (m.platform, m.platform_id) ar.id, m.platform, m.platform_id, m.url, 'artist'
        FROM matched m
        JOIN artist ar ON ar.mbid = m.mbid
        ON CONFLICT (platform, platform_id) DO NOTHING
        """,
        params,
    )
    # Losing claims must not vanish (the (Nit)neroc class, 2026-06-12): when
    # 2+ MB artists declare ONE platform page, the slot's unique key gives the
    # page one owner and ON CONFLICT silently dropped every other claim —
    # ~6,950 of them corpus-wide — orphaning artists into search-binding and
    # making operators hand-disambiguate pages MB explicitly declares. A
    # shared URL is duplicate-artist evidence: file the loser a comparison
    # card pointing at the winner (renders like the fp-collision cards).
    conn.execute(
        matched_cte + """
        INSERT INTO review_item (kind, subject_type, subject_id, reason, evidence, status)
        SELECT DISTINCT ON (loser.id, m.platform)
               'source_binding', 'artist', loser.id, 'mb_shared_url',
               jsonb_build_object(
                 'platform', m.platform, 'query', m.url, 'shared_url', m.url,
                 'url_collision', jsonb_build_object('other_artist', pi.artist_id::text)),
               'pending'
        FROM matched m
        JOIN artist loser ON loser.mbid = m.mbid
        JOIN platform_identity pi ON pi.platform = m.platform
          AND pi.platform_id = m.platform_id AND pi.artist_id != loser.id
        WHERE NOT EXISTS (
          SELECT 1 FROM review_item ri
          WHERE ri.kind = 'source_binding' AND ri.reason = 'mb_shared_url'
            AND ri.subject_id = loser.id AND ri.evidence->>'platform' = m.platform)
        """,
        params,
    )
    return dict(
        conn.execute(
            "SELECT platform, count(*) FROM platform_identity GROUP BY platform ORDER BY platform"
        ).fetchall()
    )


def main() -> None:
    import argparse

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="MB bootstrap: load mbdump + derive Tier-A identities")
    ap.add_argument("--dir", required=True, help="extracted mbdump directory (contains artist, url, ...)")
    ap.add_argument("--skip-load", action="store_true", help="only run derivation (mb_raw already loaded)")
    args = ap.parse_args()

    with psycopg.connect(Settings().database_url) as conn:
        if not args.skip_load:
            print("loading mbdump tables (this is GBs of COPY — minutes)...")
            for table, n in load_mbdump(conn, args.dir).items():
                print(f"  mb_raw.{table:14s} {n:>12,}")
        print("deriving Tier-A identities...")
        for platform, n in derive_identities(conn).items():
            print(f"  {platform:14s} {n:>12,}")
        conn.commit()
    print("done.")


if __name__ == "__main__":
    main()
