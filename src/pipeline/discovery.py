"""ADR-019 Bandcamp discovery: the Discover API → candidate ledger → gates
→ admission. "Discovery infrastructure for the underground", first organ.

Surface (validated live 2026-06-11): POST bandcamp.com/api/discover/1/
discover_web with {tag_norm_names, slice: "new", cursor} returns structured
items — band_name, band_url (subdomain = platform_id), band_location (the
MB area, free!), release_date, cursor pagination. No HTML scraping.

Flow: crawl(tags) upserts bc_candidate rows once per band → dedup gate
(platform identity known? exact-unique name → existing artist?) → admit(n)
creates mbid-NULL artist + pending bandcamp identity (THE TRICKLE VALVE —
explicit, budgeted; the wave seeder and standard cascade do the rest; full
analysis is the admission bar per ADR-019, enforced by the factory itself).
ALL traffic through cached_fetch (proxy + cache laws).
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from psycopg import Connection

from pipeline.fetch_cache import cached_fetch

DISCOVER_URL = "https://bandcamp.com/api/discover/1/discover_web"
PAGE_SIZE = 60


def _subdomain(band_url: str) -> str | None:
    """kiandray.bandcamp.com → kiandray (the bandcamp platform_id)."""
    host = urlparse(band_url).hostname or ""
    m = re.fullmatch(r"([a-z0-9-]+)\.bandcamp\.com", host)
    return m.group(1) if m else None


def crawl_tag(conn: Connection, tag: str, pages: int = 2, *, fetcher=None) -> dict:
    """Walk the 'new' slice for one tag; upsert each band once."""
    cursor = "*"
    seen = new = 0
    for _page in range(pages):
        payload = {
            "category_id": 0, "tag_norm_names": [tag], "geoname_id": 0,
            "slice": "new", "cursor": cursor, "size": PAGE_SIZE,
            "include_result_types": ["a", "s"],
        }
        import datetime as _dt

        bucket = _dt.datetime.now(_dt.UTC).date().isoformat()
        r = cached_fetch(conn, "bandcamp", DISCOVER_URL, post_json=payload,
                         fetcher=fetcher, cache_bucket=bucket)
        import json

        data = json.loads(r.body)
        results = data.get("results") or []
        for it in results:
            pid = _subdomain(it.get("band_url") or "")
            if not pid or not it.get("band_name"):
                continue
            seen += 1
            inserted = conn.execute(
                """
                INSERT INTO bc_candidate (platform_id, band_name, band_url, location,
                                          genre, tags, release_seen_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (platform_id) DO NOTHING
                RETURNING id
                """,
                (pid, it["band_name"], f"https://{pid}.bandcamp.com",
                 it.get("band_location") or None, str(it.get("band_genre_id") or ""),
                 [tag], it.get("release_date")),
            ).fetchone()
            if inserted:
                new += 1
        cursor = data.get("cursor") or ""
        if not cursor or not results:
            break
    return {"tag": tag, "seen": seen, "new": new}


def _sql_norm(name: str) -> str:
    """EXACTLY mirrors the SQL regexp_replace(lower(x),'[^a-z0-9]','','g') —
    deletion-form on BOTH sides is symmetric, so normalization asymmetry
    can't cause a wrong merge (review finding: NFKD-fold vs deletion
    disagreed on diacritics). Deletion-collisions ('Deli' vs 'Delić')
    surface as multi-match → candidate stays provisional, which is safe."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def dedup_gate(conn: Connection, limit: int = 1000) -> dict:
    """Mark candidates we already know: bound platform identity, or an
    exact-unique name match (the binding ladder's Tier-B test, reused)."""
    out = {"identity": 0, "name": 0}
    rows = conn.execute(
        "SELECT id, platform_id, band_name FROM bc_candidate WHERE status = 'candidate' "
        "ORDER BY first_seen_at LIMIT %s", (limit,)
    ).fetchall()
    for cid, pid, name in rows:
        known = conn.execute(
            "SELECT artist_id FROM platform_identity WHERE platform = 'bandcamp' AND platform_id = %s",
            (pid,),
        ).fetchone()
        if known:
            conn.execute(
                "UPDATE bc_candidate SET status = 'dedup_existing', status_reason = 'identity', "
                "artist_id = %s WHERE id = %s", (known[0], cid))
            out["identity"] += 1
            continue
        norm = _sql_norm(name)
        matches = conn.execute(
            """
            SELECT a.id FROM artist a
            WHERE regexp_replace(lower(a.display_name), '[^a-z0-9]', '', 'g') = %s
               OR (a.mbid IS NOT NULL AND EXISTS (
                     SELECT 1 FROM mb_raw.artist ma
                     JOIN mb_raw.artist_alias al ON al.artist = ma.id
                     WHERE ma.gid = a.mbid
                       AND regexp_replace(lower(al.name), '[^a-z0-9]', '', 'g') = %s))
            LIMIT 2
            """,
            (norm, norm),
        ).fetchall()
        if len(matches) == 1:
            # exact-unique → this is OUR artist on a new platform: hand the
            # identity to the EXISTING row (a binding, not a discovery)
            landed = conn.execute(
                """
                INSERT INTO platform_identity (artist_id, platform, platform_id, vanity_url, page_type)
                VALUES (%s, 'bandcamp', %s, %s, 'artist')
                ON CONFLICT DO NOTHING
                RETURNING id
                """,
                (matches[0][0], pid, f"https://{pid}.bandcamp.com"),
            ).fetchone()
            owner = matches[0][0] if landed else conn.execute(
                "SELECT artist_id FROM platform_identity WHERE platform='bandcamp' AND platform_id=%s",
                (pid,),
            ).fetchone()[0]  # concurrent binder won the race — record THE truth
            conn.execute(
                "UPDATE bc_candidate SET status = 'dedup_existing', status_reason = 'name_unique', "
                "artist_id = %s WHERE id = %s", (owner, cid))
            out["name"] += 1
        # multi-match or no match → stays 'candidate' (a multi-match here is
        # a NEW band sharing a name — provisional identity disambiguates)
    return out


def admit(conn: Connection, n: int) -> int:
    """THE TRICKLE VALVE: oldest n candidates become mbid-NULL artists with
    pending bandcamp identities. The wave seeder + cascade do everything
    else; quality gates run inside the factory (full analysis = the bar)."""
    rows = conn.execute(
        "SELECT id, platform_id, band_name, band_url FROM bc_candidate "
        "WHERE status = 'candidate' ORDER BY first_seen_at LIMIT %s", (n,)
    ).fetchall()
    admitted = 0
    for cid, pid, name, url in rows:
        aid = conn.execute(
            "INSERT INTO artist (display_name, mbid) VALUES (%s, NULL) RETURNING id", (name,)
        ).fetchone()[0]
        landed = conn.execute(
            """
            INSERT INTO platform_identity (artist_id, platform, platform_id, vanity_url, page_type)
            VALUES (%s, 'bandcamp', %s, %s, 'artist')
            ON CONFLICT DO NOTHING
            RETURNING id
            """,
            (aid, pid, url),
        ).fetchone()
        if landed is None:  # concurrent binder claimed this pid — no orphan rows
            conn.execute("DELETE FROM artist WHERE id = %s", (aid,))
            conn.execute(
                "UPDATE bc_candidate SET status = 'dedup_existing', status_reason = 'race' "
                "WHERE id = %s", (cid,))
            continue
        conn.execute(
            "UPDATE bc_candidate SET status = 'admitted', artist_id = %s WHERE id = %s",
            (aid, cid),
        )
        admitted += 1
    return admitted


def main() -> None:
    import argparse
    import json

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="Bandcamp discovery (ADR-019)")
    ap.add_argument("--tags", default="", help="comma-separated tag list to crawl")
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--admit", type=int, default=0, help="trickle valve: admit N oldest candidates")
    import sys

    argv = [a for i, a in enumerate(sys.argv[1:]) if not (a == "--" and i == 0)]
    args = ap.parse_args(argv)
    report: dict = {"crawl": [], "dedup": None, "admitted": 0}
    with psycopg.connect(Settings().database_url) as conn:
        for tag in [t.strip() for t in args.tags.split(",") if t.strip()]:
            report["crawl"].append(crawl_tag(conn, tag, args.pages))
        report["dedup"] = dedup_gate(conn)
        if args.admit:
            report["admitted"] = admit(conn, args.admit)
        conn.commit()
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
