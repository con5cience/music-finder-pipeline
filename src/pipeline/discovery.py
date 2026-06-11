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


def harvest_tag_tree(conn: Connection, *, fetcher=None) -> list[str]:
    """BC's official genre tree (27 genres + 237 subgenres), harvested live
    from any tag page's data-blob — the wave's coverage list."""
    import html as ihtml
    import json
    import re as _re

    r = cached_fetch(conn, "bandcamp", "https://bandcamp.com/tag/ambient?sort=date", fetcher=fetcher)
    blob = json.loads(ihtml.unescape(_re.search(r'data-blob="([^"]*)"', r.body.decode("utf-8", "replace")).group(1)))
    st = blob["appData"]["initialState"]
    slugs = [g["slug"] for g in st.get("genres", []) if g.get("slug")]
    slugs += [g["slug"] for g in st.get("subgenres", []) if g.get("slug")]
    seen: set[str] = set()
    return [x for x in slugs if not (x in seen or seen.add(x))]


def crawl_label(conn: Connection, label_subdomain: str, *, fetcher=None) -> dict:
    """Label-roster discovery: a label's /artists page lists its roster —
    high-trust edges (a label vouches for its bands)."""
    import re as _re

    r = cached_fetch(conn, "bandcamp", f"https://{label_subdomain}.bandcamp.com/artists", fetcher=fetcher)
    h = r.body.decode("utf-8", "replace")
    found = set(_re.findall(r'https?://([a-z0-9-]+)\.bandcamp\.com', h))
    found.discard(label_subdomain)
    new = 0
    for pid in sorted(found):
        ins = conn.execute(
            """
            INSERT INTO bc_candidate (platform_id, band_name, band_url, tags, status_reason)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (platform_id) DO NOTHING RETURNING id
            """,
            (pid, pid, f"https://{pid}.bandcamp.com", [f"label:{label_subdomain}"], "label_roster"),
        ).fetchone()
        if ins:
            new += 1
    return {"label": label_subdomain, "roster_seen": len(found), "new": new}


def discover_wave(conn: Connection, pages: int = 1, admit_budget: int = 0, *, fetcher=None) -> dict:
    """The standing organ: crawl the ENTIRE official tag tree's 'new' slices,
    dedup, admit within budget. One-shot by design (no cron law)."""
    tags = harvest_tag_tree(conn, fetcher=fetcher)
    crawls = []
    for t in tags:
        try:
            crawls.append(crawl_tag(conn, t, pages, fetcher=fetcher))
        except Exception as e:  # noqa: BLE001 — one bad tag must not kill the wave
            crawls.append({"tag": t, "error": type(e).__name__})
    dedup = dedup_gate(conn, limit=100000)
    admitted = admit(conn, admit_budget) if admit_budget else 0
    return {
        "tags_crawled": len(tags),
        "new_candidates": sum(c.get("new", 0) for c in crawls),
        "errors": sum(1 for c in crawls if "error" in c),
        "dedup": dedup,
        "admitted": admitted,
    }


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
        "SELECT id, platform_id, band_name, status_reason FROM bc_candidate WHERE status = 'candidate' "
        "ORDER BY first_seen_at LIMIT %s", (limit,)
    ).fetchall()
    for cid, pid, name, src_reason in rows:
        banned = conn.execute(
            # pid bans match by pid; NAME bans apply only to ledger rows
            # WITHOUT pids (deliberate name-bans) — a new band sharing a
            # banned artist's name is not collateral (review finding)
            """SELECT 1 FROM ban_ledger WHERE platform_ids @> %s
               OR (platform_ids = '[]'::jsonb AND lower(display_name) = lower(%s))""",
            (f'["bandcamp:{pid}"]', name),
        ).fetchone()
        if banned:
            conn.execute(
                "UPDATE bc_candidate SET status = 'rejected', status_reason = 'banned' WHERE id = %s",
                (cid,))
            continue
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
        if src_reason == "label_roster":
            # label rosters carry SUBDOMAINS as names — the name tier would
            # wrong-merge slug collisions ('silverapples' → Silver Apples).
            # Identity-tier only; they admit provisionally (review finding).
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
    ap.add_argument("--wave", action="store_true", help="crawl the ENTIRE official tag tree")
    ap.add_argument("--label", default="", help="crawl one label's roster page")
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--admit", type=int, default=0, help="trickle valve: admit N oldest candidates")
    import sys

    argv = [a for i, a in enumerate(sys.argv[1:]) if not (a == "--" and i == 0)]
    args = ap.parse_args(argv)
    report: dict = {"crawl": [], "dedup": None, "admitted": 0}
    with psycopg.connect(Settings().database_url) as conn:
        if args.wave:
            report = discover_wave(conn, args.pages, args.admit)
        else:
            if args.label:
                report["crawl"].append(crawl_label(conn, args.label))
            for tag in [t.strip() for t in args.tags.split(",") if t.strip()]:
                report["crawl"].append(crawl_tag(conn, tag, args.pages))
            report["dedup"] = dedup_gate(conn)
            if args.admit:
                report["admitted"] = admit(conn, args.admit)
        conn.commit()
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
