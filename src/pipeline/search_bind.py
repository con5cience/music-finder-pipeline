"""Slice 3d: B-tier search binding — find platform pages for MB artists that
have NO audio-role identity (92k of 543k), bind only on unambiguous evidence.

Policy (locked from live probes, conservative by design — popularity never
auto-picks among homonyms; that's how the 4.67M-corpus contamination class
was born):
  exactly ONE normalized-exact candidate (vs display name or an MB alias)
      → Tier-B platform_identity (scan_status pending → cascade picks it up)
  MULTIPLE exact candidates ("Tomasito" has three bandcamp accounts)
      → review_item kind 'source_binding' with all candidates (admin decides)
  ZERO exact candidates → ledger verdict 'none' (re-searchable later)

Every search response goes through the fetch cache. search_attempt is the
per-(artist, platform) ledger: a searched artist is never re-searched.
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.parse

from psycopg import Connection

from pipeline.fetch_cache import cached_fetch

_PUNCT = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str) -> str:
    """Diacritics-stripped, casefolded, alnum-only — 'Delić' == 'delic'."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return _PUNCT.sub("", folded.casefold())


def artist_name_keys(conn: Connection, artist_id: str, display_name: str) -> set[str]:
    """Normalized display name + MB aliases (editor-curated alternate names)."""
    keys = {normalize_name(display_name)}
    mbid = conn.execute("SELECT mbid FROM artist WHERE id = %s", (artist_id,)).fetchone()[0]
    if mbid:
        rows = conn.execute(
            "SELECT aa.name FROM mb_raw.artist_alias aa "
            "JOIN mb_raw.artist a ON a.id = aa.artist WHERE a.gid = %s",
            (mbid,),
        ).fetchall()
        keys.update(normalize_name(r[0]) for r in rows)
    keys.discard("")
    return keys


# ---- per-platform searchers: (conn, name) -> list[candidate dict] -----------
# candidate: {"name": display, "platform_id": stable id, "popularity": int}


def search_deezer(conn: Connection, name: str) -> list[dict]:
    url = f"https://api.deezer.com/search/artist?q={urllib.parse.quote(name)}&limit=10"
    res = cached_fetch(conn, "deezer", url)
    return [
        {"name": d["name"], "platform_id": str(d["id"]), "popularity": d.get("nb_fan", 0)}
        for d in json.loads(res.body).get("data", [])
    ]


def search_bandcamp(conn: Connection, name: str) -> list[dict]:
    # autocomplete is search-only (B-tier evidence), POST → cache key carries
    # the query string explicitly
    url = "https://bandcamp.com/api/bcsearch_public_api/1/autocomplete_elastic?q=" + urllib.parse.quote(name)

    def _post(_url: str):
        import urllib.request

        req = urllib.request.Request(
            "https://bandcamp.com/api/bcsearch_public_api/1/autocomplete_elastic",
            data=json.dumps({"search_text": name, "search_filter": "b",
                             "full_page": False, "fan_id": None}).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": "music-finder-pipeline/0.1 (wstiern@gmail.com)"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()

    res = cached_fetch(conn, "bandcamp", url, fetcher=_post)
    out = []
    for x in json.loads(res.body).get("auto", {}).get("results", []):
        root = x.get("item_url_root") or ""
        sub = root.removeprefix("https://").split(".bandcamp.com")[0] if ".bandcamp.com" in root else None
        if x.get("type") == "b" and sub:
            out.append({"name": x.get("name") or "", "platform_id": sub,
                        "popularity": 0})
    return out


def search_soundcloud(conn: Connection, name: str) -> list[dict]:
    from pipeline.sources.soundcloud import _oauth_fetcher

    url = (f"https://api.soundcloud.com/users?q={urllib.parse.quote(name)}"
           f"&limit=10&linked_partitioning=true")
    res = cached_fetch(conn, "soundcloud", url, fetcher=_oauth_fetcher)
    return [
        {"name": u.get("username") or "", "platform_id": u.get("permalink"),
         "popularity": u.get("track_count", 0)}
        for u in json.loads(res.body).get("collection", [])
        if u.get("permalink")
    ]


SEARCHERS = {
    "deezer": search_deezer,
    "bandcamp": search_bandcamp,
    "soundcloud": search_soundcloud,
}


def bind_artist_on_platform(
    conn: Connection, artist_id: str, display_name: str, platform: str, *, searcher=None
) -> str:
    """Search one platform for one artist; apply the policy. Returns verdict."""
    already = conn.execute(
        "SELECT 1 FROM search_attempt WHERE artist_id = %s AND platform = %s",
        (artist_id, platform),
    ).fetchone()
    if already:
        return "skipped"
    searcher = searcher or SEARCHERS[platform]
    candidates = searcher(conn, display_name)
    keys = artist_name_keys(conn, artist_id, display_name)
    exact = [c for c in candidates if normalize_name(c["name"]) in keys]

    if len(exact) == 1:
        c = exact[0]
        evidence = {
            "method": "search_exact_unique", "query": display_name,
            "candidate_name": c["name"], "candidates_total": len(candidates),
            "popularity": c["popularity"],
        }
        conn.execute(
            """
            INSERT INTO platform_identity (artist_id, platform, platform_id, page_type,
                                           binding_tier, binding_evidence)
            VALUES (%s, %s, %s, 'artist', 'B', %s)
            ON CONFLICT DO NOTHING
            """,
            (artist_id, platform, c["platform_id"], json.dumps(evidence)),
        )
        verdict = "bound"
    elif len(exact) >= 2:
        conn.execute(
            """
            INSERT INTO review_item (kind, subject_type, subject_id, reason, evidence, status)
            VALUES ('source_binding', 'artist', %s, %s, %s, 'pending')
            """,
            (artist_id,
             f"{len(exact)} exact-name candidates on {platform}",
             json.dumps({"platform": platform, "query": display_name, "candidates": exact})),
        )
        verdict = "review"
    else:
        verdict = "none"

    conn.execute(
        "INSERT INTO search_attempt (artist_id, platform, query, verdict, candidates) "
        "VALUES (%s, %s, %s, %s, %s)",
        (artist_id, platform, display_name, verdict, len(candidates)),
    )
    return verdict


def unbound_artists(conn: Connection, limit: int) -> list[tuple]:
    """Artists with no audio-role identity and at least one platform unsearched."""
    return conn.execute(
        """
        SELECT a.id, a.display_name FROM artist a
        WHERE NOT EXISTS (
            SELECT 1 FROM platform_identity pi WHERE pi.artist_id = a.id
            AND pi.platform IN ('deezer','bandcamp','soundcloud'))
        AND (SELECT count(*) FROM search_attempt sa WHERE sa.artist_id = a.id) < 3
        ORDER BY a.id
        LIMIT %s
        """,
        (limit,),
    ).fetchall()


def main() -> None:
    import argparse
    import time

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="B-tier search binding over unbound artists")
    ap.add_argument("--limit", type=int, default=100, help="artists this run")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--sleep", type=float, default=0.4, help="politeness pause between artists")
    args = ap.parse_args()

    stats = {"bound": 0, "review": 0, "none": 0, "skipped": 0}
    with psycopg.connect(Settings().database_url) as conn:
        done = 0
        while done < args.limit:
            rows = unbound_artists(conn, min(args.batch, args.limit - done))
            if not rows:
                break
            for aid, name in rows:
                for platform in SEARCHERS:
                    v = bind_artist_on_platform(conn, str(aid), name, platform)
                    stats[v] += 1
                done += 1
                time.sleep(args.sleep)
            conn.commit()
            print(f"progress: {done} artists, {stats}", flush=True)
    print(f"FINAL: {stats}", flush=True)


if __name__ == "__main__":
    main()
