"""Backfill Bandcamp human tags into bc_candidate from ALREADY-CACHED release
pages (#35) — no re-fetch. The audio-fetch path historically discarded the
page's `<a class="tag">` genres, and the only capture path (the discovery crawl)
is halted, so embedded Bandcamp artists carry at most a single generic tag. This
recovers the real per-artist tag set offline from fetch_cache.

Must run where the fetch-cache volume is mounted (i.e. inside a worker container:
`docker compose run --rm worker-io python -m pipeline.harvest_bc_tags`).
Idempotent (upsert on platform_id); coverage depends on cached bodies surviving
the storage cap — reported at the end.
"""

from __future__ import annotations

import argparse
import gzip
import pathlib
import re

import psycopg

from pipeline.config import Settings
from pipeline.sources.bandcamp import parse_bandcamp_tags, upsert_bandcamp_tags

_SUBDOMAIN_RE = re.compile(r"https?://([^.]+)\.bandcamp\.com", re.I)


def subdomain_of(url: str) -> str | None:
    m = _SUBDOMAIN_RE.match(url)
    return m.group(1).lower() if m else None


def run(limit: int | None = None, only: str | None = None) -> dict:
    s = Settings()
    cache_dir = pathlib.Path(s.fetch_cache_dir).expanduser()
    with psycopg.connect(s.database_url) as conn:
        # subdomain -> artist_id for bandcamp identities (the only ones publish can use)
        idmap: dict[str, str] = {}
        for pid, aid in conn.execute(
            "SELECT platform_id, artist_id FROM platform_identity "
            "WHERE platform='bandcamp' AND artist_id IS NOT NULL"
        ).fetchall():
            idmap[pid.lower()] = str(aid)

        # accumulate tags per subdomain from cached release/track HTML
        per_sub: dict[str, dict] = {}  # sub -> {"aid":.., "tags": ordered-set}
        missing = 0
        with conn.cursor(name="bc_cache_scan") as cur:  # server-side: fetch_cache is large
            cur.itersize = 2000
            cur.execute(
                "SELECT url, content_path FROM fetch_cache "
                "WHERE platform='bandcamp' AND status=200 AND content_type LIKE 'text/html%%'"
            )
            for url, cpath in cur:
                sub = subdomain_of(url)
                if not sub:
                    continue
                aid = idmap.get(sub)
                if not aid:
                    continue
                if only and only not in (sub, aid):
                    continue
                p = cache_dir / cpath
                try:
                    body = gzip.decompress(p.read_bytes())
                except (OSError, EOFError, gzip.BadGzipFile):
                    missing += 1  # evicted by the storage cap, or corrupt
                    continue
                e = per_sub.setdefault(sub, {"aid": aid, "tags": {}})
                for t in parse_bandcamp_tags(body):
                    e["tags"].setdefault(t, None)

        tagged = 0
        for i, (sub, e) in enumerate(per_sub.items()):
            if limit is not None and i >= limit:
                break
            if e["tags"]:
                upsert_bandcamp_tags(conn, e["aid"], sub, None, list(e["tags"]))
                tagged += 1
        conn.commit()

    report = {
        "bandcamp_identities": len(idmap),
        "subdomains_with_cache": len(per_sub),
        "artists_tagged": tagged,
        "missing_or_evicted_pages": missing,
    }
    print("harvest-bc-tags:", report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="backfill Bandcamp human tags from cached pages (#35)")
    ap.add_argument("--limit", type=int, default=None, help="cap artists upserted (smoke test)")
    ap.add_argument("--only", help="restrict to one subdomain or artist_id (verify)")
    args = ap.parse_args()
    run(limit=args.limit, only=args.only)


if __name__ == "__main__":
    main()
