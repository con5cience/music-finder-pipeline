"""Manage the curated tag black-hole — tag_manual_blocklist (#35). For tags that
should NEVER publish from any tier: mainly location-as-genre leaks via Bandcamp
human tags (cdmx, mexico, oakland, san francisco, …). Applied at the publish_rows
chokepoint, so a blocked tag is stripped from MB/Bandcamp/audio alike and a
re-publish cleans existing serving rows.

  poe block-tag <tag> [--reason r]   poe unblock-tag <tag>
  poe list-blocked-tags              poe seed-blocked-tags
"""

from __future__ import annotations

import argparse

import psycopg

from pipeline.config import Settings

# Music-city locations that recur as Bandcamp "genre" tags. Stored lowercase.
SEED: list[tuple[str, str]] = [
    (t, "location")
    for t in (
        "cdmx", "mexico", "mexico city", "oakland", "san francisco", "los angeles",
        "new york", "new york city", "brooklyn", "nyc", "la", "london", "berlin",
        "chicago", "seattle", "portland", "austin", "detroit", "atlanta",
        "toronto", "montreal", "paris", "tokyo", "melbourne", "sydney",
        "bay area", "uk", "usa", "u.s.a.", "england",
    )
]


def add(conn: psycopg.Connection, tag: str, reason: str | None = None) -> None:
    conn.execute(
        """INSERT INTO tag_manual_blocklist (tag, reason) VALUES (%s, %s)
           ON CONFLICT (tag) DO UPDATE SET reason = COALESCE(EXCLUDED.reason, tag_manual_blocklist.reason)""",
        (tag.strip().lower(), reason),
    )


def remove(conn: psycopg.Connection, tag: str) -> int:
    return conn.execute("DELETE FROM tag_manual_blocklist WHERE tag = %s", (tag.strip().lower(),)).rowcount


def main() -> None:
    ap = argparse.ArgumentParser(description="manage tag_manual_blocklist (#35)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("tag"); a.add_argument("--reason")
    r = sub.add_parser("remove"); r.add_argument("tag")
    sub.add_parser("list")
    sub.add_parser("seed")
    args = ap.parse_args()

    with psycopg.connect(Settings().database_url) as conn:
        if args.cmd == "add":
            add(conn, args.tag, args.reason)
            conn.commit()
            print(f"blocked: {args.tag.strip().lower()}")
        elif args.cmd == "remove":
            n = remove(conn, args.tag)
            conn.commit()
            print(f"unblocked: {args.tag.strip().lower()} ({n} row)")
        elif args.cmd == "seed":
            for tag, reason in SEED:
                add(conn, tag, reason)
            conn.commit()
            print(f"seeded {len(SEED)} blocked tags")
        elif args.cmd == "list":
            for tag, reason in conn.execute(
                "SELECT tag, reason FROM tag_manual_blocklist ORDER BY tag"
            ).fetchall():
                print(f"  {tag}\t{reason or ''}")


if __name__ == "__main__":
    main()
