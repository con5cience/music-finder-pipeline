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


def add(conn: psycopg.Connection, tag: str, reason: str | None = None, source: str = "human") -> None:
    t = tag.strip().lower()
    # block and approve are mutually exclusive — blocking clears any approval.
    conn.execute("DELETE FROM tag_approved WHERE tag = %s", (t,))
    conn.execute(
        """INSERT INTO tag_manual_blocklist (tag, reason, source) VALUES (%s, %s, %s)
           ON CONFLICT (tag) DO UPDATE SET reason = COALESCE(EXCLUDED.reason, tag_manual_blocklist.reason),
                                           source = EXCLUDED.source""",
        (t, reason, source),
    )


def remove(conn: psycopg.Connection, tag: str) -> int:
    return conn.execute("DELETE FROM tag_manual_blocklist WHERE tag = %s", (tag.strip().lower(),)).rowcount


def approve(conn: psycopg.Connection, tag: str, category: str | None = None, source: str = "human") -> None:
    """Thumbs-up: mark a tag reviewed-good (kept). Mutually exclusive with block."""
    t = tag.strip().lower()
    conn.execute("DELETE FROM tag_manual_blocklist WHERE tag = %s", (t,))
    conn.execute(
        """INSERT INTO tag_approved (tag, category, source) VALUES (%s, %s, %s)
           ON CONFLICT (tag) DO UPDATE SET category = COALESCE(EXCLUDED.category, tag_approved.category),
                                           source = EXCLUDED.source""",
        (t, category, source),
    )


def refresh_freq(conn: psycopg.Connection) -> None:
    """Refresh the per-tag corpus-frequency snapshot the admin Tags tab reads."""
    conn.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY tag_review_freq")


def main() -> None:
    ap = argparse.ArgumentParser(description="manage tag_manual_blocklist (#35)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("tag")
    a.add_argument("--reason")
    r = sub.add_parser("remove")
    r.add_argument("tag")
    ap_ = sub.add_parser("approve")
    ap_.add_argument("tag")
    ap_.add_argument("--category")
    sub.add_parser("list")
    sub.add_parser("seed")
    sub.add_parser("refresh-freq")
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
        elif args.cmd == "approve":
            approve(conn, args.tag, args.category)
            conn.commit()
            print(f"approved: {args.tag.strip().lower()}")
        elif args.cmd == "refresh-freq":
            refresh_freq(conn)
            conn.commit()
            print("refreshed tag_review_freq")
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
