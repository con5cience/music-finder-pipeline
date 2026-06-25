"""Seed the curated tag_alias map (alias -> canonical) so publish FOLDS garbage/
misspelled folksonomy tags onto a real genre instead of dropping them.

Strip-clean lane (high-confidence, default): a tag whose only problem is junk
punctuation ('#folk', 'e.l.e.c.t.r.o', 'alt.rock') aliases to its cleaned form IFF
that cleaned form is a known genre (approved or MB vocab). Punctuation removal
can't change meaning, so this is safe — validated 0 FP against human labels.

Edit-distance misspellings ('eletronica'->electronica) are NOT seeded here yet
(substitutions like afrobop->afropop mis-credit signal); that lane is pending.

Dry-run by default; --apply writes. Idempotent (ON CONFLICT DO NOTHING)."""

from __future__ import annotations

import argparse
import re

import psycopg

from pipeline.config import Settings

JUNK = re.compile(r"[#./\\|:;()\[\]{}*=+<>~^@?\"]")


def clean(tag: str) -> str:
    """Strip junk punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", JUNK.sub("", tag)).strip()


def strip_clean_aliases(conn: psycopg.Connection) -> dict[str, str]:
    vocab = frozenset(
        r[0]
        for r in conn.execute(
            "SELECT tag FROM tag_approved "
            "UNION SELECT lower(name) FROM mb_raw.genre "
            "UNION SELECT lower(name) FROM mb_raw.genre_alias"
        ).fetchall()
    )
    # only tags that actually carry artist signal (df>=1) are worth folding
    tags = [
        r[0]
        for r in conn.execute("SELECT DISTINCT tag FROM tag_review_freq WHERE df >= 1").fetchall()
    ]
    out: dict[str, str] = {}
    for t in tags:
        if t in vocab:
            continue
        cl = clean(t)
        if cl and cl != t and cl in vocab:
            out[t] = cl
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="seed tag_alias (strip-clean lane)")
    ap.add_argument("--apply", action="store_true", help="write aliases (default: dry-run)")
    args = ap.parse_args()
    with psycopg.connect(Settings().database_url) as conn:
        amap = strip_clean_aliases(conn)
        print(f"strip-clean aliases: {len(amap)}")
        for a in sorted(amap)[:30]:
            print(f"  {a!r} -> {amap[a]!r}")
        if args.apply:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO tag_alias (alias, canonical, source) VALUES (%s,%s,'ai') "
                    "ON CONFLICT (alias) DO NOTHING",
                    list(amap.items()),
                )
            conn.commit()
            print(f"APPLIED {len(amap)} aliases")


if __name__ == "__main__":
    main()
