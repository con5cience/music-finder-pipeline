"""Title-overlap corroboration: MB recordings as ground truth (2026-06-12).

469 blind-spot artists have MB pages only on platforms we don't probe
(tidal/qobuz) or no pages at all — but MusicBrainz lists their RECORDINGS.
If the B-tier source we embedded carries the same track titles MB says this
artist recorded, that is track-level identity evidence with zero new
platform integrations.

  >=3 distinct meaningful title matches (or 2 covering 60% of the smaller
  side)                          -> confirmed: promote B->C (method
                                    title_overlap, matches recorded)
  zero overlap with >=5 titles on BOTH sides -> refuted: source_coherence
                                    flag (publish + MB hold). MB knowing 5+
                                    recordings none of which the source has
                                    is a real disagreement, not sparsity.
  anything thinner               -> gray / no_mb_recordings: stays held,
                                    recorded so re-runs skip.

Generic titles (intro, untitled, ...) and sub-4-char titles never count
toward confirmation. Load the recording tables once with --load.

Run:  uv run poe title-corroborate -- --load   (first time: ~35M-row COPY)
      uv run poe title-corroborate --limit 100000
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from psycopg import Connection

TITLE_TABLES = {"recording": 9, "artist_credit_name": 4}
STOP_TITLES = {"intro", "outro", "untitled", "interlude", "skit", "bonus", "bonus track", "demo", "live"}
CONFIRM_MATCHES = 3
REFUTE_MIN_EACH_SIDE = 5


def ensure_title_tables(conn: Connection, *, schema: str = "mb_raw") -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}.recording (
            id bigint, gid uuid, name text, artist_credit bigint, length int,
            comment text, edits_pending int, last_updated text, video text)""")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}.artist_credit_name (
            artist_credit bigint, position int, artist bigint, name text)""")


def index_title_tables(conn: Connection, *, schema: str = "mb_raw") -> None:
    conn.execute(f"CREATE INDEX IF NOT EXISTS acn_artist_idx ON {schema}.artist_credit_name (artist)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS recording_ac_idx ON {schema}.recording (artist_credit)")


def normalize_title(t: str) -> str:
    # non-ascii PUNCTUATION becomes a space BEFORE the ascii fold (en-dashes
    # etc. would otherwise be silently dropped, fusing words); non-ascii
    # LETTERS survive into NFKD so accents fold to their base.
    t = "".join(c if c.isascii() or c.isalnum() else " " for c in t)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"\((feat|ft|with|prod)[^)]*\)", "", t, flags=re.I)
    t = re.sub(r"\b(feat|ft)\.?\s.+$", "", t, flags=re.I)
    return re.sub(r"[^a-z0-9]+", " ", t.casefold()).strip()


def _meaningful(titles: set[str]) -> set[str]:
    return {t for t in titles if len(t) >= 4 and t not in STOP_TITLES}


def mb_titles(conn: Connection, mbid: str, *, schema: str = "mb_raw") -> set[str]:
    rows = conn.execute(
        f"""SELECT r.name FROM {schema}.artist a
            JOIN {schema}.artist_credit_name acn ON acn.artist = a.id
            JOIN {schema}.recording r ON r.artist_credit = acn.artist_credit
            WHERE a.gid = %s::uuid""",
        (mbid,),
    ).fetchall()
    return {normalize_title(r[0]) for r in rows if r[0]}


def source_titles(conn: Connection, artist_id, platform: str) -> set[str]:
    rows = conn.execute(
        "SELECT binding_evidence->>'title' FROM audio_track WHERE artist_id = %s AND platform = %s",
        (artist_id, platform),
    ).fetchall()
    return {normalize_title(r[0]) for r in rows if r[0]}


def title_corroborate(conn: Connection, *, limit: int = 100000,
                      commit_each: bool = False) -> dict:
    from pipeline.corroborate import _mark

    rows = conn.execute(
        """
        SELECT a.id, a.mbid::text, pi_b.id, a.embedding_source
        FROM artist a
        JOIN platform_identity pi_b ON pi_b.artist_id = a.id
          AND pi_b.platform = a.embedding_source AND pi_b.binding_tier = 'B'
          AND pi_b.binding_evidence->'corroboration'->>'status'
              IN ('no_a_pages', 'unprobeable')
        WHERE a.mbid IS NOT NULL
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    out = {"confirmed": 0, "refuted": 0, "gray": 0, "no_mb_recordings": 0}
    for artist_id, mbid, identity_id, source in rows:
        mb = _meaningful(mb_titles(conn, mbid))
        src = _meaningful(source_titles(conn, artist_id, source))
        matches = mb & src
        if not mb:
            _mark(conn, identity_id, {"status": "no_mb_recordings", "method": "title_overlap"})
            out["no_mb_recordings"] += 1
        elif len(matches) >= CONFIRM_MATCHES or (
            len(matches) >= 2 and src and mb and len(matches) >= 0.6 * min(len(mb), len(src))
        ):
            conn.execute(
                """UPDATE platform_identity SET binding_tier = 'C',
                   binding_evidence = COALESCE(binding_evidence, '{}'::jsonb)
                     || jsonb_build_object('corroboration', %s::jsonb)
                   WHERE id = %s""",
                (json.dumps({"status": "confirmed", "method": "title_overlap",
                             "matches": len(matches), "mb_titles": len(mb),
                             "src_titles": len(src)}), identity_id))
            out["confirmed"] += 1
        elif not matches and len(mb) >= REFUTE_MIN_EACH_SIDE and len(src) >= REFUTE_MIN_EACH_SIDE:
            _mark(conn, identity_id,
                  {"status": "refuted", "method": "title_overlap",
                   "matches": 0, "mb_titles": len(mb), "src_titles": len(src)})
            dup = conn.execute(
                "SELECT 1 FROM review_item WHERE kind='source_binding' AND subject_id=%s "
                "AND reason='source_coherence' AND status='pending'", (artist_id,)).fetchone()
            if not dup:
                conn.execute(
                    """INSERT INTO review_item (kind, subject_type, subject_id, reason, evidence, status)
                       VALUES ('source_binding','artist',%s,'source_coherence',%s,'pending')""",
                    (artist_id, json.dumps({
                        "platform": source,
                        "query": f"none of MB's {len(mb)} recording titles appear on the {source} source",
                        "coherence": {"min_cosine": None, "title_matches": 0,
                                      "worst_pair": ["mb_recordings", source]},
                    })))
            out["refuted"] += 1
        else:
            _mark(conn, identity_id,
                  {"status": "gray", "method": "title_overlap",
                   "matches": len(matches), "mb_titles": len(mb), "src_titles": len(src)})
            out["gray"] += 1
        if commit_each:
            conn.commit()
    out["processed"] = len(rows)
    return out


def main() -> None:
    import argparse

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="corroborate unreachable B-tier bindings via MB recording titles")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--load", action="store_true", help="COPY recording tables from the dump first")
    ap.add_argument("--dump-dir", default="/home/will/g/db-backups/mbdump-extract/mbdump")
    args = ap.parse_args()
    with psycopg.connect(Settings().database_url) as conn:
        if args.load:
            from pipeline.mb_bootstrap import load_mbdump

            ensure_title_tables(conn)
            print("loading recording tables (~35M rows, minutes)...")
            for table, n in load_mbdump(conn, args.dump_dir, tables=TITLE_TABLES).items():
                print(f"  mb_raw.{table}: {n:,}")
            index_title_tables(conn)
            conn.commit()
        print(title_corroborate(conn, limit=args.limit, commit_each=True))


if __name__ == "__main__":
    main()
