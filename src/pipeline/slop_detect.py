"""Behavioral AI-slop detection: catalog forensics (2026-06-12).

The threat model is slop FARMS — machine-generated catalogs uploaded at
scale — which leave statistical fingerprints no single track carries:

  duration_cv        near-identical track lengths across a catalog
                     (generators emit ~fixed-length output; humans don't)
  numbered_ratio     templated series titles ("... Vol. 7", "... Part 3")
  dup_title_ratio    the same normalized title recurring in one catalog
  big_no_mb          a large catalog with no MusicBrainz identity

Scores are flag-only (review_item reason 'ai_slop'); an open flag rides
the SAME freezer as source_coherence: held from publish and from MB
submission until a human looks. Placement follows the law — full analysis
is the admission bar, so the gate runs where catalogs exist (post-ingest),
not at discovery admission where no evidence exists yet.

Bias toward gray: thresholds favor missing a borderline farm over freezing
a human underground artist.

Run:  uv run poe slop-audit            (scan + flag)
"""

from __future__ import annotations

import json
import re
import statistics

from psycopg import Connection

from pipeline.title_corroborate import normalize_title

MIN_TRACKS = 8           # below this a uniform EP is not evidence
FLAG_THRESHOLD = 0.6
# keyword series (Vol. 7) OR bare trailing numbers (Ambient Sessions 3) —
# the >=50% catalog-share threshold keeps one 'Track 2' from counting
_NUMBERED = re.compile(r"(\b(vol|volume|part|pt|episode|ep|no|#)\.?\s*\d+|\s\d{1,3})\s*$", re.I)


def score_artist(conn: Connection, artist_id) -> dict:
    rows = conn.execute(
        """
        SELECT COALESCE((t.binding_evidence->>'track_duration_s')::float, t.duration_s),
               t.binding_evidence->>'title'
        FROM audio_track t
        JOIN artist a ON a.id = t.artist_id AND t.platform = COALESCE(a.embedding_source, t.platform)
        WHERE t.artist_id = %s
        """,
        (artist_id,),
    ).fetchall()
    durations = [r[0] for r in rows if r[0] and r[0] > 0]
    # 854-artist lesson: deezer stores the CONSTANT preview length as
    # duration_s — a single-valued duration set is a data artifact, never
    # evidence (real track lengths ride binding_evidence.track_duration_s)
    if len(set(durations)) <= 1:
        durations = []
    titles = [r[1] for r in rows if r[1]]
    out = {"n_tracks": len(rows), "score": 0.0, "duration_cv": None,
           "numbered_ratio": None, "dup_title_ratio": None}
    if len(rows) < MIN_TRACKS:
        return out
    score = 0.0
    if len(durations) >= MIN_TRACKS:
        cv = statistics.pstdev(durations) / max(statistics.mean(durations), 1)
        out["duration_cv"] = round(cv, 4)
        if cv < 0.05:
            score += 0.4
        elif cv < 0.12:
            score += 0.2
    if titles:
        numbered = sum(1 for t in titles if _NUMBERED.search(t)) / len(titles)
        out["numbered_ratio"] = round(numbered, 3)
        if numbered >= 0.5:
            score += 0.25
        norm = [normalize_title(t) for t in titles]
        dup = 1 - len(set(norm)) / max(len(norm), 1)
        out["dup_title_ratio"] = round(dup, 3)
        if dup >= 0.2:
            score += 0.2
    has_mbid = conn.execute(
        "SELECT mbid IS NOT NULL FROM artist WHERE id = %s", (artist_id,)
    ).fetchone()[0]
    if len(rows) >= 30 and not has_mbid:
        score += 0.15
    out["score"] = round(min(score, 1.0), 3)
    return out


def scan_slop(conn: Connection, *, limit: int = 100000,
              threshold: float = FLAG_THRESHOLD, commit_each: bool = False) -> dict:
    """Flag-only sweep over artists with catalogs big enough to judge.
    Idempotent: any existing ai_slop item (open OR resolved) is final —
    a human's release is never re-frozen by a re-scan."""
    artists = [r[0] for r in conn.execute(
        """
        SELECT t.artist_id FROM audio_track t
        GROUP BY t.artist_id HAVING count(*) >= %s
        LIMIT %s
        """,
        (MIN_TRACKS, limit),
    ).fetchall()]
    out = {"scanned": 0, "flagged": 0}
    for aid in artists:
        seen = conn.execute(
            "SELECT 1 FROM review_item WHERE kind='source_binding' AND reason='ai_slop' "
            "AND subject_id = %s", (aid,),
        ).fetchone()
        if seen:
            continue
        s = score_artist(conn, aid)
        out["scanned"] += 1
        if s["score"] >= threshold:
            conn.execute(
                """
                INSERT INTO review_item (kind, subject_type, subject_id, reason, evidence, status)
                VALUES ('source_binding', 'artist', %s, 'ai_slop', %s, 'pending')
                """,
                (aid, json.dumps({
                    "platform": "catalog",
                    "query": f"farm-shaped catalog (score {s['score']})",
                    "slop": s,
                })),
            )
            out["flagged"] += 1
        if commit_each:
            conn.commit()
    return out


def gate_unevaluated(conn: Connection, artist_ids: list, *,
                     threshold: float = FLAG_THRESHOLD) -> dict:
    """THE CONTINUOUS GATE (2026-06-12): called by publish and the MB queue
    at their choke points. Scores any artist with no evaluation row — or
    whose catalog grew since evaluation — and files ai_slop flags for farm
    shapes in the same cycle that would otherwise expose them. Human
    verdicts stay final: an existing review_item is never re-filed."""
    if not artist_ids:
        return {"evaluated": 0, "flagged": 0}
    stale = [r[0] for r in conn.execute(
        """
        SELECT a.id FROM artist a
        LEFT JOIN slop_evaluation se ON se.artist_id = a.id
        WHERE a.id = ANY(%s)
          AND (se.artist_id IS NULL
               OR (SELECT count(*) FROM audio_track t WHERE t.artist_id = a.id) > se.n_tracks)
        """,
        ([str(x) for x in artist_ids],),
    ).fetchall()]
    out = {"evaluated": 0, "flagged": 0}
    for aid in stale:
        s = score_artist(conn, aid)
        conn.execute(
            """
            INSERT INTO slop_evaluation (artist_id, score, n_tracks, features)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (artist_id) DO UPDATE SET score = EXCLUDED.score,
              n_tracks = EXCLUDED.n_tracks, features = EXCLUDED.features,
              evaluated_at = now()
            """,
            (aid, s["score"], s["n_tracks"], json.dumps(s)))
        out["evaluated"] += 1
        if s["score"] >= threshold:
            seen = conn.execute(
                "SELECT 1 FROM review_item WHERE kind='source_binding' AND reason='ai_slop' "
                "AND subject_id = %s", (aid,)).fetchone()
            if not seen:
                conn.execute(
                    """
                    INSERT INTO review_item (kind, subject_type, subject_id, reason, evidence, status)
                    VALUES ('source_binding', 'artist', %s, 'ai_slop', %s, 'pending')
                    """,
                    (aid, json.dumps({"platform": "catalog",
                                      "query": f"farm-shaped catalog (score {s['score']})",
                                      "slop": s})))
                out["flagged"] += 1
    return out


def main() -> None:
    import argparse

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="behavioral AI-slop sweep (flag-only)")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--threshold", type=float, default=FLAG_THRESHOLD)
    args = ap.parse_args()
    with psycopg.connect(Settings().database_url) as conn:
        print(scan_slop(conn, limit=args.limit, threshold=args.threshold, commit_each=True))


if __name__ == "__main__":
    main()
