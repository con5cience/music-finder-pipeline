"""Cross-source coherence: the audio-native binding validator (2026-06-12).

Name matching can never prove two platform pages are the same act — but the
audio can. For artists with clips from 2+ platforms, the per-source centroids
of a CORRECT binding agree (measured live: dense mass at cosine 0.7-0.95),
while a wrong binding embeds a different musician (outliers at 0.25-0.49,
with an EMPTY gap 0.5-0.66 between the modes). Threshold 0.6 sits in the gap.

Flag-only + gates: low-coherence artists get a review_item (reason
'source_coherence') and are EXCLUDED from publish and MB submission until a
human resolves which source is the impostor — quality gates exclude,
relatedness ranks (the locked law).

Run:  uv run poe coherence            (scan + flag)
      uv run poe coherence -- --report  (distribution only, no writes)
"""

from __future__ import annotations

import json

from psycopg import Connection

DEFAULT_MODEL = "muq-large-msd"
# below = the impostor zone; the 0.5-0.66 gap means this is not a knife edge
COHERENCE_THRESHOLD = 0.6
MIN_CLIPS_PER_SOURCE = 2  # one clip is too noisy to indict a binding


def artist_source_coherence(
    conn: Connection, artist_id, *, model: str = DEFAULT_MODEL
) -> dict | None:
    """Min pairwise cosine between per-platform clip centroids, or None when
    the artist has fewer than two platforms with >= MIN_CLIPS_PER_SOURCE."""
    rows = conn.execute(
        """
        WITH per_source AS (
            SELECT t.platform, avg(ce.embedding) AS centroid, count(*) AS clips
            FROM clip_embedding ce JOIN audio_track t ON t.id = ce.track_id
            WHERE t.artist_id = %s AND ce.model = %s
            GROUP BY t.platform HAVING count(*) >= %s
        )
        SELECT a.platform, b.platform, 1 - (a.centroid <=> b.centroid)
        FROM per_source a JOIN per_source b ON b.platform > a.platform
        """,
        (artist_id, model, MIN_CLIPS_PER_SOURCE),
    ).fetchall()
    if not rows:
        return None
    worst = min(rows, key=lambda r: r[2])
    return {
        "min_cosine": round(float(worst[2]), 4),
        "worst_pair": [worst[0], worst[1]],
        "pairs": {f"{a}~{b}": round(float(c), 4) for a, b, c in rows},
    }


def scan_coherence(
    conn: Connection,
    *,
    threshold: float = COHERENCE_THRESHOLD,
    model: str = DEFAULT_MODEL,
    limit: int = 100000,
) -> dict:
    """Flag every multi-source artist below threshold. Idempotent: an open
    flag is never duplicated; healed artists (re-scan above threshold, e.g.
    after the impostor source was unbound) get their pending flag closed."""
    artists = [r[0] for r in conn.execute(
        """
        SELECT t.artist_id
        FROM clip_embedding ce JOIN audio_track t ON t.id = ce.track_id
        WHERE ce.model = %s
        GROUP BY t.artist_id
        HAVING count(DISTINCT t.platform) >= 2
        LIMIT %s
        """,
        (model, limit),
    ).fetchall()]
    flagged = healed = 0
    for aid in artists:
        c = artist_source_coherence(conn, aid, model=model)
        if c is None:
            continue
        open_flag = conn.execute(
            "SELECT id FROM review_item WHERE kind = 'source_binding' "
            "AND reason = 'source_coherence' AND subject_id = %s AND status = 'pending'",
            (aid,),
        ).fetchone()
        if c["min_cosine"] < threshold and not open_flag:
            conn.execute(
                """
                INSERT INTO review_item (kind, subject_type, subject_id, reason, evidence, status)
                VALUES ('source_binding', 'artist', %s, 'source_coherence', %s, 'pending')
                """,
                (aid, json.dumps({
                    "platform": c["worst_pair"][1],
                    "query": f"sources disagree acoustically (cos {c['min_cosine']})",
                    "coherence": c,
                })),
            )
            flagged += 1
        elif c["min_cosine"] >= threshold and open_flag:
            conn.execute(
                "UPDATE review_item SET status = 'rejected', note = 'auto-healed: coherence recovered', "
                "resolved_at = now() WHERE id = %s",
                (open_flag[0],),
            )
            healed += 1
    return {"scanned": len(artists), "flagged": flagged, "healed": healed}


def main() -> None:
    import argparse

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="cross-source coherence scan (flag + gate)")
    ap.add_argument("--threshold", type=float, default=COHERENCE_THRESHOLD)
    ap.add_argument("--report", action="store_true", help="distribution only, no writes")
    args = ap.parse_args()
    with psycopg.connect(Settings().database_url) as conn:
        if args.report:
            rows = conn.execute(
                """
                WITH per_source AS (
                    SELECT t.artist_id, t.platform, avg(ce.embedding) AS centroid, count(*) clips
                    FROM clip_embedding ce JOIN audio_track t ON t.id = ce.track_id
                    WHERE ce.model = %s GROUP BY 1, 2 HAVING count(*) >= %s
                )
                SELECT a.artist_id, min(1 - (a.centroid <=> b.centroid))
                FROM per_source a JOIN per_source b
                  ON b.artist_id = a.artist_id AND b.platform > a.platform
                GROUP BY a.artist_id ORDER BY 2
                """,
                (DEFAULT_MODEL, MIN_CLIPS_PER_SOURCE),
            ).fetchall()
            for aid, c in rows[:25]:
                print(f"{aid}  min_cos={c:.3f}{'  <-- SUSPECT' if c < args.threshold else ''}")
            print(f"({len(rows)} multi-source artists total)")
        else:
            out = scan_coherence(conn, threshold=args.threshold)
            conn.commit()
            print(out)


if __name__ == "__main__":
    main()
