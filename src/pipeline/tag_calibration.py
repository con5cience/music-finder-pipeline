"""Tag-score calibration: per-tag corpus moments → z-score ranking.

refresh_calibration computes mean/sd per (tag, model) over track_tag_scores
in ONE SQL pass and replaces tag_calibration wholesale (it's derived data).
Tags with n < MIN_N keep a row but consumers should prefer the global
fallback the helper applies. calibrated_tags ranks a track's tags by z-score
— the presentable ordering; raw cosine stays the stored truth.

Run:  uv run poe tag-calibrate
"""

from __future__ import annotations

from psycopg import Connection

MIN_N = 20  # below this, per-tag moments are noise — use the global moments


def refresh_calibration(conn: Connection, model: str = "muq-mulan-large") -> int:
    """Recompute all per-tag moments for a model. Returns tags calibrated."""
    conn.execute("DELETE FROM tag_calibration WHERE model = %s", (model,))
    conn.execute(
        """
        INSERT INTO tag_calibration (tag, model, mean, stddev, n)
        SELECT tag, model, avg(score), greatest(stddev(score), 1e-6), count(*)
        FROM track_tag_scores
        WHERE model = %s
        GROUP BY tag, model
        HAVING count(*) >= 2
        """,
        (model,),
    )
    return conn.execute(
        "SELECT count(*) FROM tag_calibration WHERE model = %s", (model,)
    ).fetchone()[0]


def calibrated_tags(conn: Connection, track_id, model: str = "muq-mulan-large", k: int = 10) -> list[tuple]:
    """A track's tags ranked by z-score (per-tag moments, global fallback
    below MIN_N). Returns [(tag, raw_score, z)]."""
    return conn.execute(
        """
        WITH global_moments AS (
            SELECT avg(score) gmean, greatest(stddev(score), 1e-6) gsd
            FROM track_tag_scores WHERE model = %s
        )
        SELECT tts.tag, tts.score,
               (tts.score - CASE WHEN tc.n >= %s THEN tc.mean ELSE g.gmean END)
               / CASE WHEN tc.n >= %s THEN tc.stddev ELSE g.gsd END AS z
        FROM track_tag_scores tts
        CROSS JOIN global_moments g
        LEFT JOIN tag_calibration tc ON tc.tag = tts.tag AND tc.model = tts.model
        WHERE tts.track_id = %s AND tts.model = %s
        ORDER BY z DESC
        LIMIT %s
        """,
        (model, MIN_N, MIN_N, track_id, model, k),
    ).fetchall()


def main() -> None:
    import psycopg

    from pipeline.config import Settings

    with psycopg.connect(Settings().database_url) as conn:
        n = refresh_calibration(conn)
        conn.commit()
    print(f"calibrated {n} tags", flush=True)


if __name__ == "__main__":
    main()
