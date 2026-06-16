"""Phase 0 of ADR-020: freeze a tag-quality measurement baseline.

Emits the magnet metrics that every later phase A/Bs against, from the CURRENT
factory artist_tag_scores and the CURRENT serving artists.tags, plus a
spot-check sample of published tags for human precision rating. Read-only;
writes one JSON snapshot. Re-runnable — diff a post-fix run against the frozen
baseline file.

Run (inside publish-sync, which has both DSNs over the host network):
  docker compose exec -T publish-sync .venv/bin/python -m pipeline.tag_quality_baseline
"""

from __future__ import annotations

import json
import os
import sys

import psycopg

from pipeline.config import Settings

# The named offenders called out in ADR-020 — tracked explicitly so the A/B
# reads at a glance, not just in aggregate.
NAMED_MAGNETS = [
    "kilapanga", "zamrock", "geek rock", "orthodox pop", "fm synthesis",
    "pumpcore", "pop raï", "j-rock", "hyperpop", "mod revival",
]
SPOT_CHECK_N = 300  # ADR-020: larger than 50 so precision@10 is a stable estimate


def bucketize(rates: list[float]) -> dict[str, int]:
    """Cumulative assignment-rate histogram: how many tags are assigned to >= each
    threshold of artists. Cumulative (a >=15% tag is also counted in >=10% etc.),
    matching the ADR-020 reporting."""
    return {
        "ge_15pct": sum(1 for r in rates if r >= 0.15),
        "ge_10pct": sum(1 for r in rates if r >= 0.10),
        "ge_5pct": sum(1 for r in rates if r >= 0.05),
        "ge_1pct": sum(1 for r in rates if r >= 0.01),
        "total_tags": len(rates),
    }


def _factory_metrics(conn: psycopg.Connection) -> dict:
    tagged = conn.execute("SELECT count(DISTINCT artist_id) FROM artist_tag_scores").fetchone()[0]
    total_rows = conn.execute("SELECT count(*) FROM artist_tag_scores").fetchone()[0]
    rows = conn.execute(
        "SELECT tag, count(DISTINCT artist_id) AS df FROM artist_tag_scores GROUP BY tag"
    ).fetchall()
    rates = [df / tagged for (_tag, df) in rows] if tagged else []
    rate_by_tag = {tag: df / tagged for (tag, df) in rows} if tagged else {}
    # top-100 concentration: share of all assignment rows held by the 100 biggest tags
    top100 = conn.execute(
        "SELECT coalesce(sum(c),0) FROM (SELECT count(*) c FROM artist_tag_scores "
        "GROUP BY tag ORDER BY c DESC LIMIT 100) t"
    ).fetchone()[0]
    # per-artist magnet load: avg # of >=10% tags an artist carries
    magnet_load = conn.execute(
        """
        WITH magnets AS (
          SELECT tag FROM artist_tag_scores GROUP BY tag
          HAVING count(DISTINCT artist_id)::float / %s >= 0.10
        )
        SELECT coalesce(avg(cnt), 0) FROM (
          SELECT artist_id, count(*) cnt FROM artist_tag_scores
          WHERE tag IN (SELECT tag FROM magnets) GROUP BY artist_id
        ) t
        """,
        (tagged,),
    ).fetchone()[0]
    return {
        "tagged_artists": tagged,
        "total_rows": total_rows,
        "distinct_tags": len(rows),
        "assignment_rate_histogram": bucketize(rates),
        "top100_concentration_pct": round(100 * top100 / total_rows, 1) if total_rows else 0,
        "avg_magnet_load_per_artist": round(float(magnet_load), 2),
        "named_magnet_rate_pct": {m: round(100 * rate_by_tag.get(m, 0.0), 1) for m in NAMED_MAGNETS},
    }


def _serving_metrics(conn: psycopg.Connection) -> dict:
    total = conn.execute("SELECT count(*) FROM artists WHERE tags IS NOT NULL AND tags::text <> '{}'").fetchone()[0]
    rows = conn.execute(
        "SELECT key AS tag, count(*) AS df FROM artists, jsonb_each_text(tags::jsonb) GROUP BY key"
    ).fetchall()
    rates = [df / total for (_tag, df) in rows] if total else []
    rate_by_tag = {tag: df / total for (tag, df) in rows} if total else {}
    tags_ge_5pct = sum(1 for r in rates if r >= 0.05)
    avg_tags = conn.execute(
        "SELECT coalesce(avg((SELECT count(*) FROM jsonb_object_keys(tags::jsonb))), 0) "
        "FROM artists WHERE tags IS NOT NULL AND tags::text <> '{}'"
    ).fetchone()[0]
    sample = conn.execute(
        """
        (SELECT name, mbid IS NOT NULL AS mb, tags FROM artists
         WHERE tags::text <> '{}' AND mbid IS NOT NULL ORDER BY id LIMIT %s)
        UNION ALL
        (SELECT name, mbid IS NOT NULL AS mb, tags FROM artists
         WHERE tags::text <> '{}' AND mbid IS NULL ORDER BY id LIMIT %s)
        """,
        (SPOT_CHECK_N // 2, SPOT_CHECK_N // 2),
    ).fetchall()
    return {
        "published_artists": total,
        "avg_published_tags_per_artist": round(float(avg_tags), 2),
        "assignment_rate_histogram": bucketize(rates),
        "tags_served_ge_5pct": tags_ge_5pct,
        "named_magnet_served_pct": {m: round(100 * rate_by_tag.get(m, 0.0), 1) for m in NAMED_MAGNETS},
        "spot_check_sample": [
            {"name": n, "mb_tagged": mb, "tags": tags} for (n, mb, tags) in sample
        ],
    }


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "tag_quality_baseline.json"
    snapshot: dict = {}
    with psycopg.connect(Settings().database_url) as factory:
        snapshot["factory"] = _factory_metrics(factory)
    app_dsn = os.environ.get("APP_DATABASE_URL")
    snapshot["serving"] = None
    if app_dsn:
        with psycopg.connect(app_dsn) as app:
            snapshot["serving"] = _serving_metrics(app)

    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    f_ = snapshot["factory"]
    print("=== Tag-quality baseline (ADR-020 Phase 0) ===", flush=True)
    print(f"FACTORY: {f_['tagged_artists']} tagged artists, {f_['distinct_tags']} tags, "
          f"{f_['total_rows']} rows", flush=True)
    print(f"  assignment-rate histogram (cumulative): {f_['assignment_rate_histogram']}", flush=True)
    print(f"  top-100 concentration: {f_['top100_concentration_pct']}% of all assignments", flush=True)
    print(f"  avg magnet load (>=10% tags) per artist: {f_['avg_magnet_load_per_artist']}", flush=True)
    print(f"  named magnets (assignment %): {f_['named_magnet_rate_pct']}", flush=True)
    s_ = snapshot["serving"]
    if s_:
        print(f"SERVING: {s_['published_artists']} artists, "
              f"avg {s_['avg_published_tags_per_artist']} tags each", flush=True)
        print(f"  tags served to >=5% of artists: {s_['tags_served_ge_5pct']}", flush=True)
        print(f"  named magnets (served %): {s_['named_magnet_served_pct']}", flush=True)
        print(f"  spot-check sample: {len(s_['spot_check_sample'])} artists", flush=True)
    else:
        print("SERVING: APP_DATABASE_URL not set — serving metrics skipped", flush=True)
    print(f"snapshot written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
