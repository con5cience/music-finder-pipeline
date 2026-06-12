"""Whole-dataset cleanliness audit (2026-06-12, user mandate).

Read-only sweep across every data class the binding-lane audits don't cover:
vector hygiene, tag orphans, audio-track sanity, acoustic blind spots,
factory<->serving drift, fetch-cache index orphans. Run alongside anything —
it writes nothing.  uv run poe dataset-audit
"""

from __future__ import annotations

import os
from pathlib import Path

from psycopg import Connection

DEFAULT_MODEL = "muq-large-msd"


def audit_factory(conn: Connection) -> dict[str, object]:
    out: dict[str, object] = {}
    # L2 law: stored artist vectors are unit-norm. norm^2 = -(e <#> e).
    out["artist_vectors_off_norm"] = conn.execute(
        """SELECT count(*) FROM artist_embedding
           WHERE sqrt(-(embedding <#> embedding)) NOT BETWEEN 0.99 AND 1.01"""
    ).fetchone()[0]
    out["clip_vectors_off_norm_sample20k"] = conn.execute(
        """SELECT count(*) FROM (
             SELECT embedding FROM clip_embedding LIMIT 20000) s
           WHERE sqrt(-(s.embedding <#> s.embedding)) NOT BETWEEN 0.99 AND 1.01"""
    ).fetchone()[0]
    # tags surviving on artists whose embedding was since reset
    out["tag_rows_on_unembedded_artists"] = conn.execute(
        """SELECT count(DISTINCT ats.artist_id) FROM artist_tag_scores ats
           WHERE NOT EXISTS (SELECT 1 FROM artist_embedding e WHERE e.artist_id = ats.artist_id)"""
    ).fetchone()[0]
    # audio tracks whose platform identity is gone (orphans after unbinds)
    out["tracks_without_identity"] = conn.execute(
        """SELECT count(*) FROM audio_track t
           WHERE NOT EXISTS (SELECT 1 FROM platform_identity pi
                             WHERE pi.artist_id = t.artist_id AND pi.platform = t.platform)"""
    ).fetchone()[0]
    out["tracks_duration_outliers"] = conn.execute(
        "SELECT count(*) FROM audio_track WHERE duration_s < 5 OR duration_s > 3600"
    ).fetchone()[0]
    out["artists_blank_name"] = conn.execute(
        "SELECT count(*) FROM artist WHERE btrim(coalesce(display_name,'')) = ''"
    ).fetchone()[0]
    # acoustic blind spot: embedded from a B-tier (machine-guessed) source
    # with NO second source to corroborate — coherence cannot see these
    out["embedded_from_unverified_single_source"] = conn.execute(
        """SELECT count(*) FROM artist a
           JOIN platform_identity pi ON pi.artist_id = a.id
             AND pi.platform = a.embedding_source AND pi.binding_tier = 'B'
           WHERE NOT EXISTS (
             SELECT 1 FROM audio_track t WHERE t.artist_id = a.id
               AND t.platform <> a.embedding_source)"""
    ).fetchone()[0]
    return out


def audit_fetch_cache(conn: Connection, sample: int = 2000) -> dict[str, object]:
    from pipeline.config import Settings

    cache_dir = Path(os.path.expanduser(Settings().fetch_cache_dir))
    rows = conn.execute(
        "SELECT content_path FROM fetch_cache ORDER BY random() LIMIT %s", (sample,)
    ).fetchall()
    missing = sum(1 for (p,) in rows if not (cache_dir / p).exists())
    return {"cache_index_sample": len(rows), "cache_blobs_missing_host": missing}


def audit_serving(factory: Connection, app) -> dict[str, object]:
    out: dict[str, object] = {}
    out["app_rows"] = app.execute("SELECT count(*) FROM artists").fetchone()[0]
    # ban leaks: the do-not-rediscover law must hold on the serving side
    banned = factory.execute(
        "SELECT artist_id::text, mbid::text FROM ban_ledger"
    ).fetchall()
    leak = 0
    for aid, mbid in banned:
        leak += app.execute(
            "SELECT count(*) FROM artists WHERE id::text = %s OR (mbid IS NOT NULL AND mbid = %s)",
            (aid or "00000000-0000-0000-0000-000000000000", mbid),
        ).fetchone()[0]
    out["banned_artists_in_app"] = leak
    out["app_slug_duplicates"] = app.execute(
        "SELECT count(*) FROM (SELECT slug FROM artists WHERE slug IS NOT NULL "
        "GROUP BY slug HAVING count(*) > 1) d"
    ).fetchone()[0]
    # vector staleness sample: app vector should match factory's current one
    sample = app.execute(
        "SELECT id::text, mbid, audio_embedding FROM artists "
        "WHERE audio_embedding IS NOT NULL ORDER BY random() LIMIT 300"
    ).fetchall()
    stale = 0
    for aid, mbid, app_vec in sample:
        row = factory.execute(
            """SELECT ae.embedding::text FROM artist_embedding ae JOIN artist a ON a.id = ae.artist_id
               WHERE (a.mbid::text = %s OR a.id::text = %s) AND ae.model = %s""",
            (mbid, aid, DEFAULT_MODEL),
        ).fetchone()
        if row and row[0] != app_vec:
            stale += 1
    out["stale_vectors_in_sample300"] = stale
    return out


def main() -> None:
    import psycopg

    from pipeline.config import Settings

    findings: dict[str, object] = {}
    with psycopg.connect(Settings().database_url) as conn:
        findings.update(audit_factory(conn))
        findings.update(audit_fetch_cache(conn))
        app_dsn = os.environ.get(
            "APP_DATABASE_URL", "postgresql://musicfinder:musicfinder@localhost:5433/musicfinder"
        )
        try:
            with psycopg.connect(app_dsn) as app:
                findings.update(audit_serving(conn, app))
        except Exception as exc:  # noqa: BLE001
            findings["serving_audit"] = f"skipped: {exc!r}"
    width = max(len(k) for k in findings)
    clean = True
    for k, v in findings.items():
        flag = ""
        if isinstance(v, int) and v > 0 and k not in ("app_rows", "cache_index_sample"):
            flag = "  <-- INVESTIGATE"
            clean = False
        print(f"{k:<{width}}  {v}{flag}")
    print("\nverdict:", "CLEAN" if clean else "findings above need eyes")


if __name__ == "__main__":
    main()
