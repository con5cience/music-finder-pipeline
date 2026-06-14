"""Terminal observability: what the factory is doing RIGHT NOW.

All derived from existing timestamps (artist_embedding.computed_at,
platform_identity.scanned_at, search_attempt.searched_at,
track_head_runs.computed_at) — no event tables, no instrumentation; the
ledgers ARE the telemetry. The admin Ingestion card shows the same numbers.

Run:  uv run poe factory-status            one snapshot
      uv run poe factory-status -- --watch  refresh every 10s
"""

from __future__ import annotations

from psycopg import Connection

_WINDOWS = (("10m", "10 minutes"), ("1h", "1 hour"), ("24h", "24 hours"))


def _seed_queue(conn: Connection) -> dict:
    """Pending-to-embed artists, bucketed the way the wave seeder picks them.

    Mirrors wave_seeder.select_seed_batch eligibility (scan_status='pending',
    embedding_source IS NULL): fast lanes (deezer/bandcamp/soundcloud) before
    youtube-only, and within each, provisional (mbid NULL = ADR-019 discovery)
    front-runs MB-bound. An artist with both a fast and a yt pending identity
    counts under fast (that's the lane that will seed it).
    """
    q = {"fast": {"provisional": 0, "mb": 0}, "yt": {"provisional": 0, "mb": 0}}
    rows = conn.execute(
        """
        WITH elig AS (
          SELECT a.id, (a.mbid IS NULL) AS provisional,
                 bool_or(pi.platform IN ('deezer','bandcamp','soundcloud')
                         AND pi.scan_status = 'pending') AS fast_pending,
                 bool_or(pi.platform = 'youtube'
                         AND pi.scan_status = 'pending') AS yt_pending
          FROM artist a JOIN platform_identity pi ON pi.artist_id = a.id
          WHERE a.embedding_source IS NULL
          GROUP BY a.id, a.mbid
        )
        SELECT CASE WHEN fast_pending THEN 'fast' ELSE 'yt' END AS lane,
               provisional, count(*)
        FROM elig WHERE fast_pending OR yt_pending
        GROUP BY 1, 2
        """
    ).fetchall()
    for lane, provisional, n in rows:
        q[lane]["provisional" if provisional else "mb"] = n
    return q


def snapshot(conn: Connection) -> dict:
    out: dict = {"rates": {}, "recent": [], "scans": {}, "queue": {}}
    for label, iv in _WINDOWS:
        emb, scans, heads, binds = conn.execute(
            f"""
            SELECT
              (SELECT count(*) FROM artist_embedding WHERE computed_at > now() - interval '{iv}'),
              (SELECT count(*) FROM platform_identity WHERE scanned_at > now() - interval '{iv}'),
              (SELECT count(*) FROM track_head_runs WHERE computed_at > now() - interval '{iv}'),
              (SELECT count(*) FROM search_attempt WHERE searched_at > now() - interval '{iv}')
            """
        ).fetchone()
        out["rates"][label] = {"embeds": emb, "scans": scans, "head_runs": heads, "searches": binds}
    out["recent"] = conn.execute(
        """
        SELECT a.display_name, a.embedding_source, ae.clip_count,
               round(ae.signal_ratio::numeric, 2), to_char(ae.computed_at, 'HH24:MI:SS')
        FROM artist_embedding ae JOIN artist a ON a.id = ae.artist_id
        ORDER BY ae.computed_at DESC LIMIT 10
        """
    ).fetchall()
    out["scans"] = dict(
        conn.execute(
            """
            SELECT platform || ':' || scan_status, count(*)
            FROM platform_identity
            WHERE scanned_at > now() - interval '1 hour'
            GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
    )
    out["queue"] = _seed_queue(conn)
    return out


def render(s: dict) -> str:
    lines = ["=== factory activity ==="]
    lines.append(f"{'window':>6} {'embeds':>7} {'scans':>6} {'head_runs':>10} {'searches':>9}")
    for label, r in s["rates"].items():
        lines.append(
            f"{label:>6} {r['embeds']:>7} {r['scans']:>6} {r['head_runs']:>10} {r['searches']:>9}"
        )
    if s["scans"]:
        lines.append("scan verdicts (1h): " + ", ".join(f"{k}={v}" for k, v in s["scans"].items()))
    if s.get("queue"):
        qf, qy = s["queue"]["fast"], s["queue"]["yt"]
        lines.append("--- seed queue (pending, unembedded) ---")
        lines.append(f"  tier1 fast (deezer/bc/sc)  provisional={qf['provisional']:>6}  MB-bound={qf['mb']:>8}")
        lines.append(f"  tier2 youtube-only         provisional={qy['provisional']:>6}  MB-bound={qy['mb']:>8}")
    lines.append("--- last embeds ---")
    for name, src, clips, ratio, at in s["recent"]:
        lines.append(f"  {at}  {str(name)[:28]:28s} {str(src):10s} clips={clips} ratio={ratio}")
    return "\n".join(lines)


def main() -> None:
    import argparse
    import time

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="live factory activity")
    ap.add_argument("--watch", action="store_true", help="refresh every 10s")
    args = ap.parse_args()
    while True:
        with psycopg.connect(Settings().database_url) as conn:
            print(("\033[2J\033[H" if args.watch else "") + render(snapshot(conn)), flush=True)
        if not args.watch:
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
