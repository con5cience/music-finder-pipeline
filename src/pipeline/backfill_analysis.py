"""Head backfill: run pending AnalysisHeads over ALREADY-embedded tracks.

The embed pass runs heads on new tracks; this sweeps the existing corpus
whenever a head is added or versioned (track_head_runs is the per-head
ledger — ADR-015 pluggable heads). Re-downloads via the shared self-healing
fetch, decodes once, cuts the SAME windows the embedder uses. Never re-embeds.
Since it re-cuts the exact embed windows anyway, it also archives them (#54,
best-effort) so artists swept here become re-embeddable for a future MuQ→MusicFM
swap without a re-fetch.

Run:  uv run poe analysis-backfill --limit 100000 --batch 25
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from psycopg import Connection

from pipeline.embed_job import (
    _clips_for_track,
    _decode,
    _default_refresher,
    _fetch_with_refresh,
    _try_archive,
    fetch_audio,
)
from pipeline.heads import HeadContext, run_heads


def tracks_missing_heads(
    conn: Connection, heads: list, limit: int, artist_id: str | None = None
) -> list[tuple]:
    """Embedded tracks where any head's current version hasn't run."""
    clauses, params = [], []
    for h in heads:
        clauses.append(
            "NOT EXISTS (SELECT 1 FROM track_head_runs r "
            "WHERE r.track_id = t.id AND r.head = %s AND r.version >= %s)"
        )
        params.extend([h.name, h.version])
    sql = f"""
        SELECT DISTINCT t.id, t.audio_url, t.platform, t.platform_track_id, t.artist_id::text
        FROM audio_track t
        JOIN clip_embedding ce ON ce.track_id = t.id
        WHERE t.audio_url IS NOT NULL
          AND (%s::uuid IS NULL OR t.artist_id = %s::uuid)
          AND ({" OR ".join(clauses)})
        ORDER BY 1
        LIMIT %s
    """
    return conn.execute(sql, (artist_id, artist_id, *params, limit)).fetchall()


def backfill_tracks(
    conn: Connection,
    heads: list,
    limit: int = 100,
    *,
    artist_id: str | None = None,
    fetch=fetch_audio,
    refresher=_default_refresher,
) -> tuple[int, int]:
    """Run pending heads on up to `limit` embedded tracks. (done, skipped)."""
    done = skipped = 0
    by_artist: dict[str, list] = {}
    with tempfile.TemporaryDirectory(prefix="backfill-") as tmp:
        workdir = Path(tmp)
        for tid, url, platform, ptid, owner_id in tracks_missing_heads(conn, heads, limit, artist_id):
            # Per-track isolation: one poisoned track (transient URLError out
            # of the refresher, corrupt audio, head crash) must not abort a
            # 3,151-artist overnight sweep — it killed one in 2 min (rc=1).
            try:
                path = _fetch_with_refresh(conn, url, platform, ptid, workdir, fetch, refresher)
                if path is None:
                    skipped += 1
                    continue
                mono, sr = _decode(path)
                segs = _clips_for_track(mono, sr, path, platform, None, workdir, f"bf-{tid}")
                ctx = HeadContext(
                    conn=conn, track_id=tid, artist_id=owner_id, platform=platform,
                    mono=mono, sr=sr, clip_paths=[p for _s, _e, p in segs],
                )
                run_heads(conn, heads, ctx)
                # We already re-fetched + cut the exact embed windows to run heads,
                # so archive them too (#54): a future embedder swap (MuQ→MusicFM)
                # then re-embeds locally instead of re-downloading. Best-effort
                # (savepoint-wrapped; no-ops when the disk is capped / ffmpeg absent),
                # idempotent per track (ON CONFLICT in archive_window_clips).
                _try_archive(conn, owner_id, str(tid), platform, segs)
            except Exception as exc:  # noqa: BLE001 — sweep survives, track retries next run
                print(f"backfill: track {tid} isolated failure: {type(exc).__name__}: {exc}", flush=True)
                skipped += 1
                continue
            by_artist.setdefault(owner_id, []).append(ctx.mulan_vecs)
            done += 1
    from pipeline.heads import artist_tag_pass

    for owner_id, vecs in by_artist.items():
        artist_tag_pass(conn, heads, owner_id, vecs)
    return done, skipped


def main() -> None:
    import argparse

    import psycopg

    from pipeline.config import Settings
    from pipeline.heads import build_heads
    from pipeline.tags import MulanTagScorer, load_vocabulary

    ap = argparse.ArgumentParser(description="backfill pending analysis heads over embedded tracks")
    ap.add_argument("--limit", type=int, default=100, help="total tracks this run")
    ap.add_argument("--batch", type=int, default=25, help="tracks per transaction")
    ap.add_argument("--no-tags", action="store_true", help="CPU heads only (skip MuLan heads)")
    args = ap.parse_args()

    with psycopg.connect(Settings().database_url) as conn:
        scorer = None if args.no_tags else MulanTagScorer(load_vocabulary(conn))
        heads = build_heads(scorer)
        total_done = total_skipped = 0
        while total_done + total_skipped < args.limit:
            batch = min(args.batch, args.limit - total_done - total_skipped)
            done, skipped = backfill_tracks(conn, heads, batch)
            conn.commit()  # per-batch: a crash loses at most one batch, not the run
            total_done += done
            total_skipped += skipped
            print(f"batch done={done} skipped={skipped} (total {total_done}/{total_skipped})", flush=True)
            if done == 0:
                break  # nothing analyzable left (skipped tracks would just repeat)
    print(f"analyzed={total_done} skipped={total_skipped}", flush=True)


if __name__ == "__main__":
    main()
