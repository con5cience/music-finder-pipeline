"""Wave-1 backfill: run analysis heads over ALREADY-embedded tracks.

The embed pass only analyzes pending tracks; the pre-Wave-1 corpus (614
artists) needs its heads computed retroactively. Re-downloads via the same
self-healing fetch (URLs rot), decodes once, runs CPU heads + the tag head on
the same windows the embedder used (the windower is deterministic). Never
re-embeds. Idempotent: tracks with a current-version analysis row are skipped.

Run:  uv run python -m pipeline.backfill_analysis --limit 50
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from psycopg import Connection

from pipeline.analysis import ANALYSIS_VERSION, analyze_track, upsert_track_analysis
from pipeline.embed_job import (
    WINDOWED_PLATFORMS,
    WINDOWS_PER_TRACK,
    AudioFetchError,
    _decode,
    _default_refresher,
    fetch_audio,
)
from pipeline.windows import peak_windows


def unanalyzed_embedded_tracks(conn: Connection, limit: int, artist_id: str | None = None) -> list[tuple]:
    """Embedded tracks lacking a current-version analysis row."""
    return conn.execute(
        """
        SELECT DISTINCT t.id, t.audio_url, t.platform, t.platform_track_id, t.artist_id::text
        FROM audio_track t
        JOIN clip_embedding ce ON ce.track_id = t.id
        WHERE t.audio_url IS NOT NULL
          AND (%s::uuid IS NULL OR t.artist_id = %s::uuid)
          AND NOT EXISTS (
              SELECT 1 FROM track_analysis ta
              WHERE ta.track_id = t.id AND ta.analysis_version >= %s
          )
        ORDER BY 1
        LIMIT %s
        """,
        (artist_id, artist_id, ANALYSIS_VERSION, limit),
    ).fetchall()


def backfill_tracks(
    conn: Connection,
    limit: int = 100,
    *,
    artist_id: str | None = None,
    tag_scorer=None,
    fetch=fetch_audio,
    refresher=_default_refresher,
) -> tuple[int, int]:
    """Analyze up to `limit` embedded-but-unanalyzed tracks. Returns (done, skipped)."""
    done = skipped = 0
    with tempfile.TemporaryDirectory(prefix="backfill-") as tmp:
        workdir = Path(tmp)
        for tid, url, platform, ptid, owner_id in unanalyzed_embedded_tracks(conn, limit, artist_id):
            try:
                path = fetch(url, workdir)
            except AudioFetchError:
                fresh = refresher(conn, platform, ptid) if refresher else None
                if not fresh:
                    skipped += 1
                    continue
                try:
                    path = fetch(fresh, workdir)
                except AudioFetchError:
                    skipped += 1
                    continue
            mono, sr = _decode(path)
            upsert_track_analysis(conn, tid, analyze_track(mono, sr))
            if tag_scorer is not None:
                from pipeline.tags import upsert_track_tags

                if platform in WINDOWED_PLATFORMS:
                    import soundfile as sf

                    paths = []
                    for s, e in peak_windows(mono, sr, k=WINDOWS_PER_TRACK):
                        p = workdir / f"bf-{tid}-{s}.wav"
                        sf.write(p, mono[s * sr:e * sr], sr)
                        paths.append(str(p))
                else:
                    paths = [path]
                upsert_track_tags(conn, tid, tag_scorer.score_clips(owner_id, paths))
            done += 1
    return done, skipped


def main() -> None:
    import argparse

    import psycopg

    from pipeline.config import Settings
    from pipeline.tags import MulanTagScorer, load_vocabulary

    ap = argparse.ArgumentParser(description="backfill Wave-1 analysis over embedded tracks")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--no-tags", action="store_true", help="CPU heads only (skip MuLan)")
    args = ap.parse_args()

    with psycopg.connect(Settings().database_url) as conn:
        scorer = None if args.no_tags else MulanTagScorer(load_vocabulary(conn))
        done, skipped = backfill_tracks(conn, args.limit, tag_scorer=scorer)
        conn.commit()
    print(f"analyzed={done} skipped={skipped}")


if __name__ == "__main__":
    main()
