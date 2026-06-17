"""Populate serving `deezer_top_tracks` from the factory's analyzed Deezer tracks (#26).

The embedder already captured the Deezer track IDs we analyzed
(`audio_track.platform_track_id`, platform='deezer'). This one-shot syncs them
factory->serving as embed-ready per-track widget rows, so the serving
`deezerProvider` renders the tracks we ANALYZED instead of the (buggy)
artist-level widget — with NO serving-time live fetching (no Deezer rate-limit
risk). Idempotent; safe to re-run and to fold into the hourly publish.

Env: APP_DATABASE_URL (the serving DB; no default — writing serving is deliberate).
"""
from __future__ import annotations

import json
import os

import psycopg
from psycopg import Connection

from pipeline.config import Settings

_WIDGET = "https://widget.deezer.com/widget/dark/track/{}"
MAX_TRACKS = 3


def build_deezer_top_tracks(track_ids: list[str], max_n: int = MAX_TRACKS) -> list[dict]:
    """Embed-ready per-track rows in the serving snake_case shape
    ({name, embed_url}), capped + deduped. The Deezer widget renders the real
    track title itself, so `name` is a generic label."""
    seen: list[str] = []
    for t in track_ids:
        if t and t not in seen:
            seen.append(t)
        if len(seen) >= max_n:
            break
    return [{"name": "Deezer", "embed_url": _WIDGET.format(t)} for t in seen]


def populate_deezer_embeds(factory: Connection, app: Connection, limit: int | None = None) -> int:
    """Sync analyzed Deezer track ids -> serving deezer_top_tracks. Matches the
    serving row by mbid (MB artists) or id (discovery, mbid NULL); only published
    serving rows are touched (unpublished factory artists no-op). Returns the
    number of serving rows updated."""
    rows = factory.execute(
        "SELECT a.id::text, a.mbid::text, array_agg(t.platform_track_id ORDER BY t.id) AS track_ids "
        "FROM audio_track t JOIN artist a ON a.id = t.artist_id "
        "WHERE t.platform = 'deezer' AND t.platform_track_id IS NOT NULL "
        "GROUP BY a.id, a.mbid" + (" LIMIT %s" if limit else ""),
        ((limit,) if limit else ()),
    ).fetchall()
    n = 0
    for i, (aid, mbid, track_ids) in enumerate(rows, 1):
        payload = build_deezer_top_tracks(track_ids)
        if not payload:
            continue
        js = json.dumps(payload)
        if mbid:
            n += app.execute(
                "UPDATE artists SET deezer_top_tracks = %s::jsonb WHERE mbid = %s", (js, mbid)
            ).rowcount
        else:
            n += app.execute(
                "UPDATE artists SET deezer_top_tracks = %s::jsonb WHERE id = %s", (js, aid)
            ).rowcount
        if i % 1000 == 0:
            app.commit()
    app.commit()
    return n


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="sync analyzed Deezer track ids -> serving deezer_top_tracks (#26)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    app_dsn = os.environ.get("APP_DATABASE_URL")
    if not app_dsn:
        raise SystemExit("APP_DATABASE_URL not set — writing serving is deliberate, no default")
    with psycopg.connect(Settings().database_url) as factory, psycopg.connect(app_dsn) as app:
        n = populate_deezer_embeds(factory, app, args.limit)
        print(f"deezer_top_tracks populated: {n} serving artists")


if __name__ == "__main__":
    main()
