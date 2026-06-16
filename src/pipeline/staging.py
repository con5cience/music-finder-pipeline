"""Staged embed pipeline (throughput campaign): evict everything that isn't
inference from the GPU lane.

prep_artist (CPU `prep` queue, high concurrency): selection → fetch (self-
healing) → decode once → CPU analysis head → RMS windows → clip wavs + a
manifest on the SHARED stage volume. embed_staged (gpu queue): pure model
passes — MuQ over staged clips, shared-MuLan heads, artist tag pass,
centroid — then deletes the stage dir. GPU wall-time per artist drops from
~22s (fetch-dominated) to the inference seconds.

Failure semantics: prep is idempotent (re-stages wholesale); embed retries
reuse the staged dir; a MISSING manifest (volume wiped, prep skew) falls
back to the legacy single-pass path rather than failing the artist. Clip
inserts are ON CONFLICT DO NOTHING against the (track, segment, model)
unique — retry-safe. clean_stale_stage GCs orphans (>24h).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from psycopg import Connection

from pipeline.embed_job import (
    PREVIEW_TRACKS_CAP,
    TRACKS_PER_SOURCE,
    WINDOWED_PLATFORMS,
    _clips_for_track,
    _decode,
    _default_refresher,
    _embedded_track_count,
    _fetch_with_refresh,
    _select_for_source,
    _try_archive,
    _vec_text,
    fetch_audio,
    pending_tracks,
    refresh_artist_centroid,
)


def stage_root() -> Path:
    return Path(os.environ.get("PIPELINE_STAGE_DIR", "/tmp/pipeline-stage"))


def _artist_dir(artist_id: str) -> Path:
    return stage_root() / str(artist_id)


def prep_artist(
    conn: Connection,
    artist_id: str,
    source: str | None,
    model: str,
    *,
    fetch=fetch_audio,
    refresher=_default_refresher,
) -> int:
    """Stage an artist's selected tracks: clips on disk + manifest + CPU
    analysis. Returns tracks staged (0 = nothing pending/budget spent)."""
    from pipeline.heads import CpuAnalysisHead, HeadContext, run_heads

    adir = _artist_dir(artist_id)
    if adir.exists():
        shutil.rmtree(adir)  # idempotent re-prep — ALSO clears stale dirs from
        # terminated prior workflows even when this run stages nothing (review
        # finding: early returns left year-old manifests for embed to trust)
    pending = pending_tracks(conn, artist_id, model, source)
    if not pending:
        return 0
    budget = TRACKS_PER_SOURCE if source in WINDOWED_PLATFORMS else PREVIEW_TRACKS_CAP
    if source is not None:
        budget -= _embedded_track_count(conn, artist_id, model, source)
    selected = _select_for_source(pending, source, budget)
    if not selected:
        return 0

    adir.mkdir(parents=True)
    cpu_head = [CpuAnalysisHead()]
    manifest: list[dict] = []

    # PARALLEL fetch (measured: 2.15s proxy round-trip x 12 serial fetches
    # was prep's whole bottleneck — GPU starved at 0%). Plain fetches pool;
    # failures retry SERIALLY through the refresher (it shares the DB conn,
    # which is not thread-safe). Order keyed by index → manifest order stays
    # the selection order.
    from concurrent.futures import ThreadPoolExecutor

    def _plain(args):
        i, url = args
        try:
            return i, fetch(url, adir)
        except Exception:  # noqa: BLE001 — retried via refresher below
            return i, None

    with ThreadPoolExecutor(max_workers=6) as pool:
        fetched = dict(pool.map(_plain, [(i, row[1]) for i, row in enumerate(selected)]))

    for i, (tid, url, duration_s, platform, ptid, _release, _ri, _ti) in enumerate(selected):
        path = fetched.get(i)
        if path is None:  # pooled fetch failed → serial self-healing path
            path = _fetch_with_refresh(conn, url, platform, ptid, adir, fetch, refresher)
        if path is None:
            continue  # stays pending; embed_staged works with what staged
        mono, sr = _decode(path)
        run_heads(conn, cpu_head, HeadContext(
            conn=conn, track_id=tid, artist_id=artist_id, platform=platform,
            mono=mono, sr=sr,
        ))
        try:  # fingerprint rides the already-paid decode; never blocks staging
            from pipeline.fingerprint import store_fingerprint

            store_fingerprint(conn, tid, mono, sr)
        except Exception:  # noqa: BLE001 — flag-only feature, prep must not fail on it
            pass
        segs = _clips_for_track(mono, sr, path, platform, duration_s, adir, str(tid))
        # ADR-021 Tier B: archive a compressed copy of the embedded window clips
        # before this stage dir is GC'd, so a model swap / re-window re-embeds locally.
        _try_archive(conn, str(artist_id), str(tid), platform, segs)
        manifest.append({
            "track_id": str(tid),
            "platform": platform,
            "segs": [[s, e, str(Path(p).name)] for s, e, p in segs],
        })
    if not manifest:
        shutil.rmtree(adir, ignore_errors=True)
        return 0
    (adir / "manifest.json").write_text(
        json.dumps({"model": model, "source": source, "tracks": manifest})
    )
    return len(manifest)


def embed_staged(
    conn: Connection,
    embedder,
    artist_id: str,
    source: str | None,
    signal_ratio: float | None,
    heads: list | None = None,
) -> int:
    """Pure-inference embed from the staged manifest. Falls back to the
    legacy single-pass path when no stage exists (volume wiped, old runs)."""
    from pipeline.bench.types import Clip

    adir = _artist_dir(artist_id)
    mpath = adir / "manifest.json"
    stale = mpath.exists() and json.loads(mpath.read_text()).get("source") != source
    if stale:
        shutil.rmtree(adir, ignore_errors=True)  # source flipped since prep
    if stale or not mpath.exists():
        from pipeline.embed_job import embed_artist_clips

        return embed_artist_clips(conn, embedder, artist_id, source, signal_ratio, heads=heads)

    manifest = json.loads(mpath.read_text())
    usable: list[tuple] = []
    for t in manifest["tracks"]:
        for s, e, fname in t["segs"]:
            p = adir / fname
            # fault isolation (review finding): one truncated/empty clip must
            # not fail the whole artist on every retry
            if p.exists() and p.stat().st_size > 1024:
                usable.append((t["track_id"], s, e, str(p)))
    if not usable:
        shutil.rmtree(adir, ignore_errors=True)
        return 0

    clips = [Clip(id=f"{tid}:{s}", artist_id=artist_id, path=p) for tid, s, _e, p in usable]
    vectors = embedder.embed(clips)

    embedded = 0
    for (tid, seg_start, seg_end, _p), vec in zip(usable, vectors, strict=True):
        conn.execute(
            "INSERT INTO clip_embedding (track_id, segment_start_s, segment_end_s, model, dim, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (track_id, segment_start_s, model) DO NOTHING",
            (tid, seg_start, seg_end, embedder.name, len(vec), _vec_text(vec)),
        )
        embedded += 1

    if heads:
        from pipeline.heads import HeadContext, artist_tag_pass, run_heads

        artist_vecs = []
        for t in manifest["tracks"]:
            ctx = HeadContext(
                conn=conn, track_id=t["track_id"], artist_id=artist_id,
                platform=t["platform"], mono=None, sr=0,
                clip_paths=[str(adir / f) for _s, _e, f in t["segs"]],
            )
            run_heads(conn, heads, ctx)  # cpu head already ledgered by prep
            artist_vecs.append(ctx.mulan_vecs)
        artist_tag_pass(conn, heads, artist_id, artist_vecs)

    refresh_artist_centroid(conn, artist_id, embedder.name, source, signal_ratio)
    if source is not None and embedded:
        conn.execute("UPDATE artist SET embedding_source = %s WHERE id = %s", (source, artist_id))
    # NO cleanup here (review finding): the stage dir outlives this function so
    # the CALLER deletes it only after commit — a post-return commit failure
    # would otherwise orphan the DB rollback from the disk state and force the
    # retry onto the slow legacy path. cleanup_artist_dir is the companion.
    return embedded


def cleanup_artist_dir(artist_id: str) -> None:
    shutil.rmtree(_artist_dir(artist_id), ignore_errors=True)


def clean_stale_stage(max_age_hours: float = 48.0) -> int:
    """GC stage dirs that are TRULY orphaned. Age alone is not orphanhood
    (outage post-mortem: a deep backlog stretched prep→embed gaps to hours;
    a 6h age-only GC deleted QUEUED work under a live reader and wedged the
    GPU lane for 6h). A dir with a manifest is awaiting embed — eligible
    only after max_age_hours (default 48h, generous vs any backlog). A dir
    WITHOUT a manifest is an incomplete prep — eligible after 6h."""
    root = stage_root()
    if not root.exists():
        return 0
    now = time.time()
    removed = 0
    for d in root.iterdir():
        try:  # concurrent deletion by the gpu worker is normal, not an error
            if not d.is_dir():
                continue
            age_h = (now - d.stat().st_mtime) / 3600
            has_manifest = (d / "manifest.json").exists()
            if (has_manifest and age_h > max_age_hours) or (not has_manifest and age_h > 6):
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed
