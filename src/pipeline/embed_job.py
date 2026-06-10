"""Embed-and-store: the production consumer of the embedder registry (ADR-016).

Pure functions over a psycopg connection so the store path is testable without
Temporal or model weights. The caller owns the transaction. Every stored row is
stamped with the embedder's registry `name` — the version stamp that makes a
future model swap an additive re-embed.

Segmenting: preview sources embed whole files; full-track (windowed)
sources cut RMS-peak 30s windows — see windows.py and the PLATFORMS
descriptor in queues.py.
"""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path

import soundfile as sf
from psycopg import Connection

from pipeline.bench.types import Clip, Embedder
from pipeline.queues import REFRESHERS, WINDOWED_PLATFORMS
from pipeline.windows import peak_windows

_DEFAULT_SEGMENT_S = 30
WINDOWS_PER_TRACK = 4   # 3 tracks x 4 windows ≈ the 10-12 Deezer-preview budget
TRACKS_PER_SOURCE = 3   # full-track sources: floor=3 tracks, newest across releases
MIN_TRACK_S = 60        # skits/intros pollute windows; allowed only when nothing else


class AudioFetchError(RuntimeError):
    """A track's audio could not be materialized (HTTP error, non-audio body)."""


def pending_tracks(conn: Connection, artist_id: str, model: str, source: str | None = None) -> list[tuple]:
    """Embeddable tracks for an artist that this model hasn't embedded yet.

    `source` filters to one platform — the centroid-purity path (ADR-017 §2):
    an artist's embedding clips come from exactly one source at a time.
    Requires an audio_url and a non-rejected/quarantined verification status;
    excludes tracks that already have a row for (track, segment 0, model) so
    re-runs are idempotent.
    """
    return conn.execute(
        """
        SELECT t.id, t.audio_url, t.duration_s, t.platform, t.platform_track_id,
               t.binding_evidence->>'album_path',
               (t.binding_evidence->>'release_index')::int,
               (t.binding_evidence->>'track_index')::int
        FROM audio_track t
        WHERE t.artist_id = %s
          AND (%s::text IS NULL OR t.platform = %s)
          AND t.audio_url IS NOT NULL
          AND t.verification_status NOT IN ('rejected','quarantined')
          AND NOT EXISTS (
              SELECT 1 FROM clip_embedding ce
              WHERE ce.track_id = t.id AND ce.model = %s
          )
        ORDER BY t.discovered_at, t.id
        """,
        (artist_id, source, source, model),
    ).fetchall()


_UA = "music-finder-pipeline/0.1 (wstiern@gmail.com)"


def _audio_ext(head: bytes) -> str | None:
    """Extension from content magic, or None for non-audio (HTML error bodies).

    The extension MATTERS: libsndfile's mp3 detection is extension-gated
    (verified empirically — identical bytes open as .mp3, fail as .audio), so
    downloads must be named for what they contain, never for their URL tail.
    """
    if head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return ".mp3"
    if head.startswith(b"RIFF"):
        return ".wav"
    if head.startswith(b"OggS"):
        return ".ogg"
    if head.startswith(b"fLaC"):
        return ".flac"
    return None


def fetch_audio(url: str, workdir: Path) -> str:
    """Materialize a track's audio locally. http(s) URLs download (real UA,
    status-checked, content-sniffed — signed CDN URLs come in shifting formats
    and some variants reject library UAs); anything else is treated as an
    already-local path (box-local files, tests)."""
    if not url.startswith(("http://", "https://")):
        return url
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — urls from our own audio_track rows
            if resp.status != 200:
                raise AudioFetchError(f"audio fetch HTTP {resp.status}: {url[:120]}")
            body = resp.read()
    except urllib.error.HTTPError as e:  # signed URLs expire → 403 (refreshable upstream)
        raise AudioFetchError(f"audio fetch HTTP {e.code}: {url[:120]}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        # DNS failures, refused connections, socket timeouts — same per-track
        # isolation class as HTTP errors (review finding: these escaped and
        # poisoned the whole artist batch).
        raise AudioFetchError(f"audio fetch failed ({e}): {url[:120]}") from e
    ext = _audio_ext(body[:16])
    if ext is None:
        raise AudioFetchError(f"audio fetch returned non-audio ({body[:40]!r}): {url[:120]}")
    # hash-named: URL tails carry signing junk ('*~data=...') unfit for filenames
    dest = workdir / (hashlib.sha256(url.encode()).hexdigest()[:24] + ext)
    dest.write_bytes(body)
    return str(dest)


def _vec_text(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def refresh_artist_centroid(
    conn: Connection,
    artist_id: str,
    model: str,
    source: str | None = None,
    signal_ratio: float | None = None,
) -> None:
    """Upsert the artist's centroid for one model: re-normalized mean of clips.

    With `source`, only that platform's clips enter the centroid — this is
    where centroid purity is ENFORCED, not just intended: a supersede simply
    re-runs this with the new source and the centroid flips wholesale.
    """
    conn.execute(
        """
        INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, signal_ratio, computed_at)
        SELECT t.artist_id, ce.model, max(ce.dim),
               l2_normalize(avg(ce.embedding)), count(*), %s, now()
        FROM clip_embedding ce
        JOIN audio_track t ON t.id = ce.track_id
        WHERE t.artist_id = %s AND ce.model = %s
          AND (%s::text IS NULL OR t.platform = %s)
        GROUP BY t.artist_id, ce.model
        ON CONFLICT (artist_id, model) DO UPDATE SET
            dim = EXCLUDED.dim,
            embedding = EXCLUDED.embedding,
            clip_count = EXCLUDED.clip_count,
            -- None never wipes a recorded ratio (legacy sourceless calls)
            signal_ratio = COALESCE(EXCLUDED.signal_ratio, artist_embedding.signal_ratio),
            computed_at = EXCLUDED.computed_at
        """,
        (signal_ratio, artist_id, model, source, source),
    )


def _default_refresher(conn: Connection, platform: str, platform_track_id: str) -> str | None:
    """Re-resolve an expired audio URL — dispatch DERIVED from the PLATFORMS
    descriptor (review finding: per-platform if-chains drift as sources land)."""
    import importlib

    dotted = REFRESHERS.get(platform)
    if dotted is None:
        return None
    module, _, fn_name = dotted.partition(":")
    return getattr(importlib.import_module(module), fn_name)(conn, platform_track_id)


def _fetch_with_refresh(conn, url, platform, ptid, workdir, fetch, refresher) -> str | None:
    """One fetch + one refresh-retry; None = skip (track stays pending). THE
    shared self-healing policy — embed and backfill both call this (review
    finding: the duplicated loops had already drifted cosmetically)."""
    try:
        return fetch(url, workdir)
    except AudioFetchError:
        fresh = refresher(conn, platform, ptid) if refresher else None
        if not fresh:
            return None
        try:
            return fetch(fresh, workdir)
        except AudioFetchError:
            return None


def _select_for_source(pending: list[tuple], source: str | None, budget: int = TRACKS_PER_SOURCE) -> list[tuple]:
    """Full-track sources embed up to `budget` tracks, newest across DISTINCT
    releases (walk order from evidence — uuid pks make row order a lottery);
    <60s tracks only when nothing longer exists. Preview sources embed
    everything pending. `budget` already accounts for previously-embedded
    tracks (review finding: re-runs must not grow past TRACKS_PER_SOURCE)."""
    if source not in WINDOWED_PLATFORMS:
        return pending
    if budget <= 0:
        return []
    eligible = [r for r in pending if (r[2] or 0) >= MIN_TRACK_S] or list(pending)
    eligible.sort(key=lambda r: (r[6] is None, r[6], r[7] is None, r[7]))
    chosen: list[tuple] = []
    seen_releases: set = set()
    for row in eligible:  # pass 1: one track per release
        if row[5] not in seen_releases:
            chosen.append(row)
            seen_releases.add(row[5])
        if len(chosen) == budget:
            return chosen
    for row in eligible:  # pass 2: fill from anywhere
        if row not in chosen:
            chosen.append(row)
            if len(chosen) == budget:
                break
    return chosen


def _embedded_track_count(conn: Connection, artist_id: str, model: str, source: str) -> int:
    return conn.execute(
        "SELECT count(DISTINCT ce.track_id) FROM clip_embedding ce "
        "JOIN audio_track t ON t.id = ce.track_id "
        "WHERE t.artist_id = %s AND t.platform = %s AND ce.model = %s",
        (artist_id, source, model),
    ).fetchone()[0]


def _decode(path: str):
    """Decode ONCE per track (ADR-015): every head consumes this waveform."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def _clips_for_track(
    mono, sr: int, path: str, platform: str, duration_s: int | None, workdir: Path, track_key: str
) -> list:
    """(seg_start_s, seg_end_s, clip_path) per clip this track contributes."""
    if platform not in WINDOWED_PLATFORMS:
        return [(0, duration_s or _DEFAULT_SEGMENT_S, path)]
    out = []
    for start_s, end_s in peak_windows(mono, sr, k=WINDOWS_PER_TRACK):
        clip_path = workdir / f"{track_key}-w{start_s}.wav"
        sf.write(clip_path, mono[start_s * sr:end_s * sr], sr)
        out.append((start_s, end_s, str(clip_path)))
    return out


def embed_artist_clips(
    conn: Connection,
    embedder: Embedder,
    artist_id: str,
    source: str | None = None,
    signal_ratio: float | None = None,
    *,
    fetch=fetch_audio,
    refresher=_default_refresher,
    run_analysis: bool = True,
    tag_scorer=None,
) -> int:
    """Embed all pending tracks for an artist and store stamped rows + centroid.

    Per-track isolation: a failed download (signed URLs expire → 403) triggers
    one live URL refresh + retry; still-broken tracks are SKIPPED, never
    poisoning the artist's batch — they stay pending for a later pass. If ALL
    selected tracks fail, raises AudioFetchError so the failure is VISIBLE
    (review finding: silently returning 0 let workflows complete 'embedded'
    with no centroid). A sourceless call adopts artist.embedding_source when
    set (review finding: legacy calls must not blend platforms or wipe the
    signal_ratio). Returns the number of clips embedded.
    """
    if source is None:
        locked = conn.execute(
            "SELECT embedding_source FROM artist WHERE id = %s", (artist_id,)
        ).fetchone()
        if locked and locked[0]:
            source = locked[0]  # purity lock survives sourceless (legacy) calls
    pending = pending_tracks(conn, artist_id, embedder.name, source)
    if not pending:
        # Nothing NEW to embed — but a re-run must still converge metadata:
        # if the source already has clips, restamp centroid/ratio/source so
        # supersede-targeting and publish gating see backfilled values.
        has_clips = conn.execute(
            "SELECT EXISTS (SELECT 1 FROM clip_embedding ce JOIN audio_track t ON t.id = ce.track_id "
            "WHERE t.artist_id = %s AND ce.model = %s AND (%s::text IS NULL OR t.platform = %s))",
            (artist_id, embedder.name, source, source),
        ).fetchone()[0]
        if has_clips:
            refresh_artist_centroid(conn, artist_id, embedder.name, source, signal_ratio)
            if source is not None:
                conn.execute("UPDATE artist SET embedding_source = %s WHERE id = %s", (source, artist_id))
        return 0

    budget = TRACKS_PER_SOURCE
    if source in WINDOWED_PLATFORMS:
        budget -= _embedded_track_count(conn, artist_id, embedder.name, source)
    selected = _select_for_source(pending, source, budget)
    if not selected:
        # budget already spent on a prior run — converge metadata only
        refresh_artist_centroid(conn, artist_id, embedder.name, source, signal_ratio)
        return 0
    embedded = 0
    with tempfile.TemporaryDirectory(prefix="embed-") as tmp:
        workdir = Path(tmp)
        usable: list[tuple] = []  # (track_id, seg_start, seg_end, clip_path)
        track_clip_paths: dict = {}  # track_id -> its window files (for the tag head)
        for tid, url, duration_s, platform, ptid, _release, _ri, _ti in selected:
            path = _fetch_with_refresh(conn, url, platform, ptid, workdir, fetch, refresher)
            if path is None:
                continue  # unfetchable after refresh — skip, stays pending
            # decode ONCE when any head needs the waveform (analysis, windowing);
            # preview tracks with analysis off embed their file untouched
            needs_decode = run_analysis or platform in WINDOWED_PLATFORMS
            mono, native_sr = _decode(path) if needs_decode else (None, None)
            if run_analysis:
                from pipeline.analysis import analyze_track, upsert_track_analysis

                upsert_track_analysis(conn, tid, analyze_track(mono, native_sr))
            for seg in _clips_for_track(mono, native_sr, path, platform, duration_s, workdir, str(tid)):
                usable.append((tid, *seg))
                track_clip_paths.setdefault(tid, []).append(seg[2])
        if not usable:
            raise AudioFetchError(
                f"all {len(selected)} selected tracks failed to fetch for artist {artist_id} "
                f"(source={source}) — tracks remain pending"
            )
        clips = [Clip(id=f"{tid}:{s}", artist_id=artist_id, path=path) for tid, s, _e, path in usable]
        vectors = embedder.embed(clips)

        if tag_scorer is not None:
            from pipeline.tags import replace_track_tags

            for tid, paths in track_clip_paths.items():
                replace_track_tags(conn, tid, tag_scorer.score_clips(artist_id, paths))

        for (tid, seg_start, seg_end, _path), vec in zip(usable, vectors, strict=True):
            conn.execute(
                "INSERT INTO clip_embedding (track_id, segment_start_s, segment_end_s, model, dim, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (tid, seg_start, seg_end, embedder.name, len(vec), _vec_text(vec)),
            )
            embedded += 1
    refresh_artist_centroid(conn, artist_id, embedder.name, source, signal_ratio)
    if source is not None and embedded:
        conn.execute("UPDATE artist SET embedding_source = %s WHERE id = %s", (source, artist_id))
    return embedded
