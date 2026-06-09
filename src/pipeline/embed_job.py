"""Embed-and-store: the production consumer of the embedder registry (ADR-016).

Pure functions over a psycopg connection so the store path is testable without
Temporal or model weights. The caller owns the transaction. Every stored row is
stamped with the embedder's registry `name` — the version stamp that makes a
future model swap an additive re-embed.

Segmenting: one segment per track for now (today's sources are 30s previews).
Multi-segment sampling (25/50/75%, ADR-011 style) lands with full-audio
acquisition; the schema already supports it via segment_start_s.
"""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path

from psycopg import Connection

from pipeline.bench.types import Clip, Embedder

_DEFAULT_SEGMENT_S = 30


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
        SELECT t.id, t.audio_url, t.duration_s, t.platform, t.platform_track_id
        FROM audio_track t
        WHERE t.artist_id = %s
          AND (%s::text IS NULL OR t.platform = %s)
          AND t.audio_url IS NOT NULL
          AND t.verification_status NOT IN ('rejected','quarantined')
          AND NOT EXISTS (
              SELECT 1 FROM clip_embedding ce
              WHERE ce.track_id = t.id AND ce.segment_start_s = 0 AND ce.model = %s
          )
        ORDER BY t.discovered_at
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
            signal_ratio = EXCLUDED.signal_ratio,
            computed_at = EXCLUDED.computed_at
        """,
        (signal_ratio, artist_id, model, source, source),
    )


def _default_refresher(conn: Connection, platform: str, platform_track_id: str) -> str | None:
    """Re-resolve an expired audio URL for platforms that support it."""
    if platform == "deezer":
        from pipeline.sources.deezer import refresh_preview

        return refresh_preview(conn, platform_track_id)
    return None


def embed_artist_clips(
    conn: Connection,
    embedder: Embedder,
    artist_id: str,
    source: str | None = None,
    signal_ratio: float | None = None,
    *,
    fetch=fetch_audio,
    refresher=_default_refresher,
) -> int:
    """Embed all pending tracks for an artist and store stamped rows + centroid.

    Per-track isolation: a failed download (signed URLs expire → 403) triggers
    one live URL refresh + retry; still-broken tracks are SKIPPED, never
    poisoning the artist's batch — they stay pending for a later pass.
    Returns the number of clips embedded (0 = clean no-op, centroid untouched).
    """
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

    embedded = 0
    with tempfile.TemporaryDirectory(prefix="embed-") as tmp:
        workdir = Path(tmp)
        usable: list[tuple[tuple, str]] = []
        for tid, url, duration_s, platform, ptid in pending:
            try:
                path = fetch(url, workdir)
            except AudioFetchError:
                fresh = refresher(conn, platform, ptid) if refresher else None
                if not fresh:
                    continue  # no refresh path — skip, stays pending
                try:
                    path = fetch(fresh, workdir)
                except AudioFetchError:
                    continue  # refreshed URL still broken — skip
            usable.append(((tid, duration_s), path))
        if not usable:
            return 0
        clips = [Clip(id=str(tid), artist_id=artist_id, path=path) for (tid, _), path in usable]
        vectors = embedder.embed(clips)

        for ((tid, duration_s), _path), vec in zip(usable, vectors, strict=True):
            conn.execute(
                "INSERT INTO clip_embedding (track_id, segment_start_s, segment_end_s, model, dim, embedding) "
                "VALUES (%s, 0, %s, %s, %s, %s)",
                (tid, duration_s or _DEFAULT_SEGMENT_S, embedder.name, len(vec), _vec_text(vec)),
            )
            embedded += 1
    refresh_artist_centroid(conn, artist_id, embedder.name, source, signal_ratio)
    if source is not None and embedded:
        conn.execute("UPDATE artist SET embedding_source = %s WHERE id = %s", (source, artist_id))
    return embedded
