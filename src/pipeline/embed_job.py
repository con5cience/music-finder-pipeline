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

import tempfile
import urllib.request
from pathlib import Path

from psycopg import Connection

from pipeline.bench.types import Clip, Embedder

_DEFAULT_SEGMENT_S = 30


def pending_tracks(conn: Connection, artist_id: str, model: str) -> list[tuple]:
    """Embeddable tracks for an artist that this model hasn't embedded yet.

    Requires an audio_url and a non-rejected/quarantined verification status;
    excludes tracks that already have a row for (track, segment 0, model) so
    re-runs are idempotent.
    """
    return conn.execute(
        """
        SELECT t.id, t.audio_url, t.duration_s
        FROM audio_track t
        WHERE t.artist_id = %s
          AND t.audio_url IS NOT NULL
          AND t.verification_status NOT IN ('rejected','quarantined')
          AND NOT EXISTS (
              SELECT 1 FROM clip_embedding ce
              WHERE ce.track_id = t.id AND ce.segment_start_s = 0 AND ce.model = %s
          )
        ORDER BY t.discovered_at
        """,
        (artist_id, model),
    ).fetchall()


def fetch_audio(url: str, workdir: Path) -> str:
    """Materialize a track's audio locally. http(s) URLs download; anything else
    is treated as an already-local path (box-local files, tests)."""
    if url.startswith(("http://", "https://")):
        dest = workdir / Path(url).name.split("?")[0]
        urllib.request.urlretrieve(url, dest)  # noqa: S310 — urls come from our own audio_track rows
        return str(dest)
    return url


def _vec_text(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def refresh_artist_centroid(conn: Connection, artist_id: str, model: str) -> None:
    """Upsert the artist's centroid for one model: re-normalized mean of clips."""
    conn.execute(
        """
        INSERT INTO artist_embedding (artist_id, model, dim, embedding, clip_count, computed_at)
        SELECT t.artist_id, ce.model, max(ce.dim),
               l2_normalize(avg(ce.embedding)), count(*), now()
        FROM clip_embedding ce
        JOIN audio_track t ON t.id = ce.track_id
        WHERE t.artist_id = %s AND ce.model = %s
        GROUP BY t.artist_id, ce.model
        ON CONFLICT (artist_id, model) DO UPDATE SET
            dim = EXCLUDED.dim,
            embedding = EXCLUDED.embedding,
            clip_count = EXCLUDED.clip_count,
            computed_at = EXCLUDED.computed_at
        """,
        (artist_id, model),
    )


def embed_artist_clips(conn: Connection, embedder: Embedder, artist_id: str) -> int:
    """Embed all pending tracks for an artist and store stamped rows + centroid.

    Returns the number of clips embedded (0 = clean no-op, centroid untouched).
    """
    pending = pending_tracks(conn, artist_id, embedder.name)
    if not pending:
        return 0

    with tempfile.TemporaryDirectory(prefix="embed-") as tmp:
        workdir = Path(tmp)
        clips = [Clip(id=str(tid), artist_id=artist_id, path=fetch_audio(url, workdir)) for tid, url, _ in pending]
        vectors = embedder.embed(clips)

    for (tid, _url, duration_s), vec in zip(pending, vectors, strict=True):
        conn.execute(
            "INSERT INTO clip_embedding (track_id, segment_start_s, segment_end_s, model, dim, embedding) "
            "VALUES (%s, 0, %s, %s, %s, %s)",
            (tid, duration_s or _DEFAULT_SEGMENT_S, embedder.name, len(vec), _vec_text(vec)),
        )
    refresh_artist_centroid(conn, artist_id, embedder.name)
    return len(pending)
