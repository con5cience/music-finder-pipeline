"""Zero-shot tag head (ADR-015 Wave 1): MuQ-MuLan window vectors vs the MB
canonical genre vocabulary.

Vocabulary = mb_raw.genre (2,146 editor-curated canonical names — junk-free by
construction) with mb_raw.genre_alias as the merge map ("synth punk" →
"synth-punk"); scoring uses canonicals only, so variant-merging is solved at
the vocabulary, not patched afterward. Scores are RAW cosines — compressed
(~0.0-0.35) and uncalibrated until the backfill provides per-tag corpus
distributions. Per-track top-K storage (user decision); artist aggregates
derive at publish time.

The MuLan pass shares the embed pass's decoded window files: a SECOND audio
model pass, zero additional downloads (MuQ embeddings can't cosine against
text — ADR-016 chose audio-only for similarity with eyes open).
"""

from __future__ import annotations

import numpy as np
from psycopg import Connection

from pipeline.bench.types import Clip

TAG_TOP_K = 20
TAG_MODEL = "muq-mulan-large"


def load_vocabulary(conn: Connection) -> list[str]:
    """Canonical genre names, stable order (scoring matrix rows align to this)."""
    return [r[0] for r in conn.execute("SELECT name FROM mb_raw.genre ORDER BY id").fetchall()]


def load_alias_map(conn: Connection) -> dict[str, str]:
    """alias → canonical (for merging external/MB artist tags later)."""
    return dict(
        conn.execute(
            "SELECT ga.name, g.name FROM mb_raw.genre_alias ga JOIN mb_raw.genre g ON g.id = ga.genre"
        ).fetchall()
    )


class MulanTagScorer:
    """Embeds the vocabulary once per process; scores window clip files."""

    def __init__(self, vocabulary: list[str]):
        self.vocabulary = vocabulary
        self._embedder = None
        self._vocab_matrix: np.ndarray | None = None

    def _ensure(self) -> None:
        if self._vocab_matrix is not None:
            return
        from pipeline.embedders.registry import get_embedder

        self._embedder = get_embedder(TAG_MODEL)
        self._embedder._ensure()
        self._vocab_matrix = np.asarray(self._embedder.embed_text(self.vocabulary), dtype=np.float32)

    def embed_clips(self, artist_id: str, clip_paths: list[str]) -> np.ndarray:
        """MuLan window vectors for clip files — THE seam every consumer
        (heads, tests) goes through; one pass serves tags + perceptual."""
        self._ensure()
        clips = [Clip(id=f"tag:{i}", artist_id=artist_id, path=p) for i, p in enumerate(clip_paths)]
        return np.asarray(self._embedder.embed(clips), dtype=np.float32)

    def score_clips(self, artist_id: str, clip_paths: list[str]) -> list[tuple[str, float]]:
        """Top-K (tag, score) for the mean audio vector of these window files."""
        return self.score_vectors(self.embed_clips(artist_id, clip_paths))

    def score_vectors(self, vecs, top_k: int = TAG_TOP_K) -> list[tuple[str, float]]:
        """Top-K (tag, score) for the mean of ALREADY-EMBEDDED window vectors
        (shared-vector path: one MuLan pass serves every consumer)."""
        self._ensure()
        mean = np.nan_to_num(np.asarray(vecs, dtype=np.float32)).mean(axis=0)
        mean /= np.linalg.norm(mean) + 1e-9  # NaN window vectors (zero-norm
        # clips) poisoned 37k artist tag rows before this guard
        scores = self._vocab_matrix @ mean
        top = np.argsort(scores)[::-1][:top_k]
        return [(self.vocabulary[i], float(scores[i])) for i in top]


def replace_track_tags(conn: Connection, track_id, tag_scores: list[tuple[str, float]]) -> None:
    """Replace the track's tag set wholesale (review finding: upserting only
    the new top-K left stale rows from prior scoring runs, polluting per-tag
    calibration distributions). One DELETE + one batched INSERT (review
    finding: per-tag round-trips were ~30M avoidable statements at scale)."""
    conn.execute(
        "DELETE FROM track_tag_scores WHERE track_id = %s AND model = %s", (track_id, TAG_MODEL)
    )
    if not tag_scores:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO track_tag_scores (track_id, tag, score, model) VALUES (%s, %s, %s, %s)",
            [(track_id, tag, score, TAG_MODEL) for tag, score in tag_scores],
        )


ARTIST_TAG_TOP_K = 40


def replace_artist_tags(conn: Connection, artist_id, tag_scores: list[tuple[str, float]]) -> None:
    """Artist-level tag set, scored from the ARTIST-MEAN MuLan vector — the
    full-resolution fix for the per-track-truncation pathology (an artist's
    track top-20s can be nearly disjoint; consistent signals died in the
    cut). Wholesale replace, same law as track tags."""
    conn.execute(
        "DELETE FROM artist_tag_scores WHERE artist_id = %s AND model = %s", (artist_id, TAG_MODEL)
    )
    if not tag_scores:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO artist_tag_scores (artist_id, tag, score, model) VALUES (%s, %s, %s, %s)",
            [(artist_id, tag, score, TAG_MODEL) for tag, score in tag_scores],
        )
