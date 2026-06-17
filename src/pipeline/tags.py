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


CENTERING_MU_VERSION = "muq-mulan-v1"


def refresh_centering(conn: Connection, scorer, sample: int = 20000) -> int:
    """ADR-020 Phase 5: per-tag d_i = (genre text embedding) . (corpus-mean audio
    direction), computed from the stored MuLan artist-mean vectors. Read at
    publish to demote anisotropy-aligned (scattered/magnet) tags via
    `score - C*d_i`. Needs the model (vocab text embeddings) so it runs offline
    here, NOT at publish (publish-sync carries no model). Returns tags written."""
    scorer._ensure()
    V = np.asarray(scorer._vocab_matrix, dtype=np.float32)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    rows = conn.execute(
        "SELECT embedding::text FROM artist_analysis_vector WHERE kind = 'mean' "
        "ORDER BY random() LIMIT %s",
        (sample,),
    ).fetchall()
    if not rows:
        return 0
    M = np.stack([
        np.array([float(x) for x in r[0].strip("[]").split(",")], dtype=np.float32) for r in rows
    ])
    mu = M.mean(axis=0)
    mu = mu / (np.linalg.norm(mu) + 1e-9)
    d = V @ mu  # per-tag projection onto the dominant audio direction
    vocab = scorer.vocabulary
    with conn.cursor() as cur:
        cur.execute("DELETE FROM tag_centering WHERE model = %s", (TAG_MODEL,))
        cur.executemany(
            "INSERT INTO tag_centering (tag, model, d, mu_version, n_sample) "
            "VALUES (%s, %s, %s, %s, %s)",
            [(vocab[i], TAG_MODEL, float(d[i]), CENTERING_MU_VERSION, len(rows))
             for i in range(len(vocab))],
        )
    return len(vocab)


def load_centering(conn: Connection, model: str = TAG_MODEL) -> dict[str, float]:
    """{tag: d_i} for publish-time centering; empty when not computed yet, in
    which case publish falls back to the legacy z-score ranking."""
    return {
        t: float(d)
        for t, d in conn.execute(
            "SELECT tag, d FROM tag_centering WHERE model = %s", (model,)
        ).fetchall()
    }


def refresh_audio_blocklist(conn: Connection, audio_min: float = 0.08, mb_max: int = 100) -> int:
    """ADR-020 Phase 4: the data-driven magnet prune. A tag the audio model assigns
    to > audio_min of artists but MB editors apply to < mb_max artists is an
    anisotropy artifact (kilapanga, orthodox pop, pumpcore, fm synthesis, j-rock,
    teen pop, ...). Stored in tag_audio_blocklist; excluded from the AUDIO tag tier
    at publish (MB/Bandcamp tiers untouched). Pure SQL — no model needed."""
    conn.execute("DELETE FROM tag_audio_blocklist")
    conn.execute(
        """
        INSERT INTO tag_audio_blocklist (tag, audio_pct, mb_n)
        WITH au AS (
            SELECT tag, count(DISTINCT artist_id)::float
                   / NULLIF((SELECT count(DISTINCT artist_id) FROM artist_tag_scores), 0) AS r
            FROM artist_tag_scores GROUP BY tag
        ),
        mb AS (
            SELECT coalesce(gc.name, gd.name) AS tag, count(DISTINCT at.artist) AS n
            FROM mb_raw.artist_tag at JOIN mb_raw.tag t ON t.id = at.tag
            LEFT JOIN mb_raw.genre gd ON gd.name = lower(t.name)
            LEFT JOIN mb_raw.genre_alias gal ON gal.name = lower(t.name)
            LEFT JOIN mb_raw.genre gc ON gc.id = gal.genre
            WHERE at.count > 0 AND (gd.name IS NOT NULL OR gc.name IS NOT NULL)
            GROUP BY 1
        )
        SELECT au.tag, (au.r * 100)::real, coalesce(mb.n, 0)
        FROM au LEFT JOIN mb ON mb.tag = au.tag
        WHERE au.r > %s AND coalesce(mb.n, 0) < %s
        """,
        (audio_min, mb_max),
    )
    return conn.execute("SELECT count(*) FROM tag_audio_blocklist").fetchone()[0]
