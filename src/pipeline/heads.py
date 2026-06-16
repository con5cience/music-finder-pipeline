"""ADR-015 AnalysisHead protocol — decode-once, N heads, per-head idempotency.

Each head consumes the SAME decoded waveform / window files and records its
completion in track_head_runs (track, head → version). The embed pass and the
backfill iterate the same HEADS list and skip per-head where the current
version already ran — adding a head means adding ONE list entry; old corpus
rows pick it up on the next backfill sweep automatically.

Wave-2 perceptual axes are MuLan zero-shot ANCHOR PAIRS: score = cos(audio,
positive) − cos(audio, negative). Raw and uncalibrated by design — corpus
distributions calibrate presentation later (same posture as tag scores).
Research-grade estimates, not ground truth; stored as signals, not verdicts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from psycopg import Connection

log = logging.getLogger(__name__)

PERCEPTUAL_MODEL = "muq-mulan-large"

# axis → (positive anchor, negative anchor)
AXIS_ANCHORS: dict[str, tuple[str, str]] = {
    "danceability": (
        "danceable dance music with a strong steady beat",
        "music without rhythm that is impossible to dance to",
    ),
    "valence": (
        "happy uplifting cheerful joyful music",
        "sad melancholic dark depressing music",
    ),
    "arousal": (
        "energetic intense loud aggressive music",
        "calm gentle quiet soothing music",
    ),
    "speechiness": (
        "spoken word, talking, speech recording",
        "instrumental music with no spoken words",
    ),
    "liveness": (
        "live concert recording with audience and room ambience",
        "clean polished studio recording",
    ),
    "vocalness": (
        "music with prominent vocals and singing",
        "instrumental music without any vocals",
    ),
}

INSTRUMENT_VOCAB = [
    "electric guitar", "acoustic guitar", "piano", "synthesizer", "violin",
    "cello", "drums", "drum machine", "bass guitar", "double bass", "trumpet",
    "saxophone", "flute", "clarinet", "organ", "accordion", "harp", "banjo",
    "mandolin", "sitar", "turntables and scratching", "choir", "strings section",
    "brass section", "harmonica", "marimba", "steel drums", "bagpipes",
]
INSTRUMENT_TOP_K = 5


@dataclass
class HeadContext:
    conn: Connection
    track_id: object
    artist_id: str
    platform: str
    mono: np.ndarray
    sr: int
    clip_paths: list[str] = field(default_factory=list)
    # MuLan window vectors, computed ONCE per track and shared by every
    # MuLan-consuming head (tags + perceptual EACH ran their own identical
    # pass before — a straight 2x GPU waste, found in the throughput audit).
    mulan_vecs: np.ndarray | None = None


def ensure_mulan_vecs(ctx: HeadContext, scorer) -> np.ndarray | None:
    if ctx.mulan_vecs is not None or scorer is None or not ctx.clip_paths:
        return ctx.mulan_vecs
    ctx.mulan_vecs = scorer.embed_clips(ctx.artist_id, ctx.clip_paths)
    return ctx.mulan_vecs


class CpuAnalysisHead:
    """Wave 1: integrity, MIR, fingerprint (analysis.py)."""

    name = "cpu_analysis"
    version = 1

    def run(self, ctx: HeadContext) -> None:
        from pipeline.analysis import analyze_track, upsert_track_analysis

        upsert_track_analysis(ctx.conn, ctx.track_id, analyze_track(ctx.mono, ctx.sr))


class TagHead:
    """Wave 1 (v2): zero-shot genres vs the MB vocabulary — from the SHARED
    MuLan window vectors (no second embed pass)."""

    name = "tags"
    version = 2  # v2: shared vectors; artist-level scoring added alongside

    def __init__(self, scorer):
        self._scorer = scorer

    def run(self, ctx: HeadContext) -> bool:
        vecs = ensure_mulan_vecs(ctx, self._scorer)
        if vecs is None:
            return False  # no scorer/clips — must NOT be ledgered (review finding)
        from pipeline.tags import replace_track_tags

        replace_track_tags(ctx.conn, ctx.track_id, self._scorer.score_vectors(vecs))
        return True


class PerceptualHead:
    """Wave 2: anchor-pair axes + multi-label instruments (MuLan zero-shot)."""

    name = "perceptual"
    version = 2  # v2: shared vectors

    def __init__(self, scorer):
        self._scorer = scorer  # shares the TagHead's MulanTagScorer embedder

    def _anchor_matrix(self):
        emb = self._scorer._embedder
        if getattr(self, "_anchors", None) is None:
            texts = [t for pair in AXIS_ANCHORS.values() for t in pair] + INSTRUMENT_VOCAB
            self._anchors = np.asarray(emb.embed_text(texts), dtype=np.float32)
        return self._anchors

    def run(self, ctx: HeadContext) -> bool:
        vecs = ensure_mulan_vecs(ctx, self._scorer)
        if vecs is None:
            return False  # no scorer/clips — must NOT be ledgered (review finding)
        import json

        mean = np.nan_to_num(vecs.mean(axis=0))
        mean /= np.linalg.norm(mean) + 1e-9
        anchors = self._anchor_matrix()
        # NaN guard (maintenance-window incident: one silent/zero-norm track
        # produced NaN scores; json rejects NaN and the whole batch aborted)
        scores = np.nan_to_num(anchors @ mean)

        n_axes = len(AXIS_ANCHORS)
        axis_vals = {}
        for i, axis in enumerate(AXIS_ANCHORS):
            axis_vals[axis] = float(scores[2 * i] - scores[2 * i + 1])
        inst_scores = scores[2 * n_axes:]
        top = np.argsort(inst_scores)[::-1][:INSTRUMENT_TOP_K]
        instruments = [
            {"name": INSTRUMENT_VOCAB[i], "score": round(float(inst_scores[i]), 4)} for i in top
        ]
        ctx.conn.execute(
            """
            INSERT INTO track_perceptual (track_id, danceability, valence, arousal,
                speechiness, liveness, vocalness, instruments, model, computed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (track_id) DO UPDATE SET
                danceability = EXCLUDED.danceability, valence = EXCLUDED.valence,
                arousal = EXCLUDED.arousal, speechiness = EXCLUDED.speechiness,
                liveness = EXCLUDED.liveness, vocalness = EXCLUDED.vocalness,
                instruments = EXCLUDED.instruments, model = EXCLUDED.model, computed_at = now()
            """,
            (ctx.track_id, axis_vals["danceability"], axis_vals["valence"], axis_vals["arousal"],
             axis_vals["speechiness"], axis_vals["liveness"], axis_vals["vocalness"],
             json.dumps(instruments), PERCEPTUAL_MODEL),
        )


def build_heads(tag_scorer) -> list:
    """The canonical head list — embed pass and backfill both use this."""
    return [CpuAnalysisHead(), TagHead(tag_scorer), PerceptualHead(tag_scorer)]


def pending_heads(conn: Connection, track_id, heads: list) -> list:
    """Heads whose current version hasn't run for this track."""
    ran = dict(
        conn.execute(
            "SELECT head, version FROM track_head_runs WHERE track_id = %s", (track_id,)
        ).fetchall()
    )
    return [h for h in heads if ran.get(h.name, 0) < h.version]


def artist_tag_pass(conn: Connection, heads: list, artist_id: str, vecs_list: list) -> None:
    """After per-track heads: artist tags from the ARTIST-MEAN MuLan vector
    over every window of every selected track (full resolution — the fix for
    per-track-truncation pathology)."""
    scorer = next((h._scorer for h in heads if isinstance(h, TagHead) and h._scorer), None)
    vecs = [v for v in vecs_list if v is not None]
    if scorer is None or not vecs:
        return
    from pipeline.tags import ARTIST_TAG_TOP_K, replace_artist_tags

    stacked = np.concatenate(vecs, axis=0)
    # ADR-021 Tier A: stash the vectors before scoring. Best-effort — a savepoint
    # keeps a persist failure from poisoning the (far more valuable) embed
    # transaction; the embed proceeds even if the stash fails.
    try:
        with conn.transaction():
            persist_analysis_vectors(conn, artist_id, stacked)
    except Exception:
        log.exception("analysis-vector persist failed for artist %s", artist_id)
    replace_artist_tags(conn, artist_id, scorer.score_vectors(stacked, ARTIST_TAG_TOP_K))


def persist_analysis_vectors(conn: Connection, artist_id: str, stacked: np.ndarray) -> None:
    """ADR-021 Tier A: store the MuLan per-window vectors + the artist-mean (the
    exact vector scoring uses) so any future corpus re-analysis — re-score,
    score centering, calibration/vocabulary changes, new analysis heads,
    re-aggregation/window-weighting — becomes a math pass over these stored
    vectors instead of a re-fetch + re-decode of the whole corpus.

    The stored vectors are the raw MuLan AUDIO embeddings, vocabulary-independent,
    stamped with model + window-config version. Idempotent: replaces this
    artist+model's rows wholesale (the window count can change between runs, so an
    upsert would leave stale high-idx rows)."""
    from pipeline.embed_job import _vec_text
    from pipeline.tags import TAG_MODEL
    from pipeline.windows import WINDOW_VERSION

    vecs = np.nan_to_num(np.asarray(stacked, dtype=np.float32))
    if vecs.ndim != 2 or vecs.shape[0] == 0:
        return
    mean = vecs.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-9)  # identical to score_vectors' mean
    dim = int(vecs.shape[1])
    conn.execute(
        "DELETE FROM artist_analysis_vector WHERE artist_id = %s AND model = %s",
        (artist_id, TAG_MODEL),
    )
    rows = [(artist_id, TAG_MODEL, "mean", 0, dim, _vec_text(mean), WINDOW_VERSION)]
    rows += [
        (artist_id, TAG_MODEL, "window", i, dim, _vec_text(vecs[i]), WINDOW_VERSION)
        for i in range(vecs.shape[0])
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO artist_analysis_vector "
            "(artist_id, model, kind, idx, dim, embedding, window_version) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            rows,
        )


def run_heads(conn: Connection, heads: list, ctx: HeadContext) -> int:
    """Run heads pending for this track; record completions. Returns count run."""
    todo = pending_heads(conn, ctx.track_id, heads)
    ran = 0
    for h in todo:
        # A head returning False explicitly DECLINED (scorer/vectors absent —
        # vocabulary not bootstrapped yet): never ledger a no-op, or the skip
        # becomes permanent (review finding). None (cpu head) = ran.
        if h.run(ctx) is False:
            continue
        ran += 1
        conn.execute(
            """
            INSERT INTO track_head_runs (track_id, head, version, computed_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (track_id, head) DO UPDATE SET version = EXCLUDED.version, computed_at = now()
            """,
            (ctx.track_id, h.name, h.version),
        )
    return ran
