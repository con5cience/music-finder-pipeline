"""AI-vs-human linear probe on MuQ embeddings (2026-06-12 experiment).

The 1-day question before committing to a spectrogram CNN: do MuQ's
features already separate AI-generated audio from human recordings? If a
linear probe on embeddings we compute anyway reaches high AUC, the
production ai_likelihood head is nearly free; if not, budget the CNN.

  Human side: stored clip vectors for artists whose MB begin_date_year
  <= 2015 — acts that provably predate music generators.
  AI side: a directory of known-AI clips (research sets / generator
  scrapes), embedded fp32 through the same registry.

Pure numpy logistic regression — no new deps; report AUC on a holdout.

Run:  uv run poe ai-probe -- --ai-dir /path/to/ai-clips
"""

from __future__ import annotations

import numpy as np
from psycopg import Connection

DEFAULT_MODEL = "muq-large-msd"
HUMAN_CUTOFF_YEAR = 2016  # begin_date strictly before this = pre-generator


def sample_human_vectors(conn: Connection, n: int = 4000,
                         *, model: str = DEFAULT_MODEL) -> np.ndarray:
    rows = conn.execute(
        """
        SELECT ce.embedding::text
        FROM clip_embedding ce
        JOIN audio_track t ON t.id = ce.track_id
        JOIN artist a ON a.id = t.artist_id
        JOIN mb_raw.artist ma ON ma.gid = a.mbid
        WHERE ce.model = %s AND ma.begin_date_year IS NOT NULL
          AND ma.begin_date_year < %s
        ORDER BY random() LIMIT %s
        """,
        (model, HUMAN_CUTOFF_YEAR, n),
    ).fetchall()
    import json

    return np.asarray([json.loads(r[0]) for r in rows], dtype=np.float32)


def logistic_probe(x_a: np.ndarray, x_b: np.ndarray, *, holdout: float = 0.25,
                   epochs: int = 300, lr: float = 0.1, seed: int = 5) -> dict:
    """Train a logistic probe (a=1, b=0) and report holdout AUC + accuracy."""
    rng = np.random.default_rng(seed)
    x = np.vstack([x_a, x_b]).astype(np.float64)
    y = np.concatenate([np.ones(len(x_a)), np.zeros(len(x_b))])
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]
    n_hold = int(len(x) * holdout)
    x_tr, y_tr, x_te, y_te = x[n_hold:], y[n_hold:], x[:n_hold], y[:n_hold]
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-9
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    w = np.zeros(x.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(x_tr @ w + b)))
        g = p - y_tr
        w -= lr * (x_tr.T @ g / len(x_tr) + 1e-4 * w)
        b -= lr * g.mean()
    score = x_te @ w + b
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y_te == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)
    acc = float((((score > 0) == y_te.astype(bool)).mean()))
    return {"auc": round(float(auc), 4), "holdout_acc": round(acc, 4),
            "n_train": len(x_tr), "n_holdout": len(x_te)}


def embed_dir(ai_dir: str, *, max_clips: int = 4000, chunk: int = 48) -> np.ndarray:
    """Embed AI audio as CENTERED 30s clips, in chunks.

    Two reasons this is not one big emb.embed() call: (1) SONICS entries are
    full-length songs — decoding 4000 of them to waveforms at once OOM-killed
    the process (the silent kill on the first run); (2) the human side is 30s
    clip embeddings, so for an apples-to-apples probe the AI side must be 30s
    clips too, not full tracks. _center_clip (the adjudicator's) trims + the
    chunk loop bounds peak host RAM to `chunk` waveforms."""
    import os
    import tempfile

    os.environ.setdefault("PIPELINE_FP16", "0")  # the 30s NaN law
    from pathlib import Path

    from pipeline.adjudicate import _center_clip
    from pipeline.bench.types import Clip
    from pipeline.embedders.registry import get_embedder

    paths = [p for p in sorted(Path(ai_dir).rglob("*"))
             if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg")][:max_clips]
    emb = get_embedder()
    out: list[np.ndarray] = []
    with tempfile.TemporaryDirectory(prefix="ai-probe-") as tmp:
        wd = Path(tmp)
        for i in range(0, len(paths), chunk):
            batch = []
            for j, p in enumerate(paths[i:i + chunk]):
                clip = _center_clip(str(p), wd)
                if clip:
                    batch.append(Clip(id=f"{i}-{j}", artist_id="ai", path=clip))
            if not batch:
                continue
            v = np.asarray(emb.embed(batch), dtype=np.float32)
            out.append(v[np.isfinite(v).all(axis=1)])
            for c in batch:  # don't let trimmed wavs accumulate across chunks
                try:
                    os.unlink(c.path)
                except OSError:
                    pass
    return np.vstack(out) if out else np.empty((0, 1024), dtype=np.float32)


def main() -> None:
    import argparse

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="AI-vs-human linear probe on MuQ embeddings")
    ap.add_argument("--ai-dir", required=True, help="directory of known-AI audio clips")
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()
    with psycopg.connect(Settings().database_url) as conn:
        human = sample_human_vectors(conn, args.n)
    print(f"human vectors: {human.shape}")
    ai = embed_dir(args.ai_dir, max_clips=args.n)
    print(f"ai vectors: {ai.shape}")
    out = logistic_probe(ai, human)
    print(out)
    print("verdict:", "MuQ separates — head is a linear layer" if out["auc"] >= 0.95
          else "weak separation — budget the spectrogram CNN")


if __name__ == "__main__":
    main()
