"""Wave-3 heads — ADR-015 law: heavy, GATED on demonstrated value, never
blanket-applied. These heads are NOT in build_heads(); they run only via
`poe wave3 -- --limit N [--artist ID]` on a selected slice, so demonstrated
value (or its absence) is cheap to establish.

Implemented: StructureHead (librosa laplacian segmentation — n_sections,
avg length, repetition ratio; dependency-free).
Registered-not-implemented (each needs a gigabyte-class dependency the
operator must bless): demucs_stems (source separation → stem similarity),
asr_lyrics (language/lyrics via whisper-family).
"""

from __future__ import annotations

import numpy as np
from psycopg import Connection

WAVE3_AVAILABLE = ["structure", "stems", "asr"]


class StructureHead:
    """Song-structure segmentation: laplacian spectral clustering over a
    chroma/MFCC recurrence (librosa cookbook method), summarized to scalars
    the product can use (repetition_ratio: how much of the song repeats)."""

    name = "structure"
    version = 1

    def run(self, conn: Connection, track_id, mono: np.ndarray, sr: int) -> bool:
        import librosa

        if mono.size < sr * 20:  # too short to segment meaningfully
            return False
        y = mono.astype(np.float32)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
        rec = librosa.segment.recurrence_matrix(chroma, width=3, mode="affinity", sym=True)
        # laplacian eigenvectors → k-segment boundaries (k by eigengap, capped)
        from scipy.ndimage import median_filter

        rec = median_filter(rec, size=(1, 7))
        deg = np.sum(rec, axis=1)
        lap = np.diag(deg) - rec
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
        evals, evecs = np.linalg.eigh(norm @ lap @ norm)
        gaps = np.diff(evals[:10])
        k = int(np.clip(np.argmax(gaps[1:]) + 2, 2, 8))
        x = evecs[:, :k]
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        # simple k-means (few points, few clusters — no sklearn dep)
        rng = np.random.default_rng(0)
        centers = x[rng.choice(len(x), k, replace=False)]
        for _ in range(20):
            labels = np.argmax(x @ centers.T, axis=1)
            for c in range(k):
                pts = x[labels == c]
                if len(pts):
                    centers[c] = pts.mean(axis=0)
                    centers[c] /= np.linalg.norm(centers[c]) + 1e-9
        # temporal smoothing: a section must persist ~3s, else frame-level
        # label flicker manifests as hundreds of phantom boundaries
        labels = median_filter(labels, size=31)
        bounds = list(np.flatnonzero(np.diff(labels)) + 1)
        # constraint pass: sections >= ~4s, at most 8 — merge by dropping the
        # boundary bordering the shortest section until both hold
        fps = sr / 2048
        min_frames = int(4 * fps)
        while bounds:
            edges = [0, *bounds, len(labels)]
            lengths = np.diff(edges)
            if len(lengths) <= 8 and lengths.min() >= min_frames:
                break
            shortest = int(np.argmin(lengths))
            drop = shortest - 1 if shortest == len(lengths) - 1 else shortest
            bounds.pop(max(drop, 0))
        bounds = np.array(bounds, dtype=int)
        times = librosa.frames_to_time(bounds, sr=sr, hop_length=2048)
        n_sections = len(bounds) + 1
        total_s = mono.size / sr
        # repetition: fraction of frames whose label occurs in >1 section
        section_labels = np.split(labels, bounds)
        label_section_count: dict[int, int] = {}
        for seg in section_labels:
            if len(seg):
                label_section_count[seg[0]] = label_section_count.get(seg[0], 0) + 1
        repeated = sum(len(seg) for seg in section_labels if label_section_count.get(seg[0], 0) > 1)
        rep_ratio = repeated / max(len(labels), 1)
        conn.execute(
            """
            INSERT INTO track_structure (track_id, n_sections, avg_section_s, repetition_ratio, boundaries_s)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (track_id) DO UPDATE SET n_sections = EXCLUDED.n_sections,
                avg_section_s = EXCLUDED.avg_section_s,
                repetition_ratio = EXCLUDED.repetition_ratio,
                boundaries_s = EXCLUDED.boundaries_s, computed_at = now()
            """,
            (track_id, n_sections, total_s / n_sections, float(rep_ratio),
             [float(t) for t in times]),
        )
        return True


class StemsHead:
    """Demucs source separation → per-stem energy profile. vocal_ratio is
    the honest vocal-presence signal (replaces the MuLan proxy); the four
    ratios together are an arrangement fingerprint (beat-driven vs ambient
    vs vocal-forward). GPU-heavy: maintenance windows / selected slices."""

    name = "stems"
    version = 1

    def __init__(self, separator=None):
        self._sep = separator

    def _ensure(self):
        if self._sep is None:
            try:
                import torch
                from demucs.apply import apply_model
                from demucs.pretrained import get_model
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("stems head needs the wave3 extra: uv sync --extra wave3") from e

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = get_model("htdemucs").to(device).eval()

            class _Sep:  # PyPI demucs is 4.0.x — adapt the classic API to ours
                samplerate = model.samplerate

                def separate_tensor(self, wav, sr=None):
                    with torch.no_grad():
                        out = apply_model(model, wav.unsqueeze(0).to(device), progress=False)[0]
                    return None, dict(zip(model.sources, out.cpu(), strict=True))

            self._sep = _Sep()
        return self._sep

    def run(self, conn: Connection, track_id, mono: np.ndarray, sr: int) -> bool:
        import torch

        if mono.size < sr * 10:
            return False
        sep = self._ensure()
        wav = torch.as_tensor(mono[: sr * 120], dtype=torch.float32)  # cap 2min
        wav = wav.unsqueeze(0).repeat(2, 1)  # demucs expects stereo
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, sep.samplerate)
        _, stems = sep.separate_tensor(wav, sr=sep.samplerate)
        energies = {k: float((v ** 2).mean()) for k, v in stems.items()}
        total = sum(energies.values()) or 1e-12
        r = {k: v / total for k, v in energies.items()}
        conn.execute(
            """
            INSERT INTO track_stems (track_id, vocal_ratio, drums_ratio, bass_ratio, other_ratio)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (track_id) DO UPDATE SET vocal_ratio = EXCLUDED.vocal_ratio,
                drums_ratio = EXCLUDED.drums_ratio, bass_ratio = EXCLUDED.bass_ratio,
                other_ratio = EXCLUDED.other_ratio, computed_at = now()
            """,
            (track_id, r.get("vocals", 0), r.get("drums", 0), r.get("bass", 0), r.get("other", 0)),
        )
        return True


class AsrHead:
    """Language detection (faster-whisper, first 60s). Stores language +
    confidence only — lyrics storage is a licensing question, not taken."""

    name = "asr"
    version = 1

    def __init__(self, model=None):
        self._model = model

    def _ensure(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("asr head needs the wave3 extra: uv sync --extra wave3") from e
            import torch

            self._model = WhisperModel(
                "small", device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type="int8",
            )
        return self._model

    def run(self, conn: Connection, track_id, mono: np.ndarray, sr: int) -> bool:
        if mono.size < sr * 5:
            return False
        model = self._ensure()
        import librosa

        clip = mono[: sr * 60].astype(np.float32)
        if sr != 16000:
            clip = librosa.resample(clip, orig_sr=sr, target_sr=16000)
        _segments, info = model.transcribe(clip, language=None, vad_filter=True)
        conn.execute(
            """
            INSERT INTO track_language (track_id, language, confidence)
            VALUES (%s, %s, %s)
            ON CONFLICT (track_id) DO UPDATE SET language = EXCLUDED.language,
                confidence = EXCLUDED.confidence, computed_at = now()
            """,
            (track_id, info.language, float(info.language_probability)),
        )
        return True


_HEADS = {"structure": StructureHead, "stems": StemsHead, "asr": AsrHead}
_PENDING_SQL = {
    "structure": "NOT EXISTS (SELECT 1 FROM track_structure s WHERE s.track_id = t.id)",
    "stems": "NOT EXISTS (SELECT 1 FROM track_stems s WHERE s.track_id = t.id)",
    "asr": "NOT EXISTS (SELECT 1 FROM track_language s WHERE s.track_id = t.id)",
}


def run_wave3(
    conn: Connection, limit: int, artist_id: str | None = None,
    head_name: str = "structure", published_only: bool = False,
) -> dict:
    """Selected-slice runner: fetches audio via the standard self-healing
    path, runs available wave-3 heads on tracks lacking them."""
    import tempfile
    from pathlib import Path

    from pipeline.embed_job import _decode, _default_refresher, _fetch_with_refresh, fetch_audio

    head = _HEADS[head_name]()
    # Locked rollout policy (2026-06-11): wave-3 depth goes to PUBLISHED
    # artists (embedded + mbid — the publish gate), never blanket.
    published_filter = (
        "AND t.artist_id IN (SELECT id FROM artist WHERE embedding_source IS NOT NULL AND mbid IS NOT NULL)"
        if published_only else ""
    )
    rows = conn.execute(
        f"""
        SELECT t.id, t.audio_url, t.platform, t.platform_track_id
        FROM audio_track t
        JOIN artist_embedding ae ON ae.artist_id = t.artist_id
        WHERE t.audio_url IS NOT NULL
          AND {_PENDING_SQL[head_name]}
          {"AND t.artist_id = %(aid)s" if artist_id else ""}
          {published_filter}
        ORDER BY t.id LIMIT %(lim)s
        """,
        {"lim": limit, "aid": artist_id},
    ).fetchall()
    done = skipped = 0
    with tempfile.TemporaryDirectory(prefix="wave3-") as tmp:
        for tid, url, platform, ptid in rows:
            try:  # per-track isolation: one rotted URL must not kill the slice
                path = _fetch_with_refresh(conn, url, platform, ptid, Path(tmp), fetch_audio, _default_refresher)
                if path is None:
                    skipped += 1
                    continue
                mono, sr = _decode(path)
                done += 1 if head.run(conn, tid, mono, sr) else 0
            except Exception:  # noqa: BLE001 — selective slices, skip and move on
                skipped += 1
    return {"head": head.name, "done": done, "skipped": skipped}


def main() -> None:
    import argparse
    import json

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="Wave-3 gated head runner (selected slices only)")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--artist", default=None)
    ap.add_argument("--head", default="structure", choices=sorted(_HEADS))
    ap.add_argument("--published", action="store_true", help="published artists only (locked rollout policy)")
    args = ap.parse_args()
    with psycopg.connect(Settings().database_url) as conn:
        report = run_wave3(conn, args.limit, args.artist, args.head, published_only=args.published)
        conn.commit()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
