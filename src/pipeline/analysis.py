"""Wave-1 CPU analysis heads (ADR-015 decode-once): integrity, MIR, fingerprint.

All heads consume the ALREADY-DECODED mono waveform — never re-decode, never
touch the network. Integrity is FLAG-ONLY in v1: verdicts are recorded for
corpus calibration, enforcement comes later with calibrated thresholds.
librosa/pyacoustid import lazily (heavy; workers that never analyze never pay).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from psycopg import Connection

ANALYSIS_VERSION = 1
_MAX_ANALYZE_S = 180  # MIR features on at most this much audio (tempo/key converge fast)
_SILENCE_AMP = 1e-4
_CLIP_AMP = 0.985
_SILENT_FRAC = 0.90
_CLIPPED_FRAC = 0.05
_MIN_DURATION_S = 5

_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler key profiles (probe used naive chroma-argmax; these add
# mode and are the standard template-correlation approach)
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class TrackSignals:
    analyzed_s: float
    silence_frac: float
    clip_frac: float
    integrity: str
    tempo_bpm: float | None = None
    key: str | None = None
    mode: str | None = None
    spectral_centroid_hz: float | None = None
    rms: float | None = None
    loudness_lufs: float | None = None
    fingerprint: str | None = None


def integrity_check(mono: np.ndarray, sr: int) -> tuple[float, float, str]:
    """(silence_frac, clip_frac, verdict). Flag-only: callers record, not enforce."""
    silence = float((np.abs(mono) < _SILENCE_AMP).mean())
    clip = float((np.abs(mono) > _CLIP_AMP).mean())
    if len(mono) < _MIN_DURATION_S * sr:
        return silence, clip, "short"
    if silence > _SILENT_FRAC:
        return silence, clip, "silent"
    if clip > _CLIPPED_FRAC:
        return silence, clip, "clipped"
    return silence, clip, "ok"


def chroma_mean(y: np.ndarray, sr: int) -> np.ndarray:
    """CQT chroma (sharper key detection — passes the triad ground-truth tests)
    at normal rates; stft fallback below ~11kHz where CQT's wavelet basis would
    violate Nyquist (found via 8kHz test audio)."""
    import librosa

    if sr >= 11025:
        return librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    return librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)


def detect_key(chroma_mean: np.ndarray) -> tuple[str, str]:
    """Krumhansl-Schmuckler template correlation over 24 key/mode rotations."""
    best = (-2.0, "C", "major")
    for i in range(12):
        rolled = np.roll(chroma_mean, -i)
        for mode, profile in (("major", _MAJOR), ("minor", _MINOR)):
            r = float(np.corrcoef(rolled, profile)[0, 1])
            if r > best[0]:
                best = (r, _KEYS[i], mode)
    return best[1], best[2]


def fingerprint_pcm(mono: np.ndarray, sr: int) -> str:
    """Chromaprint over already-decoded audio (raw-PCM path; no re-decode)."""
    import acoustid

    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    fp = acoustid.fingerprint(sr, 1, iter([pcm]))
    return fp.decode() if isinstance(fp, bytes) else fp


def analyze_track(mono: np.ndarray, sr: int) -> TrackSignals:
    """Run all CPU heads on one decoded track."""
    silence, clip, verdict = integrity_check(mono, sr)
    sig = TrackSignals(
        analyzed_s=len(mono) / sr, silence_frac=silence, clip_frac=clip, integrity=verdict
    )
    if verdict == "short":
        return sig  # too little audio for meaningful MIR

    import librosa
    import pyloudnorm

    y = mono[: _MAX_ANALYZE_S * sr]
    sig.tempo_bpm = float(librosa.feature.tempo(y=y, sr=sr)[0])
    chroma = chroma_mean(y, sr)
    sig.key, sig.mode = detect_key(chroma)
    sig.spectral_centroid_hz = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    sig.rms = float(librosa.feature.rms(y=y).mean())
    try:
        sig.loudness_lufs = float(pyloudnorm.Meter(sr).integrated_loudness(y.astype(np.float64)))
    except Exception:  # noqa: BLE001 — loudness can fail on degenerate audio; not worth dying for
        sig.loudness_lufs = None
    sig.fingerprint = fingerprint_pcm(mono, sr)
    return sig


def upsert_track_analysis(conn: Connection, track_id, sig: TrackSignals) -> None:
    conn.execute(
        """
        INSERT INTO track_analysis (track_id, analysis_version, fingerprint, analyzed_s,
            tempo_bpm, key, mode, spectral_centroid_hz, rms, loudness_lufs,
            silence_frac, clip_frac, integrity, computed_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (track_id) DO UPDATE SET
            analysis_version = EXCLUDED.analysis_version,
            fingerprint = EXCLUDED.fingerprint, analyzed_s = EXCLUDED.analyzed_s,
            tempo_bpm = EXCLUDED.tempo_bpm, key = EXCLUDED.key, mode = EXCLUDED.mode,
            spectral_centroid_hz = EXCLUDED.spectral_centroid_hz, rms = EXCLUDED.rms,
            loudness_lufs = EXCLUDED.loudness_lufs, silence_frac = EXCLUDED.silence_frac,
            clip_frac = EXCLUDED.clip_frac, integrity = EXCLUDED.integrity, computed_at = now()
        """,
        (track_id, ANALYSIS_VERSION, sig.fingerprint, sig.analyzed_s, sig.tempo_bpm,
         sig.key, sig.mode, sig.spectral_centroid_hz, sig.rms, sig.loudness_lufs,
         sig.silence_frac, sig.clip_frac, sig.integrity),
    )
