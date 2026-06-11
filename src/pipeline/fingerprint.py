"""Spectral fingerprinting: dedup + cross-artist same-recording detection.

Self-contained (librosa only — no chromaprint system dep): a track's mono
audio becomes a 32-band log-mel bit matrix (band-energy deltas, the
Haitsma-Kalker family), hamming-comparable across tracks. Two tiers:

- exact_hash: sha1 of the bit matrix — identical-audio dupes (same file
  uploaded twice, single + album version from the same master).
- similarity(): windowed best-offset hamming similarity in [0,1] — robust
  to different encodes/trims of the same recording. Preview (30s) vs full
  track works because the comparison slides the SHORTER fp over the longer.

Usage (FLAG-ONLY law, like the integrity gates):
- FingerprintHead (prep stage, cheap CPU) stores per-track rows.
- match_artist_duplicates: within an artist, near-identical tracks (the
  centroid double-counting case) → report rows.
- match_cross_artist: same recording under TWO artists = binding-error
  signal → review_item evidence for the Tier-C queue. Never auto-acts.
"""

from __future__ import annotations

import hashlib

import numpy as np
from psycopg import Connection

N_BANDS = 33  # 33 bands → 32-bit delta words
FP_SR = 11025
HOP = 1024  # ~93ms per frame: sub-frame phase error stays small enough that
# an arbitrary-offset excerpt (preview vs full) still aligns; 60s ≈ 2.6KB fp
SIM_THRESHOLD = 0.78  # flag-only. Measured on synthetic harness: same-recording
# variants score 0.80-1.0, unrelated tracks 0.50-0.65 — 0.78 sits mid-gap with
# margin both ways. Re-calibrate on real corpus dupes before ANY automation.
MIN_OVERLAP_FRAMES = 40  # ~15s of audio must align before we call it a match


def compute_fingerprint(mono: np.ndarray, sr: int) -> tuple[bytes, float]:
    """Bit-matrix fingerprint of the band-energy delta sign (uint32/frame)."""
    import librosa

    if sr != FP_SR:
        mono = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=FP_SR)
    mel = librosa.feature.melspectrogram(
        y=mono.astype(np.float32), sr=FP_SR, n_mels=N_BANDS, hop_length=HOP
    )
    e = np.log1p(mel)  # (33, fine-frames at ~93ms)
    # WIDE smoothing (9 fine frames ≈ 0.8s) then stride-4 sampling: each bit
    # summarizes ~1s of audio, so an arbitrary-offset excerpt (preview vs
    # full) lands within a fraction of the smoothing window — bits survive
    # phase misalignment that single-frame deltas cannot.
    if e.shape[1] >= 9:
        kernel = np.ones(9, dtype=np.float32) / 9.0
        e = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, e)
    e = e[:, ::4]
    # sign of the (band, time) double delta — the classic robust bit
    d = e[1:, 1:] - e[:-1, 1:] - (e[1:, :-1] - e[:-1, :-1])  # (32, frames-1)
    bits = (d > 0).astype(np.uint8).T  # (frames, 32): frame-major for the u4 view
    packed = np.ascontiguousarray(np.packbits(bits, axis=1, bitorder="big"))  # (frames, 4)
    words = packed.view(">u4").ravel().astype(np.uint32)
    return words.tobytes(), len(words) * HOP * 4 / FP_SR


def exact_hash(fp: bytes) -> str:
    return hashlib.sha1(fp).hexdigest()


def similarity(fp_a: bytes, fp_b: bytes) -> float:
    """Best-offset hamming similarity in [0,1]; slides the shorter over the
    longer (preview-vs-full alignment). 0.5 ≈ unrelated, 1.0 identical."""
    a = np.frombuffer(fp_a, dtype=np.uint32)
    b = np.frombuffer(fp_b, dtype=np.uint32)
    if len(a) > len(b):
        a, b = b, a
    n = len(a)
    if n < MIN_OVERLAP_FRAMES:
        return 0.0
    best = 0.0
    # full scan at the coarse step (~0.37s) — cheap at these lengths
    offsets = list(range(0, len(b) - n + 1)) or [0]
    def score(off: int) -> float:
        x = np.bitwise_xor(a, b[off : off + n])
        ham = np.unpackbits(x.view(np.uint8)).sum()
        return 1.0 - ham / (n * 32)
    for off in offsets:
        s = score(off)
        if s > best:
            best = s
    return best


def store_fingerprint(conn: Connection, track_id, mono: np.ndarray, sr: int) -> None:
    fp, seconds = compute_fingerprint(mono, sr)
    conn.execute(
        """
        INSERT INTO track_fingerprint (track_id, fp, fp_seconds, exact_hash)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (track_id) DO UPDATE SET fp = EXCLUDED.fp,
            fp_seconds = EXCLUDED.fp_seconds, exact_hash = EXCLUDED.exact_hash,
            computed_at = now()
        """,
        (track_id, fp, seconds, exact_hash(fp)),
    )


def match_artist_duplicates(conn: Connection, artist_id) -> list[dict]:
    """Near-identical tracks WITHIN one artist (centroid double-counting)."""
    rows = conn.execute(
        """
        SELECT tf.track_id, tf.fp, t.platform, t.platform_track_id
        FROM track_fingerprint tf JOIN audio_track t ON t.id = tf.track_id
        WHERE t.artist_id = %s
        """,
        (artist_id,),
    ).fetchall()
    found = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            s = similarity(bytes(rows[i][1]), bytes(rows[j][1]))
            if s >= SIM_THRESHOLD:
                found.append({
                    "track_a": str(rows[i][0]), "track_b": str(rows[j][0]),
                    "platforms": [rows[i][2], rows[j][2]], "similarity": round(s, 4),
                })
    return found


def match_cross_artist(conn: Connection, limit: int = 1000) -> int:
    """Same recording bound to DIFFERENT artists = binding-error evidence.
    Exact-hash tier only for the global pass (pairwise across the corpus is
    quadratic; hash buckets are free). Files review_items, flag-only."""
    import json

    pairs = conn.execute(
        """
        SELECT a.exact_hash, ta.artist_id, tb.artist_id, ta.id, tb.id
        FROM track_fingerprint a
        JOIN track_fingerprint b ON b.exact_hash = a.exact_hash AND b.track_id > a.track_id
        JOIN audio_track ta ON ta.id = a.track_id
        JOIN audio_track tb ON tb.id = b.track_id
        WHERE ta.artist_id != tb.artist_id
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    filed = 0
    for h, art_a, art_b, tr_a, tr_b in pairs:
        dup = conn.execute(
            """
            SELECT 1 FROM review_item
            WHERE kind = 'source_binding' AND subject_id = %s
              AND evidence->'fp_collision'->>'other_artist' = %s
            """,
            (art_a, str(art_b)),
        ).fetchone()
        if dup:
            continue
        conn.execute(
            """
            INSERT INTO review_item (kind, subject_type, subject_id, reason, evidence, status)
            VALUES ('source_binding', 'artist', %s,
                    'fingerprint: identical recording under two artists', %s, 'pending')
            """,
            (art_a, json.dumps({"fp_collision": {
                "other_artist": str(art_b), "track_a": str(tr_a),
                "track_b": str(tr_b), "exact_hash": h}})),
        )
        filed += 1
    return filed


def main() -> None:
    import argparse
    import json as _json

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="fingerprint matcher (flag-only)")
    ap.add_argument("--artist-dups", action="store_true", help="per-artist near-dup report")
    ap.add_argument("--cross-artist", action="store_true", help="file binding-error reviews")
    ap.add_argument("--limit", type=int, default=200)
    import sys

    argv = [a for i, a in enumerate(sys.argv[1:]) if not (a == "--" and i == 0)]
    args = ap.parse_args(argv)
    with psycopg.connect(Settings().database_url) as conn:
        if args.cross_artist:
            n = match_cross_artist(conn)
            conn.commit()
            print(f"cross-artist collisions filed: {n}")
        if args.artist_dups:
            artists = conn.execute(
                """
                SELECT DISTINCT t.artist_id FROM track_fingerprint tf
                JOIN audio_track t ON t.id = tf.track_id LIMIT %s
                """,
                (args.limit,),
            ).fetchall()
            total = []
            for (aid,) in artists:
                total.extend(match_artist_duplicates(conn, aid))
            print(_json.dumps({"artists_checked": len(artists), "near_dups": total[:50],
                               "near_dup_count": len(total)}, indent=2))


if __name__ == "__main__":
    main()
