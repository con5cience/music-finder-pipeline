"""Fingerprint: identity, encode-robustness (noise), preview-vs-full offset
alignment, non-match separation, per-artist dedup + cross-artist flag-only
review. Wave-3 StructureHead: sectioned synthetic song → sane summary."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.fingerprint import (
    SIM_THRESHOLD,
    compute_fingerprint,
    exact_hash,
    match_artist_duplicates,
    match_cross_artist,
    similarity,
    store_fingerprint,
)

SR = 22050
MBID_A = "00000000-feed-4bad-9bad-000000000fa1"
MBID_B = "00000000-feed-4bad-9bad-000000000fa2"


def _song(seed: int, secs: int = 60) -> np.ndarray:
    """Deterministic pseudo-music: chord-ish tones + filtered noise bursts."""
    rng = np.random.default_rng(seed)
    t = np.arange(SR * secs) / SR
    y = np.zeros_like(t)
    for f in rng.uniform(80, 800, 6):
        y += np.sin(2 * np.pi * f * t) * rng.uniform(0.1, 0.3)
    env = np.clip(np.sin(2 * np.pi * t / rng.uniform(2, 5)), 0, 1)
    return (y * env + rng.standard_normal(len(t)) * 0.02).astype(np.float32)


def test_identity_and_noise_robustness():
    a = _song(1)
    fp_a, secs = compute_fingerprint(a, SR)
    assert secs == pytest.approx(60, abs=2)
    assert similarity(fp_a, fp_a) == 1.0
    # re-encode proxy: small noise + slight gain — must stay a confident match
    noisy = (a * 0.92 + np.random.default_rng(9).standard_normal(len(a)).astype(np.float32) * 0.01)
    fp_n, _ = compute_fingerprint(noisy, SR)
    assert similarity(fp_a, fp_n) >= SIM_THRESHOLD
    # exact hash differs (different bits) but similarity catches it
    assert exact_hash(fp_a) != exact_hash(fp_n)


def test_preview_vs_full_alignment_and_separation():
    full = _song(2, secs=90)
    fp_full, _ = compute_fingerprint(full, SR)
    # a 30s excerpt starting at 0:25 — the preview case
    excerpt = full[SR * 25 : SR * 55]
    fp_prev, _ = compute_fingerprint(excerpt, SR)
    assert similarity(fp_prev, fp_full) >= SIM_THRESHOLD
    # different song entirely → well below threshold
    other = _song(3, secs=90)
    fp_other, _ = compute_fingerprint(other, SR)
    assert similarity(fp_full, fp_other) < 0.75


def _artist(conn, name, mbid):
    return conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES (%s, %s) RETURNING id", (name, mbid)
    ).fetchone()[0]


def _track(conn, artist_id, ptid):
    return conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, "
        "binding_tier, verification_status) VALUES (%s, 'deezer', %s, 'x', 'A', 'verified') "
        "RETURNING id",
        (artist_id, ptid),
    ).fetchone()[0]


def test_artist_dups_and_cross_artist_flag_only(conn):
    art_a = _artist(conn, "FP Fixture A", MBID_A)
    art_b = _artist(conn, "FP Fixture B", MBID_B)
    song = _song(7)
    t1 = _track(conn, art_a, "zz-fp-1")
    t2 = _track(conn, art_a, "zz-fp-2")  # same recording, re-encoded
    t3 = _track(conn, art_b, "zz-fp-3")  # SAME recording under another artist
    store_fingerprint(conn, t1, song, SR)
    store_fingerprint(conn, t2, song, SR)  # identical bytes → same exact_hash
    store_fingerprint(conn, t3, song, SR)
    dups = match_artist_duplicates(conn, art_a)
    assert len(dups) == 1 and dups[0]["similarity"] >= SIM_THRESHOLD
    filed = match_cross_artist(conn)
    assert filed >= 1
    # subject/other ordering follows random track uuids — accept either side
    kind, status, ev = conn.execute(
        "SELECT kind, status, evidence FROM review_item "
        "WHERE subject_id IN (%s, %s) LIMIT 1", (art_a, art_b)).fetchone()
    assert kind == "source_binding" and status == "pending"
    assert ev["fp_collision"]["other_artist"] in (str(art_a), str(art_b))
    # flag-only law: nothing deleted, nothing re-bound
    assert conn.execute("SELECT count(*) FROM audio_track WHERE artist_id IN (%s, %s)",
                        (art_a, art_b)).fetchone()[0] == 3
    # idempotent: second pass files nothing new
    assert match_cross_artist(conn) == 0


def test_structure_head_sections(conn):
    from pipeline.wave3 import StructureHead

    art = _artist(conn, "Structure Fixture", "00000000-feed-4bad-9bad-000000000fb1")
    tid = _track(conn, art, "zz-st3-1")
    # A-B-A form: verse tone-set, chorus tone-set, verse again
    a, b = _song(11, 25), _song(12, 25)
    y = np.concatenate([a, b, a])
    head = StructureHead()
    assert head.run(conn, tid, y, SR) is True
    n, avg_s, rep, bounds = conn.execute(
        "SELECT n_sections, avg_section_s, repetition_ratio, boundaries_s "
        "FROM track_structure WHERE track_id = %s", (tid,)).fetchone()
    assert 2 <= n <= 8
    assert avg_s * n == pytest.approx(75, rel=0.2)
    assert 0.0 <= rep <= 1.0
    assert all(0 < t < 75 for t in bounds)
    # too-short audio declines (returns False, no row)
    t2 = _track(conn, art, "zz-st3-2")
    assert head.run(conn, t2, _song(13, 5), SR) is False


def test_stems_head_with_fake_separator(conn):
    from pipeline.wave3 import StemsHead

    class FakeSep:
        samplerate = 44100

        def separate_tensor(self, wav, sr=None):
            import torch
            n = wav.shape[-1]
            return None, {
                "vocals": torch.full((2, n), 0.3), "drums": torch.full((2, n), 0.1),
                "bass": torch.full((2, n), 0.1), "other": torch.full((2, n), 0.1),
            }

    art = _artist(conn, "Stems Fixture", "00000000-feed-4bad-9bad-000000000fc1")
    tid = _track(conn, art, "zz-w3s-1")
    head = StemsHead(separator=FakeSep())
    assert head.run(conn, tid, _song(21, 30), SR) is True
    v, d = conn.execute(
        "SELECT vocal_ratio, drums_ratio FROM track_stems WHERE track_id = %s", (tid,)).fetchone()
    assert v == pytest.approx(0.75, abs=0.02)  # 0.09 of 0.12 total energy
    assert d == pytest.approx(0.0833, abs=0.02)
    assert head.run(conn, _track(conn, art, "zz-w3s-2"), _song(22, 3), SR) is False


def test_asr_head_with_fake_model(conn):
    from pipeline.wave3 import AsrHead

    class Info:
        language = "fr"
        language_probability = 0.91

    class FakeModel:
        def transcribe(self, clip, language=None, vad_filter=True):
            return iter(()), Info()

    art = _artist(conn, "ASR Fixture", "00000000-feed-4bad-9bad-000000000fc2")
    tid = _track(conn, art, "zz-w3a-1")
    head = AsrHead(model=FakeModel())
    assert head.run(conn, tid, _song(23, 20), SR) is True
    lang, conf = conn.execute(
        "SELECT language, confidence FROM track_language WHERE track_id = %s", (tid,)).fetchone()
    assert lang == "fr" and conf == pytest.approx(0.91)
