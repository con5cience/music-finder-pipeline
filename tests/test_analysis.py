"""Wave-1 CPU heads: integrity verdicts, key/mode ground truth on synthetic
audio (the promised confidence raiser), tempo ground truth, fingerprint
determinism, DB upsert. Tag vocabulary/alias plumbing tested against
synthetic mb_raw rows (hermetic vs the real loaded vocabulary)."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.analysis import analyze_track, detect_key, fingerprint_pcm, integrity_check, upsert_track_analysis
from pipeline.tags import load_alias_map, load_vocabulary, replace_track_tags

SR = 22050


def _sine_mix(freqs: list[float], duration_s: float = 10.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    x = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return (x / np.max(np.abs(x)) * 0.5).astype(np.float32)


def _click_track(bpm: float, duration_s: float = 20.0, sr: int = SR) -> np.ndarray:
    x = np.zeros(int(duration_s * sr), dtype=np.float32)
    step = int(sr * 60 / bpm)
    for i in range(0, len(x) - 200, step):
        x[i:i + 200] = np.hanning(200).astype(np.float32)
    return x


# --- integrity --------------------------------------------------------------


def test_integrity_silent():
    _s, _c, verdict = integrity_check(np.zeros(SR * 30, dtype=np.float32), SR)
    assert verdict == "silent"


def test_integrity_clipped():
    x = np.ones(SR * 30, dtype=np.float32)  # fully saturated
    _s, _c, verdict = integrity_check(x, SR)
    assert verdict == "clipped"


def test_integrity_short():
    assert integrity_check(_sine_mix([440], 2.0), SR)[2] == "short"


def test_integrity_ok_on_normal_audio():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(SR * 30) * 0.1).astype(np.float32)
    assert integrity_check(x, SR)[2] == "ok"


# --- key/mode ground truth (synthetic triads) --------------------------------


def _chroma_for(notes: list[int]) -> np.ndarray:
    c = np.full(12, 0.05)
    for n in notes:
        c[n % 12] = 1.0
    return c


def test_key_a_major_from_chroma():
    # A major triad: A, C#, E = pitch classes 9, 1, 4
    assert detect_key(_chroma_for([9, 1, 4])) == ("A", "major")


def test_key_c_minor_from_chroma():
    # C minor triad: C, Eb, G = 0, 3, 7
    assert detect_key(_chroma_for([0, 3, 7])) == ("C", "minor")


def test_key_from_real_synthesized_audio():
    # End-to-end: synthesized A-major triad audio through the real chroma path.

    y = _sine_mix([220.0, 277.18, 329.63])  # A3, C#4, E4
    from pipeline.analysis import chroma_mean as _cm
    chroma = _cm(y, SR)
    key, mode = detect_key(chroma)
    assert (key, mode) == ("A", "major")


# --- tempo ground truth -------------------------------------------------------


def test_tempo_on_click_track():
    import librosa

    y = _click_track(120.0)
    tempo = float(librosa.feature.tempo(y=y, sr=SR)[0])
    assert abs(tempo - 120.0) < 3.0 or abs(tempo - 60.0) < 3.0 or abs(tempo - 240.0) < 6.0
    # octave errors (60/240) are a known tempo-estimation ambiguity class; the
    # primary assertion is that 120 is recovered — flag if librosa regresses:
    assert abs(tempo - 120.0) < 3.0


# --- fingerprint --------------------------------------------------------------


def test_fingerprint_deterministic_and_nonempty():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(SR * 20) * 0.2).astype(np.float32)
    fp1 = fingerprint_pcm(x, SR)
    fp2 = fingerprint_pcm(x, SR)
    assert fp1 == fp2
    assert len(fp1) > 50


# --- analyze_track + DB --------------------------------------------------------


def test_analyze_track_populates_and_upserts(conn):
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Analysis Fixture', "
        "'00000000-feed-4bad-9bad-000000000aaa') RETURNING id"
    ).fetchone()[0]
    t = conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
        "binding_tier, verification_status) VALUES (%s,'deezer','zz-an-1','/x.mp3',30,'A','verified') "
        "RETURNING id",
        (a,),
    ).fetchone()[0]
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(SR * 30) * 0.1).astype(np.float32)
    sig = analyze_track(x, SR)
    assert sig.integrity == "ok"
    assert sig.tempo_bpm and sig.key and sig.mode and sig.fingerprint
    upsert_track_analysis(conn, t, sig)
    upsert_track_analysis(conn, t, sig)  # idempotent re-run
    row = conn.execute(
        "SELECT integrity, tempo_bpm, key, fingerprint IS NOT NULL FROM track_analysis WHERE track_id=%s",
        (t,),
    ).fetchone()
    assert row[0] == "ok" and row[1] is not None and row[3] is True


# --- vocabulary plumbing --------------------------------------------------------


@pytest.fixture
def synthetic_vocab(conn):
    conn.execute(
        "INSERT INTO mb_raw.genre (id, gid, name) VALUES "
        "(990001, '00000000-feed-4bad-9bad-00000000a001', 'zz-test-genre'), "
        "(990002, '00000000-feed-4bad-9bad-00000000a002', 'zz-other-genre')"
    )
    conn.execute(
        "INSERT INTO mb_raw.genre_alias (id, genre, name, sort_name) VALUES "
        "(990101, 990001, 'zz test genre', 'zz test genre'), "
        "(990102, 990001, 'zztestgenre', 'zztestgenre')"
    )


def test_vocabulary_and_alias_merge(conn, synthetic_vocab):
    vocab = load_vocabulary(conn)
    assert "zz-test-genre" in vocab
    aliases = load_alias_map(conn)
    # the user's exact ask: variants merge to the canonical
    assert aliases["zz test genre"] == "zz-test-genre"
    assert aliases["zztestgenre"] == "zz-test-genre"


def test_tag_scorer_does_not_memoize_failure(monkeypatch):
    # Review finding: functools.cache memoized None when the vocabulary was
    # empty at first call, silently disabling tags for the process lifetime.
    import pipeline.activities as acts
    import pipeline.tags as tags_mod

    monkeypatch.setattr(acts, "_tag_scorer_memo", [])
    calls = {"n": 0}

    def fake_vocab(conn):
        calls["n"] += 1
        return [] if calls["n"] == 1 else ["zz-genre"]

    class FakeScorer:
        def __init__(self, vocab):
            self.vocabulary = vocab

    monkeypatch.setattr(tags_mod, "load_vocabulary", fake_vocab)
    monkeypatch.setattr(tags_mod, "MulanTagScorer", FakeScorer)

    assert acts._tag_scorer() is None          # empty vocab → None, NOT cached
    scorer = acts._tag_scorer()                # vocabulary arrived → recovers
    assert scorer is not None and scorer.vocabulary == ["zz-genre"]
    assert acts._tag_scorer() is scorer        # success IS memoized


def test_backfill_analyzes_embedded_tracks_idempotently(conn, tmp_path):
    import json

    import soundfile as sf

    from pipeline.backfill_analysis import backfill_tracks

    a = conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Backfill Fixture', "
        "'00000000-feed-4bad-9bad-000000000ccc') RETURNING id"
    ).fetchone()[0]
    rng = np.random.default_rng(3)
    wav = tmp_path / "bf.wav"
    sf.write(wav, (rng.standard_normal(SR * 90) * 0.1).astype(np.float32), SR)
    t = conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
        "binding_tier, binding_evidence, verification_status) "
        "VALUES (%s,'bandcamp','zz-bf-1',%s,90,'A',%s,'verified') RETURNING id",
        (a, str(wav), json.dumps({"release_index": 0, "track_index": 0})),
    ).fetchone()[0]
    # embedded (so it qualifies) but never analyzed
    conn.execute(
        "INSERT INTO clip_embedding (track_id, segment_start_s, segment_end_s, model, dim, embedding) "
        "VALUES (%s, 0, 30, 'mock-model', 2, '[0.6,0.8]')",
        (t,),
    )

    class FakeScorer:
        def score_clips(self, artist_id, paths):
            assert paths
            return [("zz-bf-genre", 0.5)]

    from pipeline.heads import CpuAnalysisHead, TagHead

    heads = [CpuAnalysisHead(), TagHead(FakeScorer())]
    done, skipped = backfill_tracks(conn, heads, 10, artist_id=str(a))
    assert (done, skipped) == (1, 0)
    assert conn.execute(
        "SELECT integrity FROM track_analysis WHERE track_id = %s", (t,)
    ).fetchone()[0] == "ok"
    assert conn.execute(
        "SELECT count(*) FROM track_tag_scores WHERE track_id = %s", (t,)
    ).fetchone()[0] == 1
    # idempotent: every head current (track_head_runs) → nothing to do
    assert backfill_tracks(conn, heads, 10, artist_id=str(a)) == (0, 0)


def test_track_tag_upsert_roundtrip(conn):
    a = conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Tag Fixture', "
        "'00000000-feed-4bad-9bad-000000000bbb') RETURNING id"
    ).fetchone()[0]
    t = conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
        "binding_tier, verification_status) VALUES (%s,'deezer','zz-tag-1','/x.mp3',30,'A','verified') "
        "RETURNING id",
        (a,),
    ).fetchone()[0]
    replace_track_tags(conn, t, [("dubstep", 0.31), ("uk garage", 0.27)])
    # Review finding: re-scoring must REPLACE, not accumulate — stale tags
    # from a prior run would pollute the per-tag calibration distributions.
    replace_track_tags(conn, t, [("dubstep", 0.33), ("2-step", 0.29)])
    rows = dict(
        conn.execute("SELECT tag, score FROM track_tag_scores WHERE track_id = %s", (t,)).fetchall()
    )
    assert set(rows) == {"dubstep", "2-step"}  # 'uk garage' gone, not lingering
    assert abs(rows["dubstep"] - 0.33) < 1e-6


def test_perceptual_head_axes_and_instruments(conn, tmp_path):
    # Wave-2: anchor-pair axes + instrument top-k from a FAKE MuLan (hermetic).
    import soundfile as sf

    from pipeline.heads import (
        AXIS_ANCHORS,
        INSTRUMENT_TOP_K,
        INSTRUMENT_VOCAB,
        HeadContext,
        PerceptualHead,
        run_heads,
    )

    class FakeMulan:
        def embed_text(self, texts):
            # deterministic distinct unit vectors per text
            out = []
            for i, _t in enumerate(texts):
                v = np.zeros(8)
                v[i % 8] = 1.0
                out.append(v.tolist())
            return out

        def embed(self, clips):
            v = np.zeros(8)
            v[0] = 1.0  # aligns with the FIRST anchor (danceability positive)
            return [v.tolist() for _ in clips]

    class FakeScorer:
        _embedder = FakeMulan()

        def _ensure(self):
            pass

    a = conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Perceptual Fixture', "
        "'00000000-feed-4bad-9bad-000000000ddd') RETURNING id"
    ).fetchone()[0]
    t = conn.execute(
        "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
        "binding_tier, verification_status) VALUES (%s,'deezer','zz-pc-1','/x.mp3',30,'A','verified') "
        "RETURNING id",
        (a,),
    ).fetchone()[0]
    wav = tmp_path / "pc.wav"
    sf.write(wav, np.zeros(8000, dtype=np.float32), 8000)

    heads = [PerceptualHead(FakeScorer())]
    ctx = HeadContext(conn=conn, track_id=t, artist_id=str(a), platform="deezer",
                      mono=np.zeros(8000), sr=8000, clip_paths=[str(wav)])
    assert run_heads(conn, heads, ctx) == 1
    row = conn.execute(
        "SELECT danceability, instruments, model FROM track_perceptual WHERE track_id = %s", (t,)
    ).fetchone()
    # audio vec == danceability-positive anchor → cos(pos)=1, cos(neg)=0 → axis = +1
    assert abs(row[0] - 1.0) < 1e-5
    assert len(row[1]) == INSTRUMENT_TOP_K
    assert all(i["name"] in INSTRUMENT_VOCAB for i in row[1])
    assert len(AXIS_ANCHORS) == 6
    # idempotent: current version recorded → second run is a no-op
    assert run_heads(conn, heads, ctx) == 0


def test_tag_calibration_zscore_damps_prior_noise(conn):
    # A "prior-noise" tag scores ~0.45 for EVERYTHING; a real signal tag
    # scores 0.30 baseline but 0.50 on one track. Raw ranking puts the noise
    # tag first; z-ranking must invert that.
    from pipeline.tag_calibration import MIN_N, calibrated_tags, refresh_calibration
    from pipeline.tags import replace_track_tags

    a = conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Calib Fixture', "
        "'00000000-feed-4bad-9bad-000000000892') RETURNING id"
    ).fetchone()[0]
    tracks = []
    for i in range(MIN_N + 5):
        t = conn.execute(
            "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
            "binding_tier, verification_status) VALUES (%s,'deezer',%s,'/x.mp3',30,'A','verified') "
            "RETURNING id",
            (a, f"zz-cal-{i}"),
        ).fetchone()[0]
        tracks.append(t)
    target = tracks[0]
    for t in tracks:
        scores = [("zz-noise-tag", 0.55), ("zz-signal-tag", 0.50 if t == target else 0.30)]
        replace_track_tags(conn, t, scores)

    n = refresh_calibration(conn)
    assert n >= 2
    ranked = calibrated_tags(conn, target)
    by_raw = sorted(ranked, key=lambda r: -r[1])
    assert by_raw[0][0] == "zz-noise-tag"          # raw ordering: noise wins
    assert ranked[0][0] == "zz-signal-tag"         # z ordering: signal wins
    assert ranked[0][2] > 2.0                      # strongly above its own baseline

    # refresh is idempotent-replace (derived data)
    assert refresh_calibration(conn) == n
