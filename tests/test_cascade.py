"""Cascade decisions (ADR-017 §2 + per-source floors): priority, floor-ratio
fallback, scan verdicts, centroid purity. All fixture ids synthetic."""

from __future__ import annotations

from functools import partial

from pipeline.bench.mock import MockEmbedder
from pipeline.cascade import audio_identities, choose_source, floor_ratio, mark_scanned, source_yields
from pipeline.embed_job import embed_artist_clips as _embed_artist_clips

# These tests exercise cascade/selection mechanics with fake audio paths;
# Wave-1 analysis (which decodes real audio) is covered by test_analysis /
# test_embed_windowed.
embed_artist_clips = partial(_embed_artist_clips, run_analysis=False)

MBID = "00000000-feed-4bad-9bad-000000000555"


def _artist(conn) -> str:
    return conn.execute(
        "INSERT INTO artist (display_name, mbid) VALUES ('Cascade Fixture', %s) RETURNING id", (MBID,)
    ).fetchone()[0]


def _identity(conn, a, platform, pid):
    conn.execute(
        "INSERT INTO platform_identity (artist_id, platform, platform_id, page_type) "
        "VALUES (%s, %s, %s, 'artist')",
        (a, platform, pid),
    )


def _tracks(conn, a, platform, n, prefix, tmp_path=None):
    """Insert n tracks. Windowed platforms (bandcamp) get REAL wav files —
    the windowing path reads audio; preview platforms keep fake paths."""
    for i in range(n):
        if tmp_path is not None:
            import numpy as np
            import soundfile as sf

            p = tmp_path / f"{prefix}-{i}.wav"
            rng = np.random.default_rng(i)
            sf.write(p, rng.standard_normal(90 * 8000).astype(np.float32) * 0.05, 8000)
            url, dur = str(p), 90
        else:
            url, dur = f"/audio/{prefix}-{i}.mp3", 30
        conn.execute(
            "INSERT INTO audio_track (artist_id, platform, platform_track_id, audio_url, duration_s, "
            "binding_tier, binding_evidence, verification_status) "
            "VALUES (%s, %s, %s, %s, %s, 'A', %s, 'verified')",
            (a, platform, f"{prefix}-{i}", url, dur, f'{{"release_index": {i}}}'),
        )


# --- choose_source: pure policy ---------------------------------------------


def test_floor_met_wins_in_priority_order():
    # both deezer and bandcamp meet floor → deezer (priority) wins
    assert choose_source({"deezer": 10, "bandcamp": 5}) == ("deezer", 1.0)


def test_floor_met_beats_higher_ratio_lower_priority():
    # bandcamp 5/3 (1.67) vs deezer 10/10 (1.0): both >= 1 → priority decides
    got = choose_source({"deezer": 10, "bandcamp": 5})
    assert got is not None and got[0] == "deezer"


def test_floor_ratio_fallback_picks_best_thin_source():
    # 1 deezer preview (0.1) vs 2 bandcamp tracks (0.67) → bandcamp
    got = choose_source({"deezer": 1, "bandcamp": 2})
    assert got is not None
    platform, ratio = got
    assert platform == "bandcamp"
    assert 0.6 < ratio < 0.7


def test_bandcamp_below_floor_loses_to_deezer_at_floor():
    assert choose_source({"deezer": 10, "bandcamp": 2})[0] == "deezer"


def test_experimental_source_never_chosen():
    # youtube floor is None: scanned, recorded, never auto-embedded
    assert floor_ratio("youtube", 50) is None
    assert choose_source({"youtube": 50}) is None


def test_nothing_anywhere_is_none():
    assert choose_source({}) is None
    assert choose_source({"deezer": 0, "bandcamp": 0}) is None


# --- DB-backed pieces --------------------------------------------------------


def test_audio_identities_priority_order_and_audio_only(conn):
    a = _artist(conn)
    _identity(conn, a, "tidal", "zz-cas-t")      # playback asset: excluded
    _identity(conn, a, "soundcloud", "zz-cas-s")
    _identity(conn, a, "deezer", "zz-cas-d")
    rows = audio_identities(conn, a)
    assert [r[0] for r in rows] == ["deezer", "soundcloud"]
    assert all(r[2] == "pending" for r in rows)


def test_source_yields_counts_embeddable_tracks(conn):
    a = _artist(conn)
    _tracks(conn, a, "deezer", 2, "zz-cas-yd")
    _tracks(conn, a, "soundcloud", 1, "zz-cas-ys")
    assert source_yields(conn, a) == {"deezer": 2, "soundcloud": 1}


def test_mark_scanned_terminal_verdicts(conn):
    a = _artist(conn)
    _identity(conn, a, "deezer", "zz-cas-m1")
    _identity(conn, a, "soundcloud", "zz-cas-m2")
    mark_scanned(conn, "deezer", "zz-cas-m1", yield_n=7)
    mark_scanned(conn, "soundcloud", "zz-cas-m2", yield_n=0)
    rows = dict(
        conn.execute(
            "SELECT platform, scan_status FROM platform_identity WHERE artist_id = %s", (a,)
        ).fetchall()
    )
    assert rows == {"deezer": "scanned", "soundcloud": "empty"}


def test_embed_with_source_enforces_centroid_purity(conn, tmp_path):
    # tracks from two platforms exist; embedding with source='deezer' must
    # produce a centroid built ONLY from deezer clips, stamped with the ratio.
    a = _artist(conn)
    _tracks(conn, a, "deezer", 3, "zz-cas-pd")
    _tracks(conn, a, "bandcamp", 2, "zz-cas-pb", tmp_path)
    n = embed_artist_clips(conn, MockEmbedder(dim=8, name="mock-model"), a, source="deezer", signal_ratio=0.3)
    assert n == 3  # bandcamp tracks untouched
    src, ratio, clip_count = conn.execute(
        "SELECT a.embedding_source, ae.signal_ratio, ae.clip_count FROM artist a "
        "JOIN artist_embedding ae ON ae.artist_id = a.id AND ae.model = 'mock-model' WHERE a.id = %s",
        (a,),
    ).fetchone()
    assert src == "deezer"
    assert abs(ratio - 0.3) < 1e-6
    assert clip_count == 3


def test_rerun_with_no_new_clips_still_converges_metadata(conn):
    # An artist embedded BEFORE the cascade existed (ratio/source NULL): a
    # cascade re-run embeds 0 new clips but must backfill the stamps — else
    # supersede targeting never sees the pre-cascade cohort.
    a = _artist(conn)
    _tracks(conn, a, "deezer", 2, "zz-cas-r")
    emb = MockEmbedder(dim=8, name="mock-model")
    embed_artist_clips(conn, emb, a)  # pre-cascade shape: no source, no ratio
    assert conn.execute("SELECT embedding_source FROM artist WHERE id = %s", (a,)).fetchone()[0] is None

    n = embed_artist_clips(conn, emb, a, source="deezer", signal_ratio=0.2)  # cascade re-run
    assert n == 0  # nothing new embedded...
    src, ratio = conn.execute(
        "SELECT a.embedding_source, ae.signal_ratio FROM artist a "
        "JOIN artist_embedding ae ON ae.artist_id = a.id WHERE a.id = %s",
        (a,),
    ).fetchone()
    assert (src, float(ratio)) == ("deezer", 0.2)  # ...but metadata converged


def test_supersede_flips_centroid_wholesale(conn, tmp_path):
    # deezer embeds first (thin); bandcamp supersedes — centroid must rebuild
    # from bandcamp clips only and embedding_source must flip.
    a = _artist(conn)
    _tracks(conn, a, "deezer", 1, "zz-cas-sd")
    _tracks(conn, a, "bandcamp", 3, "zz-cas-sb", tmp_path)
    emb = MockEmbedder(dim=8, name="mock-model")
    embed_artist_clips(conn, emb, a, source="deezer", signal_ratio=0.1)
    embed_artist_clips(conn, emb, a, source="bandcamp", signal_ratio=1.0)
    src, clip_count, ratio, dz_clips = conn.execute(
        "SELECT a.embedding_source, ae.clip_count, ae.signal_ratio, "
        "  (SELECT count(*) FROM clip_embedding ce JOIN audio_track t ON t.id = ce.track_id "
        "   WHERE t.artist_id = a.id AND t.platform = 'deezer') "
        "FROM artist a JOIN artist_embedding ae ON ae.artist_id = a.id AND ae.model = 'mock-model' "
        "WHERE a.id = %s",
        (a,),
    ).fetchone()
    assert src == "bandcamp"
    assert dz_clips == 1  # the deezer clip still exists as a row...
    assert clip_count > dz_clips and ratio == 1.0  # ...but the centroid is bandcamp-only
    # bandcamp tracks window into multiple clips each (90s → 2-3 windows)
    assert clip_count >= 3 * 2
