"""RMS-peak window selection (ADR-017 §2: synthetic previews from full tracks).

Property-style tests on constructed signals: windows must land on energy,
never overlap, respect bounds, and degrade gracefully on short audio.
"""

from __future__ import annotations

import numpy as np

from pipeline.windows import peak_windows

SR = 1000  # tests use a toy sample rate; the math is rate-agnostic


def _signal(duration_s: int, loud_spans: list[tuple[int, int]], quiet: float = 0.01, loud: float = 0.5):
    """Quiet noise with loud bursts at the given (start_s, end_s) spans."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(duration_s * SR) * quiet
    for a, b in loud_spans:
        x[a * SR:b * SR] = rng.standard_normal((b - a) * SR) * loud
    return x.astype(np.float32)


def test_windows_land_on_the_energy():
    x = _signal(300, loud_spans=[(100, 140)])
    wins = peak_windows(x, SR, k=1, window_s=30)
    assert len(wins) == 1
    start, end = wins[0]
    assert 95 <= start <= 115  # the loud span dominates; window centers on it
    assert end - start == 30


def test_windows_never_overlap():
    x = _signal(300, loud_spans=[(50, 90), (150, 190), (250, 290)])
    wins = peak_windows(x, SR, k=4, window_s=30)
    assert len(wins) >= 3
    ordered = sorted(wins)
    for (_s1, e1), (s2, _e2) in zip(ordered, ordered[1:], strict=False):
        assert s2 >= e1  # no overlap


def test_windows_within_bounds_and_fixed_length():
    x = _signal(120, loud_spans=[(0, 10), (110, 120)])  # energy at the edges
    for start, end in peak_windows(x, SR, k=3, window_s=30):
        assert 0 <= start
        assert end <= 120
        assert end - start == 30


def test_short_track_yields_single_full_window():
    # shorter than 2 windows: one window covering what exists
    x = _signal(45, loud_spans=[(10, 30)])
    wins = peak_windows(x, SR, k=4, window_s=30)
    assert len(wins) == 1
    assert wins[0][1] - wins[0][0] == 30


def test_track_shorter_than_window_is_one_clip_of_full_length():
    x = _signal(20, loud_spans=[(5, 15)])
    wins = peak_windows(x, SR, k=4, window_s=30)
    assert wins == [(0, 20)]


def test_k_caps_window_count():
    x = _signal(600, loud_spans=[(i, i + 20) for i in range(0, 600, 60)])
    assert len(peak_windows(x, SR, k=4, window_s=30)) == 4


def test_deterministic():
    x = _signal(300, loud_spans=[(100, 140), (200, 240)])
    assert peak_windows(x, SR, k=3) == peak_windows(x, SR, k=3)
