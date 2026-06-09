"""RMS-peak window selection: cut Deezer-like "synthetic previews" from full
tracks (ADR-017 §2). Fallback sources must resemble the preview corpus —
fixed-offset sampling lands on intros/outros; energy ranking lands on hooks.

Greedy non-overlapping selection over a sliding RMS profile: rank all window
positions by RMS, take the best, suppress overlapping candidates, repeat up to
k. Pure numpy on a mono waveform; rate-agnostic.
"""

from __future__ import annotations

import numpy as np

_HOP_S = 5  # candidate window every 5s — fine enough to find hooks, cheap


def peak_windows(
    mono: np.ndarray,
    sample_rate: int,
    k: int = 4,
    window_s: int = 30,
) -> list[tuple[int, int]]:
    """Top-k non-overlapping (start_s, end_s) windows by RMS energy.

    Tracks shorter than one window return a single (0, duration) clip;
    tracks shorter than two windows return one best-effort window.
    """
    n = len(mono)
    dur_s = n // sample_rate
    if dur_s <= window_s:
        return [(0, max(dur_s, 1))]

    win = window_s * sample_rate
    hop = _HOP_S * sample_rate
    starts = np.arange(0, n - win + 1, hop)
    # RMS per candidate via cumulative energy (O(n), no window copies)
    csum = np.concatenate(([0.0], np.cumsum(mono.astype(np.float64) ** 2)))
    energy = csum[starts + win] - csum[starts]

    chosen: list[int] = []
    order = np.argsort(energy)[::-1]
    for idx in order:
        if len(chosen) == k:
            break
        s = int(starts[idx])
        if all(abs(s - c) >= win for c in chosen):
            chosen.append(s)

    return sorted((s // sample_rate, s // sample_rate + window_s) for s in chosen)
