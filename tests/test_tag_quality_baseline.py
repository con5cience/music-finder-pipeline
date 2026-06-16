"""bucketize: cumulative assignment-rate histogram for the ADR-020 baseline."""

from __future__ import annotations

from pipeline.tag_quality_baseline import bucketize


def test_bucketize_cumulative():
    rates = [0.20, 0.12, 0.08, 0.03, 0.005, 0.18]
    h = bucketize(rates)
    assert h["ge_15pct"] == 2  # 0.20, 0.18
    assert h["ge_10pct"] == 3  # + 0.12
    assert h["ge_5pct"] == 4  # + 0.08
    assert h["ge_1pct"] == 5  # + 0.03 (0.005 excluded)
    assert h["total_tags"] == 6


def test_bucketize_boundaries_are_inclusive():
    h = bucketize([0.15, 0.10, 0.05, 0.01])
    assert h["ge_15pct"] == 1
    assert h["ge_10pct"] == 2
    assert h["ge_5pct"] == 3
    assert h["ge_1pct"] == 4


def test_bucketize_empty():
    assert bucketize([]) == {"ge_15pct": 0, "ge_10pct": 0, "ge_5pct": 0, "ge_1pct": 0, "total_tags": 0}
