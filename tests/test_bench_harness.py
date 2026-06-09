"""Harness: run/compare an Embedder over a labeled clip set."""

from __future__ import annotations

from pipeline.bench.harness import compare, run_benchmark
from pipeline.bench.mock import MockEmbedder
from pipeline.bench.types import Clip


def _clips(per_artist: int = 3) -> list[Clip]:
    return [Clip(id=f"{a}-{i}", artist_id=a) for a in ("x", "y", "z") for i in range(per_artist)]


def test_clean_mock_clusters_well():
    res = run_benchmark(MockEmbedder(name="clean", noise=0.02), _clips())
    assert res.n_clips == 9
    assert res.separation > 0.2
    assert res.p_at_1 == 1.0
    assert res.clips_per_sec > 0


def test_noisy_mock_separates_worse_than_clean():
    clips = _clips(4)
    clean = run_benchmark(MockEmbedder(name="clean", noise=0.02), clips)
    noisy = run_benchmark(MockEmbedder(name="noisy", noise=0.8), clips)
    assert clean.separation > noisy.separation


def test_compare_one_result_per_model_in_order():
    res = compare([MockEmbedder(name="m1"), MockEmbedder(name="m2")], _clips())
    assert [r.model for r in res] == ["m1", "m2"]
