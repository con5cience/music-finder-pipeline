"""Run an Embedder over a labeled clip set and score throughput + quality."""

from __future__ import annotations

import time

from pipeline.bench.metrics import precision_at_1, separation
from pipeline.bench.types import BenchResult, Clip, Embedder


def run_benchmark(embedder: Embedder, clips: list[Clip]) -> BenchResult:
    t0 = time.perf_counter()
    vectors = embedder.embed(clips)
    seconds = time.perf_counter() - t0
    labels = [c.artist_id for c in clips]
    intra, inter, sep = separation(vectors, labels)
    n = len(clips)
    return BenchResult(
        model=embedder.name,
        n_clips=n,
        seconds=seconds,
        clips_per_sec=(n / seconds) if seconds > 0 else float("inf"),
        ms_per_clip=(1000.0 * seconds / n) if n else 0.0,
        intra_cosine=intra,
        inter_cosine=inter,
        separation=sep,
        p_at_1=precision_at_1(vectors, labels),
    )


def compare(embedders: list[Embedder], clips: list[Clip]) -> list[BenchResult]:
    return [run_benchmark(e, clips) for e in embedders]
