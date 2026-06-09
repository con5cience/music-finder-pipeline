"""Benchmark types: the audio Clip, the Embedder protocol, and a BenchResult."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Clip:
    """One labeled audio clip. `path` is a real file on the box; `artist_id` is
    the ground-truth label used to score same/cross-artist separation."""

    id: str
    artist_id: str
    path: str | None = None


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns clips into vectors. Real impls load `clip.path` and run
    a model; the harness only needs `name` + `embed`."""

    name: str

    def embed(self, clips: list[Clip]) -> list[list[float]]: ...


@dataclass
class BenchResult:
    model: str
    n_clips: int
    seconds: float
    clips_per_sec: float
    ms_per_clip: float
    intra_cosine: float  # mean cosine between same-artist clips
    inter_cosine: float  # mean cosine between different-artist clips
    separation: float  # intra - inter (higher = better)
    p_at_1: float  # fraction whose nearest neighbour is same-artist
