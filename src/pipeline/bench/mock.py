"""MockEmbedder — deterministic, no torch/audio.

Same-artist clips share a base vector plus a small clip-specific perturbation, so
a correct harness reports high separation + p@1. Tunable `noise` lets tests model
a good model (low noise → tight clusters) vs a poor one (high noise → blurred).
"""

from __future__ import annotations

import hashlib

from pipeline.bench.types import Clip


class MockEmbedder:
    def __init__(self, dim: int = 16, noise: float = 0.05, name: str = "mock") -> None:
        self.dim = dim
        self.noise = noise
        self.name = name

    def _vec(self, key: str, scale: float) -> list[float]:
        h = hashlib.sha256(key.encode()).digest()
        return [scale * ((h[i % len(h)] / 255.0) - 0.5) for i in range(self.dim)]

    def embed(self, clips: list[Clip]) -> list[list[float]]:
        out: list[list[float]] = []
        for c in clips:
            base = self._vec(f"artist:{c.artist_id}", 1.0)
            pert = self._vec(f"clip:{c.id}", self.noise)
            out.append([b + p for b, p in zip(base, pert, strict=True)])
        return out
