"""Discover constructible embedders — those whose model deps are installed.

`available_embedders()` returns the embedders this environment can actually run
(transformers present → CLAP/MERT; muq present → MuQ-MuLan). Weights load lazily
on first `embed`. Used by the bench to run real models on the box.
"""

from __future__ import annotations

import importlib.util

from pipeline.bench.types import Embedder


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def available_embedders(device: str | None = None) -> list[Embedder]:
    out: list[Embedder] = []
    if _has("transformers"):
        from pipeline.embedders.clap import ClapEmbedder
        from pipeline.embedders.mert import MertEmbedder

        out.append(ClapEmbedder(device))
        out.append(MertEmbedder(device))
    if _has("muq"):
        from pipeline.embedders.muq import MuQMuLanEmbedder

        out.append(MuQMuLanEmbedder(device))
    return out
