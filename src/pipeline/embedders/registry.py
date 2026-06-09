"""Embedder discovery + selection.

`get_embedder(name)` constructs the production analysis head (default: MuQ, see
ADR-016) — its `name` is the version stamp to store alongside each embedding so a
future model swap is a clean re-embed. `available_embedders()` returns every
constructible embedder, used by the bench to run all models. An embedder is
constructible when its model deps are installed (transformers → CLAP/MERT/MusicFM;
muq → MuQ/MuQ-MuLan); weights load lazily on first `embed`.
"""

from __future__ import annotations

import importlib.util

from pipeline.bench.types import Embedder

# Default analysis head. MuQ audio-only SSL wins the bench (ADR-016); swap to the
# MIT-licensed "musicfm-msd" if MuQ's CC-BY-NC weights ever become a commercial blocker.
DEFAULT_EMBEDDER = "muq-large-msd"


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _embedder_classes() -> dict[str, type[Embedder]]:
    """Map name → embedder class for everything this environment can construct."""
    classes: dict[str, type[Embedder]] = {}
    if _has("transformers"):
        from pipeline.embedders.clap import ClapEmbedder
        from pipeline.embedders.mert import MertEmbedder

        classes[ClapEmbedder.name] = ClapEmbedder
        classes[MertEmbedder.name] = MertEmbedder
        if _has("einops"):  # MusicFM's vendored code needs einops + wav2vec2_conformer
            from pipeline.embedders.musicfm import MusicFmEmbedder

            classes[MusicFmEmbedder.name] = MusicFmEmbedder
    if _has("muq"):
        from pipeline.embedders.muq import MuQEmbedder, MuQMuLanEmbedder

        classes[MuQEmbedder.name] = MuQEmbedder
        classes[MuQMuLanEmbedder.name] = MuQMuLanEmbedder
    return classes


def get_embedder(name: str | None = None, device: str | None = None) -> Embedder:
    """Construct one embedder by name (default: DEFAULT_EMBEDDER)."""
    name = name or DEFAULT_EMBEDDER
    classes = _embedder_classes()
    cls = classes.get(name)
    if cls is None:
        raise ValueError(f"unknown embedder {name!r}; available: {sorted(classes)}")
    return cls(device)


def available_embedders(device: str | None = None) -> list[Embedder]:
    """Every constructible embedder (the bench runs all of them)."""
    return [cls(device) for cls in _embedder_classes().values()]
