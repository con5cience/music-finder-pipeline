"""Discover constructible embedders (those whose deps import).

`available_embedders()` returns the embedders the current environment can build —
torch present → CLAP/MERT/MuQ classes import; the heavier transformers/muq weights
load lazily on first `embed`. Used by the bench to run real models on the box.
"""

from __future__ import annotations

from pipeline.bench.types import Embedder


def available_embedders(device: str | None = None) -> list[Embedder]:
    out: list[Embedder] = []
    for factory in (_clap, _mert, _muq):
        try:
            out.append(factory(device))
        except Exception:  # noqa: BLE001 — missing optional deps → just omit it
            pass
    return out


def _clap(device: str | None) -> Embedder:
    from pipeline.embedders.clap import ClapEmbedder

    return ClapEmbedder(device)


def _mert(device: str | None) -> Embedder:
    from pipeline.embedders.mert import MertEmbedder

    return MertEmbedder(device)


def _muq(device: str | None) -> Embedder:
    from pipeline.embedders.muq import MuQMuLanEmbedder

    return MuQMuLanEmbedder(device)
