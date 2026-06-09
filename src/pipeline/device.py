"""Compute-device selection.

CUDA on the box, MPS/CPU on the Mac, and overridable for tests/mocks via
PIPELINE_DEVICE. torch is imported lazily so the lean foundation env (no torch
installed yet) still resolves to "cpu" instead of failing — GPU/MPS detection
lights up automatically once torch is present (added in the embedding slice).
"""

from __future__ import annotations

import os


def select_device() -> str:
    """Return the torch device string ("cuda" | "mps" | "cpu").

    PIPELINE_DEVICE overrides detection (used by tests and to force CPU in
    integration/E2E runs on the Mac). Falls back to "cpu" when torch is absent.
    """
    forced = os.environ.get("PIPELINE_DEVICE")
    if forced:
        return forced.strip().lower()
    try:
        import torch  # noqa: PLC0415 — lazy: torch is heavy and optional here
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
