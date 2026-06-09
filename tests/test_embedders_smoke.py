"""Real-model smoke tests — gated (download weights). Run on the box:

    PIPELINE_RUN_MODEL_TESTS=1 uv run --group models pytest -q tests/test_embedders_smoke.py
"""

from __future__ import annotations

import math
import os
import struct
import wave

import pytest

if os.environ.get("PIPELINE_RUN_MODEL_TESTS") != "1":
    pytest.skip("set PIPELINE_RUN_MODEL_TESTS=1 to run real-model smoke tests", allow_module_level=True)

pytest.importorskip("torch")
pytest.importorskip("transformers")

from pipeline.bench.types import Clip  # noqa: E402
from pipeline.embedders.clap import ClapEmbedder  # noqa: E402
from pipeline.embedders.mert import MertEmbedder  # noqa: E402


def _write_sine(path: str, sr: int = 24000, secs: float = 3.0, freq: float = 440.0) -> None:
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(int(sr * secs)):
            w.writeframes(struct.pack("<h", int(3000 * math.sin(2 * math.pi * freq * i / sr))))


@pytest.mark.parametrize("ctor", [ClapEmbedder, MertEmbedder])
def test_real_embedder_returns_normalized_vector(tmp_path, ctor):
    p = str(tmp_path / "c.wav")
    _write_sine(p)
    vecs = ctor(device="cpu").embed([Clip("c", "x", p)])
    assert len(vecs) == 1
    assert len(vecs[0]) > 0
    assert abs(math.sqrt(sum(x * x for x in vecs[0])) - 1.0) < 1e-3
