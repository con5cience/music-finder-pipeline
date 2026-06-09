"""AudioEmbedder base: audio load/resample + batch normalization (real torch, no model)."""

from __future__ import annotations

import math
import struct
import wave

import pytest

pytest.importorskip("torch")
pytest.importorskip("soundfile")
pytest.importorskip("torchaudio")

import torch  # noqa: E402

from pipeline.bench.types import Clip  # noqa: E402
from pipeline.embedders.base import AudioEmbedder, load_audio_mono  # noqa: E402


class _FixedEmbedder(AudioEmbedder):
    name = "fixed"
    sample_rate = 48000

    def _load(self) -> None:
        pass

    def _features(self, wavs: list[torch.Tensor]) -> torch.Tensor:
        return torch.ones(len(wavs), 4) * 2.0


def _write_sine(path: str, sr: int = 16000, secs: float = 0.2, freq: float = 440.0) -> None:
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(int(sr * secs)):
            w.writeframes(struct.pack("<h", int(3000 * math.sin(2 * math.pi * freq * i / sr))))


def test_load_audio_mono_resamples(tmp_path):
    p = str(tmp_path / "s.wav")
    _write_sine(p, sr=16000, secs=0.2)
    wav = load_audio_mono(p, 48000)
    assert wav.ndim == 1
    assert wav.shape[0] > 8000  # ~0.2s resampled to 48 kHz


def test_embed_normalizes_and_shapes(tmp_path):
    a, b = str(tmp_path / "a.wav"), str(tmp_path / "b.wav")
    _write_sine(a, freq=440)
    _write_sine(b, freq=880)
    vecs = _FixedEmbedder().embed([Clip("a", "x", a), Clip("b", "x", b)])
    assert len(vecs) == 2
    assert len(vecs[0]) == 4
    for v in vecs:
        assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-5
        assert all(abs(x - 0.5) < 1e-6 for x in v)  # ones*2 → normalized 0.5 each
