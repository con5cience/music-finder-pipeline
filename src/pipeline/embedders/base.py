"""AudioEmbedder base: load+mono+resample, lazy model load, mean-batch, L2-norm.

Subclasses implement `_load` (build the model) and `_features` (wavs → (B, dim)
tensor, pre-normalization). The base owns audio I/O and normalization so every
embedder is consistent and the model-specific code stays tiny.
"""

from __future__ import annotations

import abc

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from pipeline.bench.types import Clip
from pipeline.device import select_device


def load_audio_mono(path: str, target_sr: int) -> torch.Tensor:
    """Read an audio file as a mono float32 waveform resampled to target_sr."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, channels)
    wav = torch.from_numpy(np.ascontiguousarray(data.mean(axis=1), dtype=np.float32))
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)
    return wav


class AudioEmbedder(abc.ABC):
    name: str
    sample_rate: int
    supports_text: bool = False

    def __init__(self, device: str | None = None) -> None:
        self.device = device or select_device()
        self._loaded = False

    @abc.abstractmethod
    def _load(self) -> None: ...

    @abc.abstractmethod
    def _features(self, wavs: list[torch.Tensor]) -> torch.Tensor:
        """Return (batch, dim) embeddings before normalization."""

    def _ensure(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def embed(self, clips: list[Clip]) -> list[list[float]]:
        self._ensure()
        wavs = [load_audio_mono(c.path, self.sample_rate) for c in clips if c.path]
        with torch.no_grad():
            feats = self._features(wavs)
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        return feats.cpu().tolist()
