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
    # Clips per forward pass. Kept small so variable-length models (e.g. MERT,
    # which feeds raw waveforms through conv stacks) don't blow up GPU memory on
    # the longest clip in a batch. Override per-embedder if a model can take more.
    batch_size: int = 8

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
        out: list[torch.Tensor] = []
        # fp16 autocast on cuda: activations halve (throughput audit: VRAM was
        # the cap on embed concurrency — 13.1GB peak at 3). Weights stay fp32;
        # outputs are normalized in fp32 below, so stored vectors are
        # numerically stable. Opt out via PIPELINE_FP16=0.
        import os

        use_amp = self.device.startswith("cuda") and os.environ.get("PIPELINE_FP16", "1") != "0"
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            for i in range(0, len(wavs), self.batch_size):
                feats = self._features(wavs[i : i + self.batch_size])
                feats = torch.nn.functional.normalize(feats.float(), dim=-1)
                out.append(feats.cpu())
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        if not out:
            return []
        return torch.cat(out).tolist()
