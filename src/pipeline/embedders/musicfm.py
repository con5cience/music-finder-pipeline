"""MusicFM embedder — audio-only music SSL, MIT-licensed (permissive fallback).

MusicFM has no pip/transformers package, so the model code is vendored under
`_vendor/musicfm/` (see its VENDOR.md). This is the commercial-safe swap target
for MuQ: MIT weights (HF `minzwon/MusicFM`) vs MuQ's CC-BY-NC. Not the default
(MuQ scores ~0.03 higher) — kept wired so a license-driven swap is trivial.

Layer 7 of the 12-layer Conformer was the best similarity layer in the bench
(ADR-016); deeper layers get anisotropic and score worse.
"""

from __future__ import annotations

import torch

from pipeline.embedders.base import AudioEmbedder


class MusicFmEmbedder(AudioEmbedder):
    name = "musicfm-msd"
    sample_rate = 24000
    model_id = "minzwon/MusicFM"
    layer = 7  # best similarity layer in the bench (see ADR-016)
    batch_size = 4  # raw-waveform conv stack — keep small for the 15.5 GB GPU

    def _load(self) -> None:
        from huggingface_hub import hf_hub_download

        from pipeline.embedders._vendor.musicfm.model.musicfm_25hz import MusicFM25Hz

        stat = hf_hub_download(self.model_id, "msd_stats.json")
        ckpt = hf_hub_download(self.model_id, "pretrained_msd.pt")
        self.model = MusicFM25Hz(is_flash=False, stat_path=stat, model_path=ckpt).to(self.device).eval()

    def _features(self, wavs: list[torch.Tensor]) -> torch.Tensor:
        batch = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True).to(self.device)
        hidden = self.model.get_latent(batch, layer_ix=self.layer)  # (B, T, H)
        return hidden.mean(dim=1)  # mean over time → (B, H)
