"""MERT SSL music embedder — mean-pool over layers + time (audio-only)."""

from __future__ import annotations

import torch

from pipeline.embedders.base import AudioEmbedder


class MertEmbedder(AudioEmbedder):
    name = "mert-v1-95M"
    sample_rate = 24000
    model_id = "m-a-p/MERT-v1-95M"

    def _load(self) -> None:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True).to(self.device).eval()

    def _features(self, wavs: list[torch.Tensor]) -> torch.Tensor:
        inputs = self.processor(
            [w.numpy() for w in wavs],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=0)  # (layers+1, B, T, H)
        return hidden.mean(dim=0).mean(dim=1)  # mean over layers, then time → (B, H)
