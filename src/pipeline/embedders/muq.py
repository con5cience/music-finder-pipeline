"""MuQ-MuLan embedder — joint audio-text, recommendation SOTA (needs `muq`).

NOTE: the `muq` package is box-primary; the text-embedding kwarg is verified on
the box on first run.
"""

from __future__ import annotations

import torch

from pipeline.embedders.base import AudioEmbedder


class MuQMuLanEmbedder(AudioEmbedder):
    name = "muq-mulan-large"
    sample_rate = 24000
    supports_text = True
    model_id = "OpenMuQ/MuQ-MuLan-large"

    def _load(self) -> None:
        from muq import MuQMuLan

        self.model = MuQMuLan.from_pretrained(self.model_id).to(self.device).eval()

    def _features(self, wavs: list[torch.Tensor]) -> torch.Tensor:
        batch = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True).to(self.device)
        return self.model(wavs=batch)  # (B, dim) audio embeddings

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        with torch.no_grad():
            feats = self.model(texts=list(texts))
        return torch.nn.functional.normalize(feats.float(), dim=-1).cpu().tolist()
