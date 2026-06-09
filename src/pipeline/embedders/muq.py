"""MuQ embedders (need the `muq` package, box-primary).

- MuQEmbedder      — audio-only SSL (OpenMuQ/MuQ-large-msd-iter). The DEFAULT
  analysis head: best similarity/p@1 in the bench (ADR-016). Weights are
  CC-BY-NC-4.0 (non-commercial) — fine for now; the MIT MusicFmEmbedder is the
  permissive swap target if that ever blocks us.
- MuQMuLanEmbedder — joint audio-text, recommendation SOTA; `embed_text` gives
  zero-shot tagging.
"""

from __future__ import annotations

import torch

from pipeline.embedders.base import AudioEmbedder


class MuQEmbedder(AudioEmbedder):
    """Audio-only MuQ SSL backbone — mean-pool the final hidden state over time."""

    name = "muq-large-msd"
    sample_rate = 24000
    model_id = "OpenMuQ/MuQ-large-msd-iter"

    def _load(self) -> None:
        from muq import MuQ

        self.model = MuQ.from_pretrained(self.model_id).to(self.device).eval()

    def _features(self, wavs: list[torch.Tensor]) -> torch.Tensor:
        batch = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True).to(self.device)
        out = self.model(batch)
        return out.last_hidden_state.mean(dim=1)  # (B, T, H) → (B, H)


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
