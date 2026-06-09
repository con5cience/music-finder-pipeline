"""LAION-CLAP music embedder (joint audio-text → tags ~free via embed_text)."""

from __future__ import annotations

import torch

from pipeline.embedders.base import AudioEmbedder


def _embedding(out: object) -> torch.Tensor:
    """Pull the projected embedding tensor out of a CLAP features call.

    transformers 5.x returns a BaseModelOutputWithPooling (`pooler_output`, 512-d);
    older versions returned a tensor or `*_embeds`. Handle all.
    """
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("audio_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    raise TypeError(f"unexpected CLAP features output: {type(out).__name__}")


class ClapEmbedder(AudioEmbedder):
    name = "laion-clap-music"
    sample_rate = 48000
    supports_text = True
    model_id = "laion/larger_clap_music"

    def _load(self) -> None:
        from transformers import ClapModel, ClapProcessor

        self.processor = ClapProcessor.from_pretrained(self.model_id)
        self.model = ClapModel.from_pretrained(self.model_id).to(self.device).eval()

    def _features(self, wavs: list[torch.Tensor]) -> torch.Tensor:
        inputs = self.processor(
            audio=[w.numpy() for w in wavs],  # transformers 5.x renamed `audios` → `audio`
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return _embedding(self.model.get_audio_features(**inputs))

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Zero-shot tag/text embeddings in the shared space (for tagging)."""
        self._ensure()
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = _embedding(self.model.get_text_features(**inputs))
        return torch.nn.functional.normalize(feats.float(), dim=-1).cpu().tolist()
