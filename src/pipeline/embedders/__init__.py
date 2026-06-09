"""Audio similarity embedders (the primary analysis head).

Benchmarked lineup for music similarity / artist fingerprinting (ADR-016):
  - MuQEmbedder     (OpenMuQ/MuQ-large-msd-iter) — audio-only SSL, DEFAULT (best p@1; CC-BY-NC)
  - MusicFmEmbedder (minzwon/MusicFM, vendored)  — audio-only SSL, MIT permissive fallback
  - ClapEmbedder    (laion/larger_clap_general)  — joint audio-text, Apache, tags ~free
  - MertEmbedder    (m-a-p/MERT-v1-95M)          — SSL baseline (CC-BY-NC)
  - MuQMuLanEmbedder(OpenMuQ/MuQ-MuLan-large)    — joint audio-text, rec SOTA

`registry.get_embedder(name)` builds the production head (default MuQ); the name is
the version stamp for stored embeddings. Each implements the bench `Embedder`
protocol (`name` + `embed(clips)`); joint models expose `embed_text`
(supports_text=True). torch + model deps live in the optional `models` / `muq`
dependency groups.
"""
