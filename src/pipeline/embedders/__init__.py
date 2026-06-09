"""Audio similarity embedders (the primary analysis head).

Research-backed shortlist for music similarity/recommendation (ADR-015):
  - ClapEmbedder    (laion/larger_clap_music)   — joint audio-text, tags ~free
  - MertEmbedder    (m-a-p/MERT-v1-95M)          — SSL, strong for recommendation
  - MuQMuLanEmbedder(OpenMuQ/MuQ-MuLan-large)    — joint audio-text, rec SOTA

Each implements the bench `Embedder` protocol (`name` + `embed(clips)`); the joint
models also expose `embed_text` (supports_text=True) for zero-shot tagging. torch
+ model deps live in the optional `models` / `muq` dependency groups.
"""
