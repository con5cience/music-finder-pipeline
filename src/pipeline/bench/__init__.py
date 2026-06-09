"""Audio-model benchmark harness.

Model-agnostic: any Embedder (LAION-CLAP variants, MERT, MusiCNN, …) plugs in via
the `Embedder` protocol. Measures O4 (throughput / cost per clip) and O1
(embedding quality — same-artist clips should cluster tighter than cross-artist).
Real torch embedders run on the box; the MockEmbedder lets the harness be tested
without torch/audio.
"""
