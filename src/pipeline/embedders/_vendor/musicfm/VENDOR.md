# Vendored: MusicFM (audio-only music SSL embedder)

Source: https://github.com/minzwon/musicfm @ b83ebedb401bcef639b26b05c0c8bee1dc2dfe71
License: MIT (commercial-use OK) — see upstream repo LICENSE.
Weights: Hugging Face `minzwon/MusicFM` (pretrained_msd.pt + msd_stats.json), MIT.

Why vendored: MusicFM has no pip/transformers package — the only load path is the
authors' repo code. We copy the minimal inference set and rewrite `from musicfm.modules`
-> relative imports. `flash_conformer.py` is intentionally omitted (only used under
is_flash=True; we run is_flash=False, which uses transformers' wav2vec2_conformer).

Role: permissive (MIT) fallback embedder for if/when MuQ's CC-BY-NC weights become a
commercial blocker. See ADR-016. Not the default (MuQ is). Do not edit beyond the
import rewrite; re-vendor from upstream to update.
