"""Run the benchmark.

  uv run --group models --group muq python -m pipeline.bench --clips clips/
      → benchmark the real models on a labeled clip set (downloads weights once)

  uv run poe bench            → mock demo (no models needed), shows the harness
"""

from __future__ import annotations

import argparse

from pipeline.bench.clips import load_clip_dir
from pipeline.bench.harness import compare
from pipeline.bench.mock import MockEmbedder
from pipeline.bench.types import BenchResult, Clip, Embedder


def _print(results: list[BenchResult]) -> None:
    print(f"{'model':16} {'clips/s':>10} {'ms/clip':>9} {'intra':>7} {'inter':>7} {'sep':>7} {'p@1':>6}")
    for r in results:
        print(
            f"{r.model:16} {r.clips_per_sec:10.1f} {r.ms_per_clip:9.3f} "
            f"{r.intra_cosine:7.3f} {r.inter_cosine:7.3f} {r.separation:7.3f} {r.p_at_1:6.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="audio-embedder benchmark (O1 quality + O4 throughput)")
    ap.add_argument("--clips", help="dir of labeled clips: <dir>/<artist>/<file>.{wav,flac,mp3,...}")
    ap.add_argument("--device", help="cpu | mps | cuda (default: auto-detect)")
    args = ap.parse_args()

    if not args.clips:
        clips = [Clip(id=f"{a}-{i}", artist_id=a) for a in ("artistA", "artistB", "artistC") for i in range(4)]
        embedders: list[Embedder] = [
            MockEmbedder(name="mock-clean", noise=0.05),
            MockEmbedder(name="mock-noisy", noise=0.6),
        ]
        print("(mock demo — pass --clips <dir> to benchmark real models)\n")
        _print(compare(embedders, clips))
        return

    from pipeline.embedders.registry import available_embedders

    clips = load_clip_dir(args.clips)
    if len(clips) < 2:
        raise SystemExit(f"need >=2 clips under {args.clips}/  (layout: <dir>/<artist>/<file>.wav)")
    models = available_embedders(args.device)
    if not models:
        raise SystemExit("no embedders available — install the `models` (and `muq`) dependency groups")
    n_artists = len({c.artist_id for c in clips})
    print(f"benchmarking {len(models)} model(s) on {len(clips)} clips across {n_artists} artists "
          f"(first run downloads weights)\n")
    _print(compare(models, clips))


if __name__ == "__main__":
    main()
