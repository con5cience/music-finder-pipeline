"""Demo run: compare two (mock) models on a synthetic clip set and print a table.

`uv run poe bench`. Tomorrow on the box, register real torch Embedders here in
place of the mocks; the harness is unchanged.
"""

from __future__ import annotations

from pipeline.bench.harness import compare
from pipeline.bench.mock import MockEmbedder
from pipeline.bench.types import Clip


def main() -> None:
    clips = [Clip(id=f"{a}-{i}", artist_id=a) for a in ("artistA", "artistB", "artistC") for i in range(4)]
    embedders = [MockEmbedder(name="mock-clean", noise=0.05), MockEmbedder(name="mock-noisy", noise=0.6)]
    results = compare(embedders, clips)

    print(f"{'model':14} {'clips/s':>10} {'ms/clip':>9} {'intra':>7} {'inter':>7} {'sep':>7} {'p@1':>6}")
    for r in results:
        print(
            f"{r.model:14} {r.clips_per_sec:10.1f} {r.ms_per_clip:9.3f} "
            f"{r.intra_cosine:7.3f} {r.inter_cosine:7.3f} {r.separation:7.3f} {r.p_at_1:6.2f}"
        )


if __name__ == "__main__":
    main()
