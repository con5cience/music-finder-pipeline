"""Calibrate the Deezer→Bandcamp auto-bind bar from a human-labelled review
sample. Run AFTER a review-only pass has been labelled.

Input JSON (file via --labeled, else stdin):
  [{"correct": "<subdomain>"|null, "candidates": [<scorecard>, ...]}, ...]
where each <scorecard> is exactly what the binder wrote to review_item.evidence
(match / audio.median / audio.n / margin).

Prints the highest-recall (median_bar, margin_floor, min_tracks) that clears the
target precision — the setting to pass to `poe deezer-bandcamp-recover
--auto-bind-threshold ...`. Pure analysis: no DB, no network.
"""

from __future__ import annotations

import argparse
import json
import sys

from pipeline.deezer_bandcamp import calibrate


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate the Deezer→Bandcamp auto-bind bar")
    ap.add_argument("--labeled", help="labelled JSON file (default: stdin)")
    ap.add_argument("--target-precision", type=float, default=0.99)
    args = ap.parse_args()

    data = json.load(open(args.labeled)) if args.labeled else json.load(sys.stdin)
    res = calibrate(data, target_precision=args.target_precision)
    rec = res["recommended"]

    print(json.dumps({k: res[k] for k in ("target_precision", "total_labeled", "total_correct")}, indent=2))
    if rec is None:
        print(f"\nNo setting reaches precision {args.target_precision} — stay review-only or relax the target.",
              flush=True)
    else:
        print(f"\n→ auto-bind at median>={rec['median_bar']}, margin>={rec['margin_floor']}, "
              f"min_tracks={rec['min_tracks']}: precision {rec['precision']:.3f}, recall {rec['recall']:.3f} "
              f"({rec['correct']}/{rec['binds']} binds).", flush=True)


if __name__ == "__main__":
    main()
