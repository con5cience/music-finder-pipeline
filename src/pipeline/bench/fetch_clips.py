"""Source the benchmark's own labeled clips — no manual dataset prep.

Given a list of artist *names*, resolve each to a Deezer **artist entity** and
download 3-5 of that entity's own top-track preview MP3s into the
``clips/<slug>/<trackid>.mp3`` layout that :func:`pipeline.bench.clips.load_clip_dir`
already reads. A per-artist ``manifest.json`` records provenance (resolved artist
id, source, per-clip track id + URL) so every label is auditable.

CORRECTNESS (ADR-015 source-correctness law): we never label audio from a
name-keyed *song* search. We resolve name -> artist ID, then keep a top track
only when that artist is both the track's ``artist.id`` and a ``role=="Main"``
contributor — dropping features, remixes credited elsewhere, and compilation
misattribution. An unresolvable / ambiguous name is SKIPPED, never guessed: a
missing artist is harmless to the benchmark; a wrong-artist clip poisons the
same-vs-cross separation metric.

Deezer is the primary source (unauthenticated public REST, label-correct, strong
underground coverage, live-verified 2026). Preview URLs are signed and
short-lived — we download in the same pass and never persist the URL.

HTTP is injected (``get_json`` / ``get_bytes``) so the logic is unit-testable
offline; the defaults are stdlib ``urllib``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import unicodedata
import urllib.parse
import urllib.request
from collections.abc import Callable

DEEZER_API = "https://api.deezer.com"
USER_AGENT = "music-finder-pipeline/0.1 (benchmark clip fetcher)"
GetJson = Callable[[str], dict]
GetBytes = Callable[[str], bytes]


def _ascii_fold(s: str) -> str:
    """Strip diacritics: 'Björk' -> 'Bjork'."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def slugify(name: str) -> str:
    """Filesystem-safe label dir: lowercase ascii, non-alnum -> single dash."""
    folded = _ascii_fold(name).lower()
    out, prev_dash = [], False
    for ch in folded:
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    return "".join(out).strip("-")


def normalize_name(name: str) -> str:
    """Canonical form for name-equality: ascii-folded, casefolded, alnum-only."""
    return "".join(ch for ch in _ascii_fold(name).casefold() if ch.isalnum())


def _default_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (https only, our URLs)
        return json.loads(resp.read().decode("utf-8"))


def _default_get_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return resp.read()


def resolve_deezer_artist(name: str, *, get_json: GetJson = _default_get_json) -> int | None:
    """Name -> Deezer artist id. Require a normalized-name match; tie-break by fans."""
    q = urllib.parse.urlencode({"q": name, "limit": 10})
    data = get_json(f"{DEEZER_API}/search/artist?{q}").get("data", [])
    target = normalize_name(name)
    matches = [a for a in data if normalize_name(a.get("name", "")) == target]
    if not matches:
        return None
    best = max(matches, key=lambda a: a.get("nb_fan", 0))
    return int(best["id"])


def deezer_top_main_tracks(artist_id: int, n: int, *, get_json: GetJson = _default_get_json) -> list[dict]:
    """That artist's top tracks, kept only where they are the Main act and a preview exists."""
    q = urllib.parse.urlencode({"limit": max(n * 2, n)})
    data = get_json(f"{DEEZER_API}/artist/{artist_id}/top?{q}").get("data", [])
    kept: list[dict] = []
    for t in data:
        if not t.get("preview"):
            continue
        if t.get("artist", {}).get("id") != artist_id:
            continue
        is_main = any(c.get("id") == artist_id and c.get("role") == "Main" for c in t.get("contributors", []))
        if not is_main:
            continue
        kept.append(t)
        if len(kept) >= n:
            break
    return kept


def fetch_artist_clips(
    name: str,
    out_root: str,
    *,
    n: int = 12,
    get_json: GetJson = _default_get_json,
    get_bytes: GetBytes = _default_get_bytes,
    sleep: float = 0.0,
) -> dict:
    """Resolve one artist and download up to ``n`` provably-own preview clips.

    Returns a manifest dict (also written to ``<out_root>/<slug>/manifest.json``
    on success). On an unresolvable name, returns ``status='unresolved'`` and
    writes nothing.
    """
    slug = slugify(name)
    artist_id = resolve_deezer_artist(name, get_json=get_json)
    if artist_id is None:
        return {"name": name, "slug": slug, "status": "unresolved", "source": "deezer", "clips": []}

    if sleep:
        time.sleep(sleep)
    tracks = deezer_top_main_tracks(artist_id, n, get_json=get_json)
    if not tracks:
        return {"name": name, "slug": slug, "status": "no_tracks",
                "source": "deezer", "artist_id": artist_id, "clips": []}

    adir = os.path.join(out_root, slug)
    os.makedirs(adir, exist_ok=True)
    clips: list[dict] = []
    for t in tracks:
        tid = t["id"]
        # Preview URL is signed + short-TTL: download immediately, never persist the URL.
        audio = get_bytes(t["preview"])
        with open(os.path.join(adir, f"{tid}.mp3"), "wb") as fh:
            fh.write(audio)
        clips.append({"track_id": tid, "title": t.get("title", ""), "url": t["preview"]})
        if sleep:
            time.sleep(sleep)

    manifest = {"name": name, "slug": slug, "status": "ok", "source": "deezer",
                "artist_id": artist_id, "clips": clips}
    with open(os.path.join(adir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _read_artist_list(path: str) -> list[str]:
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch labeled benchmark clips by artist name (Deezer).")
    ap.add_argument("artists", nargs="*", help="artist names (or use --file)")
    ap.add_argument("--file", help="newline-delimited artist list (# comments allowed)")
    ap.add_argument("--out", default="clips", help="output root (default: clips/)")
    ap.add_argument("-n", type=int, default=12, help="clips per artist (default: 12, for stable cosine)")
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between API calls (politeness)")
    args = ap.parse_args()

    names = list(args.artists)
    if args.file:
        names += _read_artist_list(args.file)
    if not names:
        raise SystemExit("no artists given — pass names or --file <list>")

    ok = skipped = total_clips = 0
    for name in names:
        m = fetch_artist_clips(name, args.out, n=args.n, sleep=args.sleep)
        if m["status"] == "ok":
            ok += 1
            total_clips += len(m["clips"])
            print(f"  ok       {name}  (#{m['artist_id']}, {len(m['clips'])} clips)")
        else:
            skipped += 1
            print(f"  {m['status']:9}{name}")
    print(f"\n{ok} artists, {total_clips} clips -> {args.out}/   ({skipped} skipped)")


if __name__ == "__main__":
    main()
