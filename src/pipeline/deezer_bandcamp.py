"""Deezer→Bandcamp recovery: find Bandcamp pages for artists stuck on the poor
Deezer playback widget, gated by an AUDIO CONFIDENCE SCORECARD — never by name.

Context: `search_bind` binds name-matched pages for artists with NO audio (it
cannot fingerprint). Our targets are the opposite — Deezer-served artists that
ALREADY have an `artist_embedding` (a MuQ centroid from their Deezer audio). That
anchor lets us do the corroboration `search_bind` always wanted but never could.

Name only ever NOMINATES a candidate; the bind is the audio. A single cosine is
not enough (validated: genre-twins reach ~0.78 vs a foreign centroid), so the
scorecard stacks independent signals so the bulk auto-resolves and only a small
ambiguous middle needs a human:

  - multi-track agreement: embed several of the candidate's tracks; a coincidental
    single-track match collapses across K tracks (median + min cosine vs anchor).
  - margin / kNN-rank: candidate's cosine to the TARGET minus its cosine to its
    nearest OTHER artist-centroid (HNSW). High margin = specifically-this-artist,
    not generically-this-genre.

This is strictly safer than search_bind's name-only auto-bind — it adds gates,
never removes one — and answers the contamination history (popularity auto-pick →
4.67M-corpus class; typo auto-bind → 77 wrong centroids).

Modes:
  auto_bind_threshold=None  → REVIEW-ONLY (default): every name-match → review_item
      carrying its full scorecard; nothing is ever auto-bound. Calibration on a
      labelled review sample sets the bar before auto-bind is enabled.
  auto_bind_threshold=float → exactly ONE exact-name candidate clearing ALL gates
      (median≥bar, >=min_tracks, margin≥floor) auto-binds (Tier-B, evidenced);
      anything ambiguous → review.

Standalone + self-throttled (NOT a queue activity) so the post-launch bulk run
never shares the live embed/rate pool.
"""

from __future__ import annotations

import json
import statistics

import numpy as np
from psycopg import Connection

from pipeline.fetch_cache import cached_fetch
from pipeline.search_bind import _edit1, artist_name_keys, normalize_name, search_bandcamp
from pipeline.sources.bandcamp import parse_discography, parse_tralbum


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _mean_vec(vecs: list[list[float]]) -> list[float]:
    return np.mean(np.asarray(vecs, dtype=np.float64), axis=0).tolist()


def fetch_anchor(conn: Connection, artist_id: str) -> list[float] | None:
    """The artist's MuQ audio centroid (the verification anchor), or None if the
    artist was never embedded (then we cannot fingerprint → not eligible)."""
    row = conn.execute(
        "SELECT embedding::text FROM artist_embedding "
        "WHERE artist_id = %s AND model = 'muq-large-msd' "
        "ORDER BY computed_at DESC LIMIT 1",
        (artist_id,),
    ).fetchone()
    if not row or not row[0]:
        return None
    return [float(x) for x in row[0].strip("[]").split(",") if x]


def nearest_other_cosine(conn: Connection, vec: list[float], exclude_artist_id: str) -> float | None:
    """Cosine of `vec` to the nearest OTHER artist centroid (HNSW, muq model).
    Powers the margin signal: a genre-generic match scores ~as high here as to
    the target → low margin → review. None if no other centroid is reachable."""
    lit = "[" + ",".join(repr(float(x)) for x in vec) + "]"
    row = conn.execute(
        """
        SELECT 1 - ((embedding)::vector(1024) <=> %s::vector(1024)) AS cos
        FROM artist_embedding
        WHERE model = 'muq-large-msd' AND artist_id <> %s
        ORDER BY (embedding)::vector(1024) <=> %s::vector(1024)
        LIMIT 1
        """,
        (lit, exclude_artist_id, lit),
    ).fetchone()
    return float(row[0]) if row else None


def deezer_served_unbound(conn: Connection, limit: int) -> list[tuple]:
    """Deezer-served artists eligible for recovery: have Deezer + an embedding,
    no Bandcamp yet, Bandcamp not already searched."""
    return conn.execute(
        """
        SELECT a.id, a.display_name FROM artist a
        WHERE EXISTS (SELECT 1 FROM platform_identity p
                      WHERE p.artist_id = a.id AND p.platform = 'deezer')
          AND NOT EXISTS (SELECT 1 FROM platform_identity p
                          WHERE p.artist_id = a.id AND p.platform = 'bandcamp')
          AND EXISTS (SELECT 1 FROM artist_embedding e WHERE e.artist_id = a.id)
          AND NOT EXISTS (SELECT 1 FROM search_attempt s
                          WHERE s.artist_id = a.id AND s.platform = 'bandcamp')
        ORDER BY a.id LIMIT %s
        """,
        (limit,),
    ).fetchall()


def _score_candidate(conn, anchor, candidate, match, *, embedder, nearest_other_fn, exclude_artist_id) -> dict:
    """Build the audio confidence scorecard for one name-matched candidate."""
    sc = {"name": candidate["name"], "subdomain": candidate["platform_id"], "match": match,
          "audio": None, "nearest_other": None, "margin": None, "confidence": None}
    embs = embedder(conn, candidate["platform_id"]) or []
    cosines = sorted((_cosine(anchor, e) for e in embs), reverse=True)
    if not cosines:
        return sc
    median = statistics.median(cosines)
    sc["audio"] = {"n": len(cosines), "median": round(median, 4),
                   "min": round(min(cosines), 4), "max": round(max(cosines), 4)}
    if nearest_other_fn is not None:
        no = nearest_other_fn(conn, _mean_vec(embs), exclude_artist_id)
        if no is not None:
            sc["nearest_other"] = round(no, 4)
            sc["margin"] = round(median - no, 4)
    sc["confidence"] = sc["audio"]["median"]  # headline; calibration maps the full card later
    return sc


def _eligible(sc: dict, *, threshold: float, min_tracks: int, margin_floor: float) -> bool:
    a = sc["audio"]
    return (
        sc["match"] == "exact"
        and a is not None and a["n"] >= min_tracks and a["median"] >= threshold
        and sc["margin"] is not None and sc["margin"] >= margin_floor
    )


def recover_artist_bandcamp(
    conn: Connection,
    artist_id: str,
    display_name: str,
    *,
    embedder,
    searcher=None,
    nearest_other_fn=None,
    anchor: list[float] | None = None,
    fuzzy: bool = True,
    auto_bind_threshold: float | None = None,
    min_tracks: int = 2,
    margin_floor: float = 0.10,
    _rescore: bool = False,
) -> str:
    """Recover a Bandcamp page for one Deezer-served artist via the audio
    scorecard. `embedder(conn, subdomain) -> list[list[float]]` returns the
    candidate's per-track embeddings ([] = unfetchable).
    Returns: skipped | no_anchor | none | review | bound.
    """
    if not _rescore and conn.execute(
        "SELECT 1 FROM search_attempt WHERE artist_id = %s AND platform = 'bandcamp'",
        (artist_id,),
    ).fetchone():
        return "skipped"

    if anchor is None:
        anchor = fetch_anchor(conn, artist_id)
    if anchor is None:
        return "no_anchor"  # no fingerprint → never name-bind; stays eligible later

    searcher = searcher or search_bandcamp
    candidates = searcher(conn, display_name)
    keys = artist_name_keys(conn, artist_id, display_name)

    def _match(cname: str) -> str | None:
        n = normalize_name(cname)
        if n in keys:
            return "exact"
        if fuzzy and any(_edit1(n, k) for k in keys):
            return "typo1"
        return None

    scored = [
        _score_candidate(conn, anchor, c, mt, embedder=embedder,
                         nearest_other_fn=nearest_other_fn, exclude_artist_id=artist_id)
        for c in candidates if (mt := _match(c["name"])) is not None
    ]
    scored.sort(key=lambda s: (s["confidence"] is not None, s["confidence"] or 0.0), reverse=True)

    if not scored:
        verdict = "none"
    elif auto_bind_threshold is not None:
        eligible = [s for s in scored
                    if _eligible(s, threshold=auto_bind_threshold, min_tracks=min_tracks, margin_floor=margin_floor)]
        if len(eligible) == 1:
            s = eligible[0]
            conn.execute(
                """
                INSERT INTO platform_identity (artist_id, platform, platform_id, page_type,
                                               binding_tier, binding_evidence)
                VALUES (%s, 'bandcamp', %s, 'artist', 'B', %s)
                ON CONFLICT DO NOTHING
                """,
                (artist_id, s["subdomain"], json.dumps({
                    "method": "deezer_bandcamp_audio", "query": display_name,
                    "scorecard": s, "candidates_total": len(candidates),
                })),
            )
            verdict = "bound"
        else:
            verdict = _review(conn, artist_id, display_name, scored)
    else:
        verdict = _review(conn, artist_id, display_name, scored)

    conn.execute(
        "INSERT INTO search_attempt (artist_id, platform, query, verdict, candidates) "
        "VALUES (%s, 'bandcamp', %s, %s, %s)",
        (artist_id, display_name, verdict, len(candidates)),
    )
    return verdict


def _review(conn: Connection, artist_id: str, display_name: str, scored: list[dict]) -> str:
    conn.execute(
        """
        INSERT INTO review_item (kind, subject_type, subject_id, reason, evidence, status)
        VALUES ('source_binding', 'artist', %s, %s, %s, 'pending')
        """,
        (artist_id, f"{len(scored)} audio-scored Bandcamp candidate(s) for Deezer-served artist",
         json.dumps({"platform": "bandcamp", "query": display_name,
                     "method": "deezer_bandcamp_audio", "candidates": scored})),
    )
    return "review"


# ---- real candidate embedder (the GPU/network path; review-only run uses it) --

def bandcamp_top_tracks(conn, subdomain: str, k: int = 3, *, fetcher=None, cache_dir=None) -> list:
    """Up to k streamable BcTracks for a candidate page (newest-first), via the
    fetch cache + proxy laws. Single-release artists fall back to the root page."""
    base = f"https://{subdomain}.bandcamp.com"

    def get(path: str) -> bytes:
        return cached_fetch(conn, "bandcamp", base + path, fetcher=fetcher, cache_dir=cache_dir).body

    releases = parse_discography(get("/music"))[:5] or [""]
    tracks: list = []
    for path in releases:
        if len(tracks) >= k:
            break
        parsed = parse_tralbum(get(path))
        if parsed:
            tracks.extend(parsed["tracks"])
    return tracks[:k]


def _group_means(vectors: list[list[float]], counts: list[int]) -> list[list[float]]:
    """Collapse a flat window-vector list back to one mean vector per track."""
    out, idx = [], 0
    for n in counts:
        chunk = vectors[idx:idx + n]
        idx += n
        if chunk:
            out.append(_mean_vec([list(v) for v in chunk]))
    return out


def make_bandcamp_embedder(embedder, *, k: int = 3, fetcher=None, cache_dir=None):
    """Build the candidate embedder: subdomain → up to k per-track MuQ embeddings
    (mean of each track's RMS-peak windows). Reuses the production embed atoms
    (fetch_audio / _decode / _clips_for_track / embedder.embed). Network+GPU —
    exercised by the real run, not the hermetic suite."""

    def _embed(conn, subdomain: str) -> list[list[float]]:
        import tempfile
        from pathlib import Path

        from pipeline.bench.types import Clip
        from pipeline.embed_job import _clips_for_track, _decode, fetch_audio

        tracks = bandcamp_top_tracks(conn, subdomain, k, fetcher=fetcher, cache_dir=cache_dir)
        if not tracks:
            return []
        counts: list[int] = []
        clips: list = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            for i, t in enumerate(tracks):
                try:
                    path = fetch_audio(t.stream_url, workdir)
                except Exception:  # noqa: BLE001 — per-track isolation; a dead stream just drops
                    counts.append(0)
                    continue
                mono, sr = _decode(path)
                segs = _clips_for_track(mono, sr, path, "bandcamp", t.duration_s, workdir, f"t{i}")
                counts.append(len(segs))
                clips.extend(Clip(id=f"{i}:{s}", artist_id="recover", path=p) for s, _e, p in segs)
            if not clips:
                return []
            vectors = [list(v) for v in embedder.embed(clips)]  # GPU — inside the tempdir (reads clip files)
        return _group_means(vectors, counts)

    return _embed


def main() -> None:
    import argparse
    import time

    import psycopg

    from pipeline.config import Settings
    from pipeline.embedders.registry import get_embedder

    ap = argparse.ArgumentParser(
        description="Deezer→Bandcamp recovery (review-only by default; self-throttled, manual)"
    )
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--k", type=int, default=3, help="candidate tracks to embed per artist")
    ap.add_argument("--sleep", type=float, default=1.0, help="politeness pause between artists")
    ap.add_argument("--auto-bind-threshold", type=float, default=None,
                    help="enable auto-bind at this audio median (omit = review-only)")
    ap.add_argument("--min-tracks", type=int, default=2)
    ap.add_argument("--margin-floor", type=float, default=0.10)
    args = ap.parse_args()

    embedder = get_embedder()
    bc_embed = make_bandcamp_embedder(embedder, k=args.k)
    stats: dict[str, int] = {}
    with psycopg.connect(Settings().database_url) as conn:
        done = 0
        while done < args.limit:
            rows = deezer_served_unbound(conn, min(args.batch, args.limit - done))
            if not rows:
                break
            for aid, name in rows:
                try:
                    v = recover_artist_bandcamp(
                        conn, str(aid), name, embedder=bc_embed,
                        nearest_other_fn=nearest_other_cosine,
                        auto_bind_threshold=args.auto_bind_threshold,
                        min_tracks=args.min_tracks, margin_floor=args.margin_floor,
                    )
                except Exception as e:  # noqa: BLE001 — per-artist isolation (mined fleet law)
                    print(f"  error {name!r}: {e}", flush=True)
                    conn.rollback()
                    v = "error"
                stats[v] = stats.get(v, 0) + 1
                done += 1
                time.sleep(args.sleep)
            conn.commit()
            print(f"progress {done}: {stats}", flush=True)
    print(f"FINAL: {stats}", flush=True)


if __name__ == "__main__":
    main()
