"""Acoustic auto-adjudication of pending candidate bindings (2026-06-12).

The operator should not hand-disambiguate 3,400 name-match candidates when
the audio can testify. For each pending search-candidate review item whose
artist already has a centroid (embedded from a confirmed source), probe up
to PROBE_CLIPS preview clips per candidate page, embed them, and compare
against the artist's centroid:

  cosine >= CONFIRM (0.8)  and it's the ONLY confirmed candidate
      -> approve the item (decision.method=auto_coherence; the poller binds)
  every candidate < REJECT (0.5)
      -> reject the item (acoustically none match)
  otherwise (gray zone, or 2+ confirmed = probable re-uploads/duplicates)
      -> leave pending, ANNOTATE each candidate with its cosine so the
         human pass is a glance, not an investigation

Thresholds come from the measured coherence distribution: same-act mass at
0.7-0.95, impostors below 0.5, an EMPTY gap between. Probes are read-only
(audio_track is never written), page/API fetches ride cached_fetch (proxy
law) and self-throttle to the platform's io_rate budget.

Run inside a maintenance window (host GPU; worker-gpu stopped):
  uv run poe adjudicate -- --limit 50          # pilot
  uv run poe adjudicate -- --limit 100000      # the queue
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
from psycopg import Connection

from pipeline.fetch_cache import cached_fetch
from pipeline.queues import PLATFORMS

DEFAULT_MODEL = "muq-large-msd"
CONFIRM = 0.8
REJECT = 0.5
PROBE_CLIPS = 3
_CLIP_S = 30.0

_last_fetch: dict[str, float] = {}


def _polite(platform: str) -> None:
    """Self-throttle page/API probes to the platform's server-side budget —
    host-side bulk runs bypass the Temporal rate caps, so we keep them here."""
    budget = max(PLATFORMS[platform].io_rate, 0.2)
    wait = _last_fetch.get(platform, 0.0) + 1.0 / budget - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_fetch[platform] = time.monotonic()


def probe_candidate_urls(conn: Connection, platform: str, platform_id: str,
                         *, max_clips: int = PROBE_CLIPS) -> list[str]:
    """Up to max_clips audio URLs for a candidate page. READ-ONLY: never
    touches audio_track/platform_identity. Returns [] when unprobeable."""
    try:
        if platform == "deezer":
            from pipeline.sources.deezer import _API, parse_tracks

            _polite(platform)
            body = cached_fetch(conn, "deezer", f"{_API}/artist/{platform_id}/top?limit=10").body
            return [t.preview_url for t in parse_tracks(body, platform_id)[:max_clips]]
        if platform == "soundcloud":
            from pipeline.sources.soundcloud import (
                _API, _oauth_fetcher, parse_tracks, resolve_stream_url)
            import urllib.parse

            _polite(platform)
            res = cached_fetch(conn, "soundcloud", f"{_API}/resolve?url=" + urllib.parse.quote(
                f"https://soundcloud.com/{platform_id}", safe=""), fetcher=_oauth_fetcher)
            if res.status == 404:
                return []
            user = json.loads(res.body)
            _polite(platform)
            res = cached_fetch(conn, "soundcloud",
                               f"{_API}/users/{user['id']}/tracks?limit=20&linked_partitioning=true",
                               fetcher=_oauth_fetcher)
            urls = []
            for t in parse_tracks(res.body):
                s = resolve_stream_url(str(t["id"]))
                if s:
                    urls.append(s)
                if len(urls) >= max_clips:
                    break
            return urls
        if platform == "bandcamp":
            from pipeline.sources.bandcamp import parse_discography, parse_tralbum

            _polite(platform)
            base = f"https://{platform_id}.bandcamp.com"
            body = cached_fetch(conn, "bandcamp", f"{base}/music").body
            urls: list[str] = []
            for rel in parse_discography(body)[:2]:
                _polite(platform)
                tral = parse_tralbum(cached_fetch(conn, "bandcamp", base + rel).body)
                for t in (tral or {}).get("tracks", []):
                    urls.append(t.stream_url)
                    if len(urls) >= max_clips:
                        return urls
            return urls
    except Exception:  # noqa: BLE001 — an unprobeable candidate is just gray
        return []
    return []


def _center_clip(path: str, workdir: Path) -> str | None:
    """A centered <=30s wav slice — previews pass through, full tracks trim."""
    import soundfile as sf

    try:
        info = sf.info(path)
        if info.duration <= _CLIP_S + 1:
            return path
        start = int((info.duration - _CLIP_S) / 2 * info.samplerate)
        data, sr = sf.read(path, start=start, frames=int(_CLIP_S * info.samplerate))
        out = str(workdir / (Path(path).stem + "-c30.wav"))
        sf.write(out, data, sr)
        return out
    except Exception:  # noqa: BLE001
        return None


def candidate_cosine(conn: Connection, centroid: np.ndarray, platform: str,
                     platform_id: str, embedder, workdir: Path, fetch) -> float | None:
    """Mean clip-vector cosine vs the artist centroid; None = unprobeable."""
    urls = probe_candidate_urls(conn, platform, platform_id)
    clips = []
    for i, url in enumerate(urls):
        try:
            raw = fetch(url, workdir)
            clip = _center_clip(raw, workdir)
            if clip:
                clips.append(clip)
        except Exception:  # noqa: BLE001
            continue
    if not clips:
        return None
    from pipeline.bench.types import Clip

    vecs = np.asarray(embedder.embed(
        [Clip(id=f"probe-{i}", artist_id="probe", path=p) for i, p in enumerate(clips)]
    ), dtype=np.float32)
    mean = vecs.mean(axis=0)
    mean /= max(float(np.linalg.norm(mean)), 1e-9)
    return float(np.dot(mean, centroid))


def adjudicate_pending(conn: Connection, *, embedder, limit: int = 50,
                       model: str = DEFAULT_MODEL, fetch=None,
                       confirm: float = CONFIRM, reject: float = REJECT,
                       commit_each: bool = False) -> dict:
    """commit_each=True for the long live run (crash keeps progress); tests
    and library callers keep transaction control (the conftest rollback
    contract — a mid-call commit PERSISTS fixtures into the shared test DB,
    the 2026-06-12 14-test contamination)."""
    if fetch is None:
        from pipeline.embed_job import fetch_audio as fetch
    items = conn.execute(
        """
        SELECT ri.id, ri.subject_id, ri.evidence, ae.embedding::text
        FROM review_item ri
        JOIN artist_embedding ae ON ae.artist_id = ri.subject_id AND ae.model = %s
        WHERE ri.kind = 'source_binding' AND ri.status = 'pending'
          AND ri.reason NOT IN ('mb_shared_url', 'source_coherence')
          AND ri.evidence ? 'candidates'
          AND NOT (ri.evidence ? 'url_collision') AND NOT (ri.evidence ? 'fp_collision')
          AND COALESCE((ri.evidence->>'adjudicated')::bool, false) IS NOT TRUE
        ORDER BY ri.created_at, ri.id LIMIT %s
        """,
        (model, limit),
    ).fetchall()
    out = {"approved": 0, "rejected": 0, "annotated": 0, "unprobeable": 0}
    for rid, artist_id, evidence, emb_text in items:
        centroid = np.asarray(json.loads(emb_text), dtype=np.float32)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
        platform = evidence.get("platform")
        cands = evidence.get("candidates") or []
        with tempfile.TemporaryDirectory(prefix="adjudicate-") as tmp:
            scored = []
            for c in cands:
                cos = candidate_cosine(conn, centroid, platform, str(c.get("platform_id")),
                                       embedder, Path(tmp), fetch)
                scored.append({**c, "acoustic": None if cos is None else round(cos, 4)})
        known = [c for c in scored if c["acoustic"] is not None]
        confirmed = [c for c in known if c["acoustic"] >= confirm]
        evidence["candidates"] = scored
        evidence["adjudicated"] = True
        if not known:
            out["unprobeable"] += 1
            conn.execute("UPDATE review_item SET evidence = %s WHERE id = %s AND status = 'pending'",
                         (json.dumps(evidence), rid))
        elif len(confirmed) == 1 and len(known) == len(cands):
            c = confirmed[0]
            evidence["decision"] = {"platform": platform, "platform_id": str(c["platform_id"]),
                                    "method": "auto_coherence", "cosine": c["acoustic"]}
            conn.execute(
                "UPDATE review_item SET status='approved', evidence=%s, "
                "note='auto: candidate audio matches artist centroid' "
                "WHERE id=%s AND status='pending'",
                (json.dumps(evidence), rid))
            out["approved"] += 1
        elif known and all(c["acoustic"] < reject for c in known) and len(known) == len(cands):
            conn.execute(
                "UPDATE review_item SET status='rejected', evidence=%s, resolved_at=now(), "
                "note='auto: no candidate sounds like this artist' "
                "WHERE id=%s AND status='pending'",
                (json.dumps(evidence), rid))
            out["rejected"] += 1
        else:
            # gray zone / multi-confirmed / partly unprobeable: human decides,
            # but now with per-candidate acoustics on the card
            conn.execute("UPDATE review_item SET evidence = %s WHERE id = %s AND status = 'pending'",
                         (json.dumps(evidence), rid))
            out["annotated"] += 1
        if commit_each:
            conn.commit()
    out["processed"] = len(items)
    return out


def main() -> None:
    import argparse

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="acoustically adjudicate pending candidate bindings")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--confirm", type=float, default=CONFIRM)
    ap.add_argument("--reject", type=float, default=REJECT)
    args = ap.parse_args()
    from pipeline.embedders.registry import get_embedder

    embedder = get_embedder()
    with psycopg.connect(Settings().database_url) as conn:
        print(adjudicate_pending(conn, embedder=embedder, limit=args.limit,
                                 confirm=args.confirm, reject=args.reject,
                                 commit_each=True))


if __name__ == "__main__":
    main()
