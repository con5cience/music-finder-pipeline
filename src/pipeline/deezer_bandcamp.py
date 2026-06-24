"""Deezer→Bandcamp recovery: find Bandcamp pages for artists currently stuck on
the poor Deezer playback widget, gated by AUDIO FINGERPRINT — never by name.

Context: `search_bind` binds name-matched pages for artists with NO audio
(it cannot fingerprint). Our targets are the opposite — Deezer-served artists
that ALREADY have an `artist_embedding` (a MuQ centroid from their Deezer audio).
That anchor lets us do the corroboration `search_bind` always wanted but never
could ("a human (or, later, fingerprint corroboration) confirms"):

  candidate (name-matched) → fetch its audio → MuQ-embed → cosine vs the anchor.

Name match only ever *nominates* a candidate. The bind is the audio. This is
strictly safer than search_bind's exact-name auto-bind — it adds a gate, never
removes one — and is the direct answer to the contamination history (popularity
auto-pick → 4.67M-corpus class; typo auto-bind → 77 wrong centroids).

Modes:
  auto_bind_threshold=None  → REVIEW-ONLY (default): every name-match becomes a
      review_item carrying its audio score; nothing is ever auto-bound. Used for
      validation — humans confirm and the scores set the eventual bar.
  auto_bind_threshold=float → exactly ONE exact-name candidate scoring >= the bar
      auto-binds (Tier-B, audio-evidenced); everything else → review.

Standalone + self-throttled (NOT a queue activity) so the post-launch bulk run
never shares the live embed/rate pool.
"""

from __future__ import annotations

import json

import numpy as np
from psycopg import Connection

from pipeline.search_bind import _edit1, artist_name_keys, normalize_name, search_bandcamp


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def fetch_anchor(conn: Connection, artist_id: str) -> list[float] | None:
    """The artist's MuQ audio centroid (the verification anchor), or None if the
    artist was never embedded (then we cannot fingerprint → not eligible)."""
    row = conn.execute(
        "SELECT embedding::text FROM artist_embedding WHERE artist_id = %s "
        "ORDER BY computed_at DESC LIMIT 1",
        (artist_id,),
    ).fetchone()
    if not row or not row[0]:
        return None
    return [float(x) for x in row[0].strip("[]").split(",") if x]


def deezer_served_unbound(conn: Connection, limit: int) -> list[tuple]:
    """Deezer-served artists eligible for recovery: have Deezer + an embedding,
    no Bandcamp yet, and Bandcamp not already searched."""
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
        ORDER BY a.id
        LIMIT %s
        """,
        (limit,),
    ).fetchall()


def recover_artist_bandcamp(
    conn: Connection,
    artist_id: str,
    display_name: str,
    *,
    embedder,
    searcher=None,
    anchor: list[float] | None = None,
    fuzzy: bool = True,
    auto_bind_threshold: float | None = None,
    _rescore: bool = False,
) -> str:
    """Recover a Bandcamp page for one Deezer-served artist via audio gate.

    `embedder(conn, subdomain) -> list[float] | None`: the candidate's audio
    embedding (real impl fetches its top track + MuQ-embeds; None = unfetchable).
    Returns verdict: skipped | no_anchor | none | review | bound.
    """
    if not _rescore and conn.execute(
        "SELECT 1 FROM search_attempt WHERE artist_id = %s AND platform = 'bandcamp'",
        (artist_id,),
    ).fetchone():
        return "skipped"

    if anchor is None:
        anchor = fetch_anchor(conn, artist_id)
    if anchor is None:
        # No fingerprint → audio gate impossible → never name-bind. Skip without
        # a verdict so it stays eligible once an embedding exists.
        return "no_anchor"

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

    # Score every NAME-matched candidate by audio cosine vs the anchor.
    scored: list[dict] = []
    for c in candidates:
        mt = _match(c["name"])
        if mt is None:
            continue
        emb = embedder(conn, c["platform_id"])
        score = _cosine(anchor, emb) if emb else None
        scored.append({"name": c["name"], "subdomain": c["platform_id"], "match": mt, "audio_score": score})
    scored.sort(key=lambda s: (s["audio_score"] is not None, s["audio_score"] or 0.0), reverse=True)

    verdict: str
    if not scored:
        verdict = "none"
    elif auto_bind_threshold is not None:
        # Auto-bind ONLY a single exact-name candidate clearing the audio bar;
        # any ambiguity (multiple, typo-tier, or below-bar) → review.
        eligible = [s for s in scored if s["match"] == "exact" and s["audio_score"] is not None
                    and s["audio_score"] >= auto_bind_threshold]
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
                    "candidate_name": s["name"], "audio_score": s["audio_score"],
                    "candidates_total": len(candidates),
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
