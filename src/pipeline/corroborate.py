"""Acoustic corroboration of the single-source blind spot (2026-06-12 audit).

1,407 artists are embedded entirely from ONE machine-guessed (B-tier)
binding — the coherence gate cannot judge them (no second source). Most have
OTHER MB-declared (A-tier) pages whose audio is ground-truth-ish: probe a few
clips from those pages and compare against the artist's centroid.

  cosine >= CONFIRM  -> the B binding is corroborated by MB-declared audio:
                        promote B->C (method history kept; becomes
                        MB-payload eligible per the A/C provenance gate)
  cosine <  REJECT   -> the centroid does NOT sound like the artist's
                        MB-declared audio: file a source_coherence flag
                        (the existing gate holds publish + MB submission)
  otherwise/silent   -> recorded as unprobeable/gray; re-runs skip

Same thresholds as the coherence gate (the measured empty gap). Run with
the adjudicator's coexistence pattern — never alongside it (politeness
budgets are per-process):  uv run poe corroborate --limit 100000
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from psycopg import Connection

from pipeline.adjudicate import CONFIRM, REJECT, candidate_cosine

DEFAULT_MODEL = "muq-large-msd"
PROBEABLE = ("deezer", "bandcamp", "soundcloud", "apple_music", "youtube")


def corroborate_blind_spot(conn: Connection, *, embedder, limit: int = 50,
                           model: str = DEFAULT_MODEL, fetch=None,
                           confirm: float = CONFIRM, reject: float = REJECT,
                           commit_each: bool = False,
                           platforms: tuple = PROBEABLE,
                           retry_status: str | None = None) -> dict:
    if fetch is None:
        from pipeline.embed_job import fetch_audio as fetch
    if retry_status:
        conn.execute(
            """UPDATE platform_identity
               SET binding_evidence = binding_evidence - 'corroboration'
               WHERE binding_tier = 'B'
                 AND binding_evidence->'corroboration'->>'status' = %s""",
            (retry_status,))
    rows = conn.execute(
        """
        SELECT a.id, pi_b.id, ae.embedding::text,
               (SELECT json_agg(json_build_object('platform', p2.platform, 'pid', p2.platform_id))
                FROM platform_identity p2
                WHERE p2.artist_id = a.id AND p2.binding_tier = 'A'
                  AND p2.platform = ANY(%s)) AS a_pages
        FROM artist a
        JOIN platform_identity pi_b ON pi_b.artist_id = a.id
          AND pi_b.platform = a.embedding_source AND pi_b.binding_tier = 'B'
          AND pi_b.binding_evidence->'corroboration' IS NULL
        JOIN artist_embedding ae ON ae.artist_id = a.id AND ae.model = %s
        WHERE NOT EXISTS (SELECT 1 FROM audio_track t WHERE t.artist_id = a.id
                          AND t.platform <> a.embedding_source)
        LIMIT %s
        """,
        (list(platforms), model, limit),
    ).fetchall()
    out = {"confirmed": 0, "refuted": 0, "unprobeable": 0, "gray": 0, "no_a_pages": 0}
    for artist_id, identity_id, emb_text, a_pages in rows:
        if not a_pages:
            _mark(conn, identity_id, {"status": "no_a_pages"})
            out["no_a_pages"] += 1
            if commit_each:
                conn.commit()
            continue
        centroid = np.asarray(json.loads(emb_text), dtype=np.float32)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
        best = None
        with tempfile.TemporaryDirectory(prefix="corroborate-") as tmp:
            for page in a_pages:
                cos, _no_audio = candidate_cosine(
                    conn, centroid, page["platform"], str(page["pid"]),
                    embedder, Path(tmp), fetch)
                if cos is not None and (best is None or cos > best[0]):
                    best = (cos, page["platform"])
        if best is None:
            _mark(conn, identity_id, {"status": "unprobeable"})
            out["unprobeable"] += 1
        elif best[0] >= confirm:
            # MB-declared audio agrees with the centroid: the machine guess
            # is corroborated — tier C, original method kept in evidence
            conn.execute(
                """UPDATE platform_identity SET binding_tier = 'C',
                   binding_evidence = COALESCE(binding_evidence, '{}'::jsonb)
                     || jsonb_build_object('corroboration', %s::jsonb)
                   WHERE id = %s""",
                (json.dumps({"status": "confirmed", "cosine": round(best[0], 4),
                             "vs_platform": best[1]}), identity_id))
            out["confirmed"] += 1
        elif best[0] < reject:
            _mark(conn, identity_id,
                  {"status": "refuted", "cosine": round(best[0], 4), "vs_platform": best[1]})
            dup = conn.execute(
                "SELECT 1 FROM review_item WHERE kind='source_binding' AND subject_id=%s "
                "AND reason='source_coherence' AND status='pending'", (artist_id,)).fetchone()
            if not dup:
                conn.execute(
                    """INSERT INTO review_item (kind, subject_type, subject_id, reason, evidence, status)
                       VALUES ('source_binding','artist',%s,'source_coherence',%s,'pending')""",
                    (artist_id, json.dumps({
                        "platform": best[1],
                        "query": f"embed source disagrees with MB-declared {best[1]} audio (cos {round(best[0], 4)})",
                        "coherence": {"min_cosine": round(best[0], 4),
                                      "worst_pair": ["centroid", best[1]]},
                    })))
            out["refuted"] += 1
        else:
            _mark(conn, identity_id,
                  {"status": "gray", "cosine": round(best[0], 4), "vs_platform": best[1]})
            out["gray"] += 1
        if commit_each:
            conn.commit()
    out["processed"] = len(rows)
    return out


def _mark(conn: Connection, identity_id, corroboration: dict) -> None:
    conn.execute(
        """UPDATE platform_identity
           SET binding_evidence = COALESCE(binding_evidence, '{}'::jsonb)
             || jsonb_build_object('corroboration', %s::jsonb)
           WHERE id = %s""",
        (json.dumps(corroboration), identity_id))


def main() -> None:
    import argparse
    import os

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="acoustically corroborate single-source B-tier bindings")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--platforms", default=",".join(PROBEABLE))
    ap.add_argument("--retry-status", default=None,
                    help="clear corroboration markers with this status and re-attempt (e.g. no_a_pages)")
    args = ap.parse_args()
    os.environ.setdefault("PIPELINE_FP16", "0")  # the 30s-clip NaN lesson
    from pipeline.embedders.registry import get_embedder

    embedder = get_embedder()
    with psycopg.connect(Settings().database_url) as conn:
        print(corroborate_blind_spot(conn, embedder=embedder, limit=args.limit,
                                     platforms=tuple(args.platforms.split(",")),
                                     retry_status=args.retry_status,
                                     commit_each=True))


if __name__ == "__main__":
    main()
