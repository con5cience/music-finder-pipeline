"""Publish: factory → serving DB (the app's artists table). Local app DB now,
cloud sync later (locked decision).

Publishes EMBEDDED artists only (embedding_source stamped — the cascade's
quality gate). Upsert by mbid; idempotent re-publish refreshes everything we
own. We write ONLY pipeline-owned fields: identity (name/slug/mbid), platform
URLs derived from identities, tags (calibrated z-scores: top K with z > 0,
weight = round(z)+1 so the app's integer-weight convention holds), and the
MuQ centroid (app migration 047 made the column untyped). App-owned fields
(top-tracks caches, image_url, user-facing state) are never touched.

Run:  uv run poe publish -- --limit 1000
Env:  APP_DATABASE_URL (no default — publishing is deliberate)
"""

from __future__ import annotations

import os
import re
import unicodedata

from psycopg import Connection

from pipeline.tag_calibration import ARTIST_SUFFIX

TAG_K = 10
# ADR-020 P2: hard-drop a tag assigned to more than this share of the corpus.
# Default 1.0 = OFF, because a pure rate ceiling can't separate spurious magnets
# (kilapanga 0.22) from legitimately-broad genres (rock 0.16, edm 0.17) — they
# overlap — so we rely on the soft idf (ln(N/df)) ranking instead. Lower this
# only to force-drop a specific pathological tag.
MAGNET_RATE_CEILING = 1.0
# ADR-020 P3: keep only tags whose idf-adjusted z is at least this fraction of the
# artist's OWN best — stops force-padding a thin-signal artist to TAG_K slots with
# demoted magnets (real-data sim: avg magnets/artist 1.35 -> 0.56, ~4 tags kept).
TAG_REL_GATE = 0.5
# ADR-020 P5: centering strength. When tag_centering data exists, publish ranks by
# `score - CENTERING_C * d_i` (d_i = tag alignment with the dominant audio
# direction) instead of z*idf — demotes the anisotropy-aligned scattered/magnet
# tags that the z-score (which divides by per-tag spread) actually inflated.
# 0.585 = the corpus-mean projection; validated to recover ~2/3 of full re-embed
# centering from stored scores alone. Empty tag_centering -> legacy z*idf path.
CENTERING_C = float(os.environ.get("PIPELINE_CENTERING_C", "0.585"))
# Audio tier shape: aim for the curated target of 1-4 intrinsic tags (like MB's),
# not a padded list. Magnet-pruned + centered, keep only the top few within a
# TIGHT margin of the best; min 1 so an audio artist is never empty.
AUDIO_TAG_K = int(os.environ.get("PIPELINE_AUDIO_TAG_K", "4"))
AUDIO_REL_GATE = float(os.environ.get("PIPELINE_AUDIO_GATE", "0.85"))

# Audio tier method (ADR-025): borrow genres from an artist's nearest-SOUNDING
# MB-labeled neighbors instead of trusting raw zero-shot text-similarity tags
# (which hallucinate niche genres). Validated ~3.4x F1 / ~4x precision over the
# centered zero-shot tier, with no hallucinations (anchors carry only real MB
# genres). EMB_MODEL = the audio-embedding model (factory partial HNSW
# idx_artist_embedding_muq_ann). KNN_GATE = relative keep-margin vs the top vote.
EMB_MODEL = os.environ.get("PIPELINE_EMBED_MODEL", "muq-large-msd")
AUDIO_KNN_K = int(os.environ.get("PIPELINE_AUDIO_KNN_K", "25"))       # neighbors voted
AUDIO_KNN_FETCH = int(os.environ.get("PIPELINE_AUDIO_KNN_FETCH", "80"))  # ANN over-fetch (then filter to anchors)
AUDIO_KNN_GATE = float(os.environ.get("PIPELINE_AUDIO_KNN_GATE", "0.7"))

_URL_BUILDERS = {
    "deezer": lambda pid: f"https://www.deezer.com/artist/{pid}",
    "bandcamp": lambda pid: f"https://{pid}.bandcamp.com",
    "soundcloud": lambda pid: f"https://soundcloud.com/{pid}",
    "youtube": lambda pid: f"https://www.youtube.com/channel/{pid}",
    "tidal": lambda pid: f"https://tidal.com/browse/artist/{pid}",
}
_URL_COLUMNS = {p: f"{p}_url" for p in _URL_BUILDERS}


def slug_base(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", folded.casefold()).strip("-") or "artist"


def resolve_slug(app: Connection, name: str, key: str) -> str:
    """The app's own slug convention (its migration 031 indexes exactly this):
    clean base for the first holder, base-2/-3/... on homonym collisions.
    Idempotent: a row that already holds a slug in the family keeps it."""
    base = slug_base(name)
    mine = app.execute(
        "SELECT slug FROM artists WHERE (mbid = %s OR id::text = %s) "
        "AND (slug = %s OR slug ~ ('^' || %s || '-[0-9]+$'))",
        (key, key, base, base),
    ).fetchone()
    if mine:
        return mine[0]  # stable across re-publishes
    taken = {
        r[0] for r in app.execute(
            "SELECT slug FROM artists WHERE (slug = %s OR slug ~ ('^' || %s || '-[0-9]+$')) "
            "AND coalesce(mbid, '') != %s AND id::text != %s",
            (base, base, key, key),
        ).fetchall()
    }
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def publishable_artists(conn: Connection, limit: int, since=None, after_id=None) -> list[tuple]:
    """All embedded artists, or — incremental mode — only those whose
    embedding or artist tags changed since the watermark. Banned artists
    never publish (ban_ledger, the do-not-rediscover law)."""
    since_sql = """
          AND (ae.computed_at >= %(since)s
               OR EXISTS (SELECT 1 FROM artist_tag_scores ats
                          WHERE ats.artist_id = a.id AND ats.computed_at >= %(since)s))
    """ if since is not None else ""
    after_sql = "AND a.id > %(after_id)s" if after_id is not None else ""
    return conn.execute(
        f"""
        SELECT a.id, a.mbid::text, a.display_name, a.embedding_source,
               ae.embedding::text, ae.model, ae.signal_ratio
        FROM artist a
        JOIN artist_embedding ae ON ae.artist_id = a.id
        WHERE a.embedding_source IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM ban_ledger b WHERE b.artist_id = a.id
                          OR (a.mbid IS NOT NULL AND b.mbid = a.mbid))
          -- integrity freezer (2026-06-12): open acoustic-disagreement or
          -- AI-slop flags hold the artist from serving until a human looks
          AND NOT EXISTS (SELECT 1 FROM review_item ri WHERE ri.subject_id = a.id
                          AND ri.reason IN ('source_coherence', 'ai_slop')
                          AND ri.status = 'pending')
          {since_sql}
          {after_sql}
        ORDER BY a.id LIMIT %(limit)s
        """,
        {"limit": limit, "since": since, "after_id": after_id},
    ).fetchall()


def publish_incremental(factory: Connection, app: Connection, limit: int = 100000) -> int:
    """Hourly-sync mode: advance the watermark FIRST (no lost-update window
    — anything changing mid-run lands next hour), publish changed artists,
    and prune app rows for newly banned ones."""
    wm = factory.execute("SELECT last_run FROM publish_watermark WHERE id = 'default'").fetchone()[0]
    # clock_timestamp: wall clock, not txn-frozen now()
    factory.execute("UPDATE publish_watermark SET last_run = clock_timestamp() WHERE id = 'default'")
    # drain-until-empty via KEYSET pagination (second review catch: without
    # advancing after_id, the same first window re-published forever)
    n = 0
    after = None
    anchors = load_anchor_genres(factory)  # kNN audio-tier anchors, built once
    while True:
        rows = publishable_artists(factory, limit, since=wm, after_id=after)
        if not rows:
            break
        # continuous slop gate: evaluate first-time/grown catalogs in the
        # same cycle; freshly flagged artists drop out before publishing
        from pipeline.slop_detect import gate_unevaluated

        gated = gate_unevaluated(factory, [r[0] for r in rows])
        # keyset advance comes from the PRE-filter batch: an all-flagged
        # batch leaves rows empty, and rows[-1] would crash the hourly sync
        # (caught by the end-to-end choke test before it ever ran live)
        after = rows[-1][0]
        full_batch = len(rows) == limit
        if gated["flagged"]:
            flagged_now = {r[0] for r in factory.execute(
                """SELECT subject_id FROM review_item WHERE reason='ai_slop'
                   AND status='pending' AND subject_id = ANY(%s)""",
                ([str(r[0]) for r in rows],)).fetchall()}
            rows = [r for r in rows if r[0] not in flagged_now]
        publish_rows(factory, app, rows, anchors=anchors)
        n += len(rows)
        if not full_batch:
            break
    for (aid, mbid) in factory.execute(
        "SELECT artist_id, mbid::text FROM ban_ledger WHERE banned_at >= %s", (wm,)
    ).fetchall():
        app.execute("DELETE FROM artists WHERE id = %s OR (mbid IS NOT NULL AND mbid = %s)",
                    (str(aid) if aid else "00000000-0000-0000-0000-000000000000", mbid))
    prune_lost_embeddings(factory, app)
    return n


def republish_all(
    factory: Connection,
    app: Connection,
    batch: int = 2000,
    commit_each: bool = True,
    start_after=None,
    progress=None,
) -> int:
    """Full re-publish: re-derive EVERY serving field for every embedded artist,
    in COMMITTED keyset batches. Use when a change the incremental watermark does
    NOT trigger must reach serving — e.g. a tag-calibration change marks no artist
    'changed', so publish_incremental skips them all.

    Why not just reset the watermark and let publish_incremental run? Because that
    re-derives 125k rows in ONE transaction (observed: a 2h+ open txn, no progress
    visibility, everything lost on interrupt). This commits per batch, so progress
    is durable and visible. Idempotent (publish_rows upserts), so a crash loses
    only the in-flight batch — resume with start_after = the last committed id.

    Does NOT advance the watermark: the caller sets it to the run's start time so
    the next hourly incremental picks up anything embedded DURING the run."""
    total = 0
    after = start_after
    anchors = load_anchor_genres(factory)  # kNN audio-tier anchors, built once
    while True:
        rows = publishable_artists(factory, batch, since=None, after_id=after)
        if not rows:
            break
        after = rows[-1][0]
        publish_rows(factory, app, rows, anchors=anchors)
        if commit_each:
            app.commit()
            factory.commit()  # also release the read snapshot — don't hold it all run
        total += len(rows)
        if progress:
            progress(total, after)
        if len(rows) < batch:
            break
    return total


def prune_lost_embeddings(factory: Connection, app: Connection, batch: int = 10000) -> int:
    """Remove serving rows whose factory artist no longer has an embedding.

    The poisoned-well gap (2026-06-12): publish only upserts embedded artists
    and deletes banned ones — a RESET embedding (binding remediation, NaN
    purges) silently stranded its stale row + vector on the serving side
    forever. Conservative by construction: only rows whose artist factory
    POSITIVELY KNOWS (matched by mbid or id) and which have zero embeddings
    are deleted; anything factory can't match is left alone."""
    pruned = 0
    last = None
    while True:
        rows = app.execute(
            "SELECT id::text, mbid FROM artists"
            + (" WHERE id > %s" if last is not None else "")
            + " ORDER BY id LIMIT %s",
            ((last, batch) if last is not None else (batch,)),
        ).fetchall()
        if not rows:
            break
        ids = [r[0] for r in rows]
        mbids = [r[1] for r in rows]
        # Index-friendly anti-join. The old single query joined on
        # (a.mbid = x.mbid::uuid OR a.id::text = x.app_id): the OR across two
        # columns plus the a.id::text cast defeated every index and seq-scanned
        # the whole artist table per batch — ~30 min at 100k+ published, which
        # held the publish transaction's watermark lock long enough that hourly
        # runs overlapped and never committed (the 2026-06-15 sync wedge). Split
        # into two index-driven branches (artist_pkey on id, idx_artist_mbid on
        # mbid) UNIONed by app_id — identical semantics, no seq scan.
        victims = [r[0] for r in factory.execute(
            """
            SELECT x.app_id::text
            FROM unnest(%(ids)s::uuid[]) AS x(app_id)
            JOIN artist a ON a.id = x.app_id
            WHERE NOT EXISTS (SELECT 1 FROM artist_embedding e WHERE e.artist_id = a.id)
            UNION
            SELECT x.app_id::text
            FROM unnest(%(ids)s::uuid[], %(mbids)s::uuid[]) AS x(app_id, mbid)
            JOIN artist a ON a.mbid = x.mbid
            WHERE x.mbid IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM artist_embedding e WHERE e.artist_id = a.id)
            """,
            {"ids": ids, "mbids": mbids},
        ).fetchall()]
        if victims:
            pruned += app.execute(
                "DELETE FROM artists WHERE id = ANY(%s::uuid[])", (victims,)
            ).rowcount
        last = ids[-1]
        if len(rows) < batch:
            break
    return pruned


def artist_language(conn: Connection, artist_id) -> str | None:
    """Majority ASR language over the artist's tracks (wave-3, sparse
    coverage by design — None until the artist has ≥2 agreeing tracks)."""
    row = conn.execute(
        """
        SELECT language, count(*) FROM track_language tl
        JOIN audio_track t ON t.id = tl.track_id
        WHERE t.artist_id = %s AND tl.confidence >= 0.6
        GROUP BY language ORDER BY 2 DESC LIMIT 1
        """,
        (artist_id,),
    ).fetchone()
    return row[0] if row and row[1] >= 2 else None


def artist_location(conn: Connection, artist_id) -> str | None:
    """Discovered artists carry their Bandcamp profile location (the MB
    area hint) — published so the product can show WHERE the underground is."""
    row = conn.execute(
        "SELECT location FROM bc_candidate WHERE artist_id = %s AND location IS NOT NULL",
        (artist_id,),
    ).fetchone()
    return row[0] if row else None


def artist_perceptual(conn: Connection, artist_id) -> dict | None:
    """Wave-2 axis means over the artist's tracks (the scorer's wₚ input)."""
    row = conn.execute(
        """
        SELECT avg(danceability), avg(valence), avg(arousal),
               avg(speechiness), avg(liveness), avg(vocalness)
        FROM track_perceptual tp JOIN audio_track t ON t.id = tp.track_id
        WHERE t.artist_id = %s
        """,
        (artist_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    keys = ("danceability", "valence", "arousal", "speechiness", "liveness", "vocalness")
    return {k: round(float(v), 4) for k, v in zip(keys, row, strict=True)}


def artist_urls(conn: Connection, artist_id) -> dict[str, str]:
    out: dict[str, str] = {}
    for platform, pid in conn.execute(
        "SELECT platform, platform_id FROM platform_identity WHERE artist_id = %s", (artist_id,)
    ).fetchall():
        builder = _URL_BUILDERS.get(platform)
        if builder and platform not in out:
            out[_URL_COLUMNS[platform]] = builder(pid)
    return out


def artist_tags(conn: Connection, artist_id, g_moments=None) -> dict[str, int]:
    """Calibrated artist-level tags. PRIMARY: artist_tag_scores (scored from
    the artist-mean MuLan vector at embed time — full resolution, no
    per-track-truncation pathology), z-ranked against per-tag corpus moments.
    FALLBACK until the v2 sweep covers an artist: the per-track aggregation
    (coverage-weighted, known-noisy on preview sources)."""
    if g_moments is None:
        g_moments = conn.execute(
            "SELECT avg(score), greatest(stddev(score), 1e-6), count(DISTINCT artist_id) FROM artist_tag_scores"
        ).fetchone()
    gmean, gsd = (g_moments[0] or 0.0), (g_moments[1] or 1e-6)
    n_corpus = (
        float(g_moments[2] or 1)
        if len(g_moments) > 2
        else float(conn.execute("SELECT count(DISTINCT artist_id) FROM artist_tag_scores").fetchone()[0] or 1)
    )
    primary = conn.execute(
        """
        SELECT tag, z, z * idf AS z_adj FROM (
            SELECT ats.tag,
                   (ats.score - coalesce(tc.mean, %s)) / coalesce(tc.stddev, %s) AS z,
                   -- idf = ln(N/df): df is the ARTIST-source n (one row per artist
                   -- per tag). An over-assigned magnet gets a small idf, a rare tag
                   -- a large one, so z*idf demotes magnets while a genuinely-broad
                   -- genre survives on its high z (ADR-020 P2).
                   ln(%s / greatest(coalesce(tc.n, 1), 1)::float) AS idf,
                   coalesce(tc.n, 0)::float / %s AS rate
            FROM artist_tag_scores ats
            -- ADR-020 P1: ARTIST-source moments (model||'#artist'), matching the
            -- distribution being scored.
            LEFT JOIN tag_calibration tc ON tc.tag = ats.tag AND tc.model = ats.model || %s
            WHERE ats.artist_id = %s AND ats.score != 'NaN'::real  -- NaN armor (pg NaN-equality law)
        ) s
        WHERE rate < %s  -- hard ceiling (default OFF at 1.0)
        ORDER BY z_adj DESC LIMIT %s
        """,
        (gmean, gsd, n_corpus, n_corpus, ARTIST_SUFFIX, artist_id, MAGNET_RATE_CEILING, TAG_K),
    ).fetchall()
    if primary:
        # Order/keep by z_adj (idf-demoted). P3 relative gate: keep only tags
        # within TAG_REL_GATE of the artist's OWN best z_adj, so a thin-signal
        # artist publishes its few real tags rather than TAG_K padded with demoted
        # magnets. Weight from the plain z so tag_vector magnitudes stay normal.
        top = max(float(za) for _t, _z, za in primary)
        floor = TAG_REL_GATE * top
        return {
            tag: max(1, round(float(z)) + 1)
            for tag, z, z_adj in primary
            if float(z_adj) > 0 and float(z_adj) >= floor
        }
    rows = conn.execute(
        """
        WITH g AS (SELECT avg(score) gmean, greatest(stddev(score), 1e-6) gsd
                   FROM track_tag_scores),
        z AS (
            SELECT tts.tag,
                   avg((tts.score - coalesce(tc.mean, g.gmean))
                       / coalesce(tc.stddev, g.gsd)) AS mz,
                   count(*) AS cnt
            FROM track_tag_scores tts
            JOIN audio_track t ON t.id = tts.track_id
            CROSS JOIN g
            LEFT JOIN tag_calibration tc ON tc.tag = tts.tag AND tc.model = tts.model
            WHERE t.artist_id = %s
              AND tts.score != 'NaN'::real  -- pg law: NaN = NaN is TRUE; x=x can't detect it
            GROUP BY tts.tag
        )
        SELECT tag, mz, mz * sqrt(cnt) AS ranked
        FROM z WHERE mz > 0 ORDER BY ranked DESC LIMIT %s
        """,
        (artist_id, TAG_K),
    ).fetchall()
    return {tag: max(1, round(float(z)) + 1) for tag, z, _r in rows}


# --- batch (set-based) reads: one query per batch instead of per artist. The
# per-artist helpers above stay (single-artist callers + the rare tag fallback);
# publish_rows uses these so a full re-publish is ~one query each, not 5*N. All
# key on artist_id::text so lookups match regardless of psycopg uuid adaptation.


def artist_urls_batch(conn: Connection, aids: list) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for aid, platform, pid in conn.execute(
        "SELECT artist_id::text, platform, platform_id FROM platform_identity "
        "WHERE artist_id = ANY(%s::uuid[])",
        ([str(a) for a in aids],),
    ).fetchall():
        builder = _URL_BUILDERS.get(platform)
        if builder:
            out.setdefault(aid, {})[_URL_COLUMNS[platform]] = builder(pid)
    return out


def artist_perceptual_batch(conn: Connection, aids: list) -> dict[str, dict]:
    keys = ("danceability", "valence", "arousal", "speechiness", "liveness", "vocalness")
    out: dict[str, dict] = {}
    for row in conn.execute(
        """
        SELECT t.artist_id::text, avg(danceability), avg(valence), avg(arousal),
               avg(speechiness), avg(liveness), avg(vocalness)
        FROM track_perceptual tp JOIN audio_track t ON t.id = tp.track_id
        WHERE t.artist_id = ANY(%s::uuid[]) GROUP BY t.artist_id
        """,
        ([str(a) for a in aids],),
    ).fetchall():
        if row[1] is not None:
            out[row[0]] = {k: round(float(v), 4) for k, v in zip(keys, row[1:], strict=True)}
    return out


def artist_language_batch(conn: Connection, aids: list) -> dict[str, str]:
    out: dict[str, str] = {}
    for aid, lang, _cnt in conn.execute(
        """
        SELECT aid, language, cnt FROM (
            SELECT t.artist_id::text AS aid, tl.language, count(*) AS cnt,
                   row_number() OVER (PARTITION BY t.artist_id ORDER BY count(*) DESC) AS rn
            FROM track_language tl JOIN audio_track t ON t.id = tl.track_id
            WHERE t.artist_id = ANY(%s::uuid[]) AND tl.confidence >= 0.6
            GROUP BY t.artist_id, tl.language
        ) s WHERE rn = 1 AND cnt >= 2
        """,
        ([str(a) for a in aids],),
    ).fetchall():
        out[aid] = lang
    return out


def artist_location_batch(conn: Connection, aids: list) -> dict[str, str]:
    out: dict[str, str] = {}
    for aid, loc in conn.execute(
        "SELECT DISTINCT ON (artist_id) artist_id::text, location FROM bc_candidate "
        "WHERE artist_id = ANY(%s::uuid[]) AND location IS NOT NULL ORDER BY artist_id",
        ([str(a) for a in aids],),
    ).fetchall():
        out[aid] = loc
    return out


def artist_tags_fallback_batch(conn: Connection, aids: list) -> dict[str, dict]:
    """Batched twin of artist_tags' FALLBACK (track-aggregation) path, for the
    artists with no non-NaN artist_tag_scores. Per-artist this query is heavy
    (track_tag_scores JOIN audio_track), so looping it for the NaN-poisoned old
    artists was the real cost (a 2000-id sample: 141 fell back = ~70s). One
    query, window-ranked top-K per artist — same global track moments, same
    mz>0 / mz*sqrt(cnt) ranking as the per-artist code."""
    if not aids:
        return {}
    out: dict[str, dict] = {}
    for aid, tag, mz in conn.execute(
        """
        WITH g AS (
            SELECT avg(score) AS gmean, greatest(stddev(score), 1e-6) AS gsd FROM track_tag_scores
        ),
        z AS (
            SELECT t.artist_id::text AS aid, tts.tag,
                   avg((tts.score - coalesce(tc.mean, g.gmean)) / coalesce(tc.stddev, g.gsd)) AS mz,
                   count(*) AS cnt
            FROM track_tag_scores tts
            JOIN audio_track t ON t.id = tts.track_id
            CROSS JOIN g
            LEFT JOIN tag_calibration tc ON tc.tag = tts.tag AND tc.model = tts.model
            LEFT JOIN tag_audio_blocklist bl ON bl.tag = tts.tag
            WHERE t.artist_id = ANY(%(aids)s::uuid[]) AND tts.score != 'NaN'::real
              AND bl.tag IS NULL   -- magnet prune (ADR-020 P4), same as the primary tier
            GROUP BY t.artist_id, tts.tag
        ),
        ranked AS (
            SELECT aid, tag, mz, row_number() OVER (PARTITION BY aid ORDER BY mz * sqrt(cnt) DESC) AS rn
            FROM z WHERE mz > 0
        )
        SELECT aid, tag, mz FROM ranked WHERE rn <= %(k)s
        """,
        {"aids": [str(a) for a in aids], "k": AUDIO_TAG_K},
    ).fetchall():
        out.setdefault(aid, {})[tag] = max(1, round(float(mz)) + 1)
    return out


def artist_tags_batch(conn: Connection, aids: list, g_moments) -> dict[str, dict]:
    """THE artist-tag ranking (one entry point). Uses centering — rank by
    `score - C*d_i`, ADR-020 Phase 5 — once tag_centering is populated; until then
    the legacy z*idf path (ADR-020 P1-P3). Same shared track-aggregation fallback
    either way. publish_rows calls only this."""
    if conn.execute("SELECT EXISTS (SELECT 1 FROM tag_centering)").fetchone()[0]:
        return _artist_tags_centered(conn, aids, g_moments)
    return _artist_tags_zidf(conn, aids, g_moments)


def _artist_tags_zidf(conn: Connection, aids: list, g_moments) -> dict[str, dict]:
    """Legacy ranking (ADR-020 P1-P3): artist-moment z, idf demote, top-K,
    relative gate. The fallback before centering data exists / if reverted."""
    gmean, gsd = (g_moments[0] or 0.0), (g_moments[1] or 1e-6)
    n_corpus = float(g_moments[2] or 1)
    ids = [str(a) for a in aids]
    out: dict[str, dict] = {}
    for aid, tag, z in conn.execute(
        """
        WITH s AS (
            SELECT ats.artist_id::text AS aid, ats.tag,
                   (ats.score - coalesce(tc.mean, %(gm)s)) / coalesce(tc.stddev, %(gs)s) AS z,
                   ln(%(n)s / greatest(coalesce(tc.n, 1), 1)::float) AS idf,
                   coalesce(tc.n, 0)::float / %(n)s AS rate
            FROM artist_tag_scores ats
            LEFT JOIN tag_calibration tc ON tc.tag = ats.tag AND tc.model = ats.model || %(suf)s
            WHERE ats.artist_id = ANY(%(aids)s::uuid[]) AND ats.score != 'NaN'::real
        ),
        r AS (SELECT aid, tag, z, z * idf AS z_adj FROM s WHERE rate < %(ceil)s),
        ranked AS (
            SELECT aid, tag, z, z_adj,
                   row_number() OVER (PARTITION BY aid ORDER BY z_adj DESC) AS rn,
                   max(z_adj) OVER (PARTITION BY aid) AS top_za
            FROM r
        )
        SELECT aid, tag, z FROM ranked
        WHERE rn <= %(k)s AND z_adj > 0 AND z_adj >= %(gate)s * top_za
        """,
        {"gm": gmean, "gs": gsd, "n": n_corpus, "suf": ARTIST_SUFFIX, "aids": ids,
         "ceil": MAGNET_RATE_CEILING, "k": TAG_K, "gate": TAG_REL_GATE},
    ).fetchall():
        out.setdefault(aid, {})[tag] = max(1, round(float(z)) + 1)
    # fallback for artists with NO non-NaN artist_tag_scores (matches the
    # per-artist code: primary empty -> track aggregation). NaN-poisoned old
    # artists make this non-trivial, so it is BATCHED too (looping it per artist
    # was the whole cost).
    covered = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT artist_id::text FROM artist_tag_scores "
            "WHERE artist_id = ANY(%s::uuid[]) AND score != 'NaN'::real",
            (ids,),
        ).fetchall()
    }
    uncovered = [a for a in aids if str(a) not in covered]
    if uncovered:
        out.update(artist_tags_fallback_batch(conn, uncovered))
    return out


def _artist_tags_centered(conn: Connection, aids: list, g_moments, c: float | None = None) -> dict[str, dict]:
    """The AUDIO tag tier (lowest priority; MB + Bandcamp win first). Rank by
    `score - C*d_i` (centering, demotes the anisotropy direction), EXCLUDE the
    data-driven magnet blocklist (tag_audio_blocklist — orthodox pop, kilapanga,
    j-rock, …), and keep only the top 1-4 within a TIGHT margin of the best — the
    curated target of a few intrinsic tags, never a padded list. min 1 (rn=1
    always) so an audio artist is never empty. Weight from the calibrated z."""
    if c is None:
        c = CENTERING_C
    gmean, gsd = (g_moments[0] or 0.0), (g_moments[1] or 1e-6)
    ids = [str(a) for a in aids]
    out: dict[str, dict] = {}
    for aid, tag, z in conn.execute(
        """
        WITH s AS (
            SELECT ats.artist_id::text AS aid, ats.tag,
                   (ats.score - coalesce(tc.mean, %(gm)s)) / coalesce(tc.stddev, %(gs)s) AS z,
                   ats.score - %(c)s * coalesce(cn.d, 0.0) AS centered
            FROM artist_tag_scores ats
            LEFT JOIN tag_centering cn ON cn.tag = ats.tag AND cn.model = ats.model
            LEFT JOIN tag_calibration tc ON tc.tag = ats.tag AND tc.model = ats.model || %(suf)s
            LEFT JOIN tag_audio_blocklist bl ON bl.tag = ats.tag
            WHERE ats.artist_id = ANY(%(aids)s::uuid[]) AND ats.score != 'NaN'::real
              AND bl.tag IS NULL   -- magnet prune (ADR-020 P4)
        ),
        ranked AS (
            SELECT aid, tag, z, centered,
                   row_number() OVER (PARTITION BY aid ORDER BY centered DESC) AS rn,
                   max(centered) OVER (PARTITION BY aid) AS top_c
            FROM s
        )
        SELECT aid, tag, z FROM ranked
        -- top 1-4 within a tight margin; rn=1 always kept (never empty)
        WHERE rn <= %(k)s AND (rn = 1 OR (centered > 0 AND centered >= %(gate)s * top_c))
        """,
        {"gm": gmean, "gs": gsd, "c": c, "suf": ARTIST_SUFFIX, "aids": ids,
         "k": AUDIO_TAG_K, "gate": AUDIO_REL_GATE},
    ).fetchall():
        out.setdefault(aid, {})[tag] = max(1, round(float(z)) + 1)
    covered = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT artist_id::text FROM artist_tag_scores "
            "WHERE artist_id = ANY(%s::uuid[]) AND score != 'NaN'::real", (ids,)
        ).fetchall()
    }
    uncovered = [a for a in aids if str(a) not in covered]
    if uncovered:
        out.update(artist_tags_fallback_batch(conn, uncovered))
    return out


def mb_genres_batch(conn: Connection, aids: list) -> dict[str, dict]:
    """MusicBrainz EDITORIAL genres per artist — accurate by construction (the
    primary tag source; audio tags are only the fallback where MB has none).
    MB tags are filtered to the canonical genre vocab (drops non-genres like
    'canadian'/'seen live') and alias-merged ('synth-pop'->'synthpop'), with
    editorial vote counts summed across spellings; capped at TAG_K by count.
    Empty for artists MB has no genres for (~78%, the underground)."""
    out: dict[str, dict] = {}
    for aid, genre, cnt in conn.execute(
        """
        SELECT a.id::text AS aid, coalesce(gc.name, gd.name) AS genre, at.count AS cnt
        FROM artist a
        JOIN mb_raw.artist mra ON mra.gid::text = a.mbid::text
        JOIN mb_raw.artist_tag at ON at.artist = mra.id AND at.count > 0
        JOIN mb_raw.tag t ON t.id = at.tag
        LEFT JOIN mb_raw.genre gd ON gd.name = lower(t.name)
        LEFT JOIN mb_raw.genre_alias gal ON gal.name = lower(t.name)
        LEFT JOIN mb_raw.genre gc ON gc.id = gal.genre
        WHERE a.id = ANY(%s::uuid[]) AND a.mbid IS NOT NULL
          AND (gd.name IS NOT NULL OR gc.name IS NOT NULL)
        """,
        ([str(a) for a in aids],),
    ).fetchall():
        d = out.setdefault(aid, {})
        d[genre] = d.get(genre, 0) + int(cnt)  # sum votes across spellings
    capped: dict[str, dict] = {}
    for aid, gd in out.items():
        top = sorted(gd.items(), key=lambda kv: -kv[1])[:TAG_K]
        capped[aid] = {g: max(1, c) for g, c in top}
    return capped


def bandcamp_tags_batch(conn: Connection, aids: list) -> dict[str, dict]:
    """HUMAN Bandcamp tags per artist (bc_candidate.tags) — the middle tier: what
    the artist self-describes on the site. Used where MB has no editorial genres;
    later clobbered by MB once the artist is approved + reingested. Uniform weight
    (human tags carry no score), capped at TAG_K — the HUMAN-tier cap (same as MB),
    NOT the tight AUDIO_TAG_K: these are trustworthy human tags, so we keep the
    full set rather than truncate the best one alphabetically (the serving-side
    IDF down-weights generic tags like 'electronic' anyway, #30). Empty for
    artists with no Bandcamp tags (so the audio tier takes over)."""
    out: dict[str, dict] = {}
    for aid, tag in conn.execute(
        """
        SELECT DISTINCT bc.artist_id::text AS aid, lower(bt) AS tag
        FROM bc_candidate bc, unnest(bc.tags) bt
        WHERE bc.artist_id = ANY(%s::uuid[]) AND bt IS NOT NULL AND bt <> ''
        ORDER BY aid, tag
        """,
        ([str(a) for a in aids],),
    ).fetchall():
        d = out.setdefault(aid, {})
        if len(d) < TAG_K:
            d[tag] = 1
    return out


_TOKEN_SEP = re.compile(r"[\-_/]+")  # treat - _ / as word breaks for tokenization


def recover_genre_tokens(tag: str, approved: frozenset) -> list[str]:
    """Greedy longest-match: pull the APPROVED genre n-grams contained in a
    compound Bandcamp tag, so a band that self-tags only 'progressive doom' (a
    rare, unapproved compound) still contributes the approved genre 'doom'.
    Separators (- _ /) count as word breaks. Longest match wins so 'atmospheric
    black metal' yields 'black metal', not 'metal'. Returns [] when no approved
    genre word is present (e.g. 'voodoo') — pure junk recovers nothing."""
    words = _TOKEN_SEP.sub(" ", tag).split()
    out: list[str] = []
    n = len(words)
    i = 0
    while i < n:
        hit_end = 0
        for j in range(n, i, -1):  # try the longest span starting at i first
            if " ".join(words[i:j]) in approved:
                out.append(" ".join(words[i:j]))
                hit_end = j
                break
        i = hit_end if hit_end else i + 1
    return out


def allowlist_bc_tags(tags: dict, approved: frozenset, blocked: frozenset) -> dict:
    """Genre-only gate for the Bandcamp folksonomy tier. Keep APPROVED tags whole;
    for an UNDECIDED tag recover any approved genre tokens it contains (compound
    tokenization); drop BLOCKED tags and never tokenize them (an explicit block
    wins over recovery). Overlapping recoveries keep the max weight. Pure → unit
    tested; an empty result means the artist has no approved Bandcamp signal and
    should fall through to the audio tier."""
    out: dict[str, int] = {}
    for tag, w in tags.items():
        if tag in blocked:
            continue
        if tag in approved:
            out[tag] = max(out.get(tag, 0), w)
            continue
        for tok in recover_genre_tokens(tag, approved):
            out[tok] = max(out.get(tok, 0), w)
    return out


def merge_human_tiers(mb: dict, bc: dict) -> dict:
    """UNION the MB-editorial + Bandcamp-human tag tiers (both are human curation).
    Was a cascade (mb OR bc) that dropped an artist's rich Bandcamp folksonomy
    whenever MB had even one editorial genre — e.g. autumn-us showed only
    'gothic rock' while its Bandcamp page lists goth/post-punk/shoegaze/…. A tag
    present in BOTH tiers is corroborated, so weights ADD (it leads); BC-only tags
    keep their uniform weight 1. Capped at TAG_K by weight (alphabetical
    tie-break for determinism). The manual blocklist + serving-side IDF still
    apply downstream."""
    merged = dict(mb)
    for tag, w in bc.items():
        merged[tag] = merged.get(tag, 0) + w
    top = sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))[:TAG_K]
    return dict(top)


def load_anchor_genres(conn: Connection) -> dict[str, frozenset]:
    """artist_id(str) -> frozenset(MB editorial genres) for EVERY MB-covered
    artist — the kNN label-propagation anchors (ADR-025). Same vocab/alias logic
    as mb_genres_batch. Built once per publish run and passed into publish_rows
    (NOT module-cached, so tests on a rolled-back DB stay isolated)."""
    out: dict[str, set] = {}
    for aid, genre in conn.execute(
        """
        SELECT a.id::text, coalesce(gc.name, gd.name) AS genre
        FROM artist a
        JOIN mb_raw.artist mra ON mra.gid::text = a.mbid::text
        JOIN mb_raw.artist_tag at ON at.artist = mra.id AND at.count > 0
        JOIN mb_raw.tag t ON t.id = at.tag
        LEFT JOIN mb_raw.genre gd ON gd.name = lower(t.name)
        LEFT JOIN mb_raw.genre_alias gal ON gal.name = lower(t.name)
        LEFT JOIN mb_raw.genre gc ON gc.id = gal.genre
        WHERE a.mbid IS NOT NULL AND (gd.name IS NOT NULL OR gc.name IS NOT NULL)
        """
    ).fetchall():
        out.setdefault(aid, set()).add(genre)
    return {k: frozenset(v) for k, v in out.items()}


def _artist_tags_knn(conn: Connection, aids: list, g_moments, anchors: dict[str, frozenset]) -> dict[str, dict]:
    """AUDIO tier (ADR-025): borrow genres from an artist's nearest-SOUNDING
    MB-labeled neighbors. For each artist, query the factory audio HNSW for its
    closest 'anchor' artists (those with real MB genres), similarity-weight-vote
    their genres, keep the top within AUDIO_KNN_GATE of the best (capped
    AUDIO_TAG_K, min 1). Anchors carry only real MB genres, so this CANNOT
    hallucinate niche tags. Artists with no anchor neighbors (sparse spaces,
    tests, or a missing embedding) fall back to the centered zero-shot tier so
    the tier is never empty."""
    out: dict[str, dict] = {}
    nofit: list = []
    # Seed embeddings fetched up front: the ANN's query vector MUST be a bound
    # PARAMETER (a literal), not a correlated column — pgvector's HNSW only
    # engages for a constant query vector, so a CROSS JOIN seed seq-scans all
    # ~150k vectors per artist (~2s). As a parameter it's an index probe (~ms).
    embs = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT artist_id::text, embedding::text FROM artist_embedding "
            "WHERE artist_id = ANY(%s::uuid[]) AND model = %s",
            ([str(a) for a in aids], EMB_MODEL),
        ).fetchall()
    }
    for aid in aids:
        qvec = embs.get(str(aid))
        if qvec is None:
            nofit.append(aid)
            continue
        rows = conn.execute(
            """
            SELECT n.artist_id::text, -((n.embedding)::vector(1024) <#> %(q)s::vector(1024)) AS sim
            FROM artist_embedding n
            WHERE n.model = %(m)s AND n.artist_id <> %(a)s::uuid
            ORDER BY (n.embedding)::vector(1024) <#> %(q)s::vector(1024)
            LIMIT %(f)s
            """,
            {"q": qvec, "a": str(aid), "m": EMB_MODEL, "f": AUDIO_KNN_FETCH},
        ).fetchall()
        votes: dict[str, float] = {}
        total = 0.0
        kept = 0
        for nid, sim in rows:
            g = anchors.get(nid)
            if not g:
                continue  # neighbor isn't an MB-labeled anchor — skip
            w = max(0.0, float(sim))
            for genre in g:
                votes[genre] = votes.get(genre, 0.0) + w
            total += w
            kept += 1
            if kept >= AUDIO_KNN_K:
                break
        if not votes or total <= 0:
            nofit.append(aid)
            continue
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])[:AUDIO_TAG_K]
        top = ranked[0][1]
        out[str(aid)] = {
            g: max(1, round(v / total * 10))
            for g, v in ranked
            if g == ranked[0][0] or v >= AUDIO_KNN_GATE * top
        }
    if nofit:  # no sonic anchors -> centered zero-shot fallback (keeps tier non-empty)
        out.update(_artist_tags_centered(conn, nofit, g_moments))
    return out


def publish_artists(factory: Connection, app: Connection, limit: int = 1000, since=None) -> int:
    """Upsert embedded artists into the serving DB. Returns artists published."""
    return publish_rows(factory, app, publishable_artists(factory, limit, since))


def publish_rows(factory: Connection, app: Connection, rows: list[tuple], anchors: dict | None = None) -> int:
    import json

    if not rows:
        return 0
    # set-based reads: one query each for the whole batch, not 5 per artist.
    aids = [r[0] for r in rows]
    urls_by = artist_urls_batch(factory, aids)
    # Tags (ADR-022/025), priority order, never empty:
    #   1. MB editorial genres (accurate)  2. Bandcamp human tags
    #   3. AUDIO tier = genres borrowed from nearest-sounding MB-labeled neighbors
    #      (kNN label-propagation, ADR-025), centered zero-shot as last-resort fallback.
    mb_by = mb_genres_batch(factory, aids)
    bc_by = bandcamp_tags_batch(factory, aids)
    # Genre-only ALLOWLIST + compound tokenization for the Bandcamp folksonomy
    # tier (decision 2026-06-20): the BC human tier is a messy folksonomy, so only
    # curator-APPROVED genre tags survive. Approved tags pass whole; an UNDECIDED
    # compound is tokenized to recover the approved genre inside it (so a band that
    # self-tags only 'progressive doom' still contributes 'doom'); blocked + pure
    # junk ('voodoo') drop out. This keeps the long undecided tail out of serving
    # WITHOUT a hand-block of ~90k tail tags, while losing no real sub-genre
    # signal. MB-editorial and audio tiers are trusted and deliberately NOT gated.
    # Runs BEFORE audio_aids is computed so an artist left with no approved BC
    # signal drops out of bc_by and falls through to the audio tier.
    if factory.execute("SELECT to_regclass('tag_approved')").fetchone()[0] is not None:
        approved = frozenset(r[0] for r in factory.execute("SELECT tag FROM tag_approved").fetchall())
        bl_exists = factory.execute("SELECT to_regclass('tag_manual_blocklist')").fetchone()[0] is not None
        blocked_bc = (
            frozenset(r[0] for r in factory.execute("SELECT tag FROM tag_manual_blocklist").fetchall())
            if bl_exists else frozenset()
        )
        bc_by = {a: r for a, d in bc_by.items() if (r := allowlist_bc_tags(d, approved, blocked_bc))}
    audio_aids = [a for a in aids if str(a) not in mb_by and str(a) not in bc_by]
    audio_by: dict[str, dict] = {}
    if audio_aids:
        g_artist = factory.execute(
            "SELECT avg(score), greatest(stddev(score), 1e-6), count(DISTINCT artist_id) FROM artist_tag_scores"
        ).fetchone()
        # anchors built once per run by the caller; build here if called directly.
        if anchors is None:
            anchors = load_anchor_genres(factory)
        audio_by = _artist_tags_knn(factory, audio_aids, g_artist, anchors)
    # MB + Bandcamp are UNIONED (both human curation); audio is the fallback only
    # when an artist has neither human tier. (Was a cascade that hid Bandcamp
    # folksonomy behind a lone MB genre — see merge_human_tiers.)
    def _human_or_audio(a: str) -> dict:
        mb = mb_by.get(a) or {}
        bc = bc_by.get(a) or {}
        return merge_human_tiers(mb, bc) if (mb or bc) else (audio_by.get(a) or {})

    tags_by = {str(a): _human_or_audio(str(a)) for a in aids}
    # Curated black-hole (#35): drop manually-blocklisted tags across EVERY tier
    # — mainly location-as-genre leaks via Bandcamp human tags (cdmx, mexico,
    # oakland, …). Single chokepoint so MB/BC/audio are all covered; tags are
    # stored lowercase, matching the per-tier lowercasing above.
    has_bl = factory.execute("SELECT to_regclass('tag_manual_blocklist')").fetchone()[0] is not None
    blocked = (
        {r[0] for r in factory.execute("SELECT tag FROM tag_manual_blocklist").fetchall()} if has_bl else set()
    )
    if blocked:
        tags_by = {a: {t: w for t, w in tags.items() if t not in blocked} for a, tags in tags_by.items()}
    perc_by = artist_perceptual_batch(factory, aids)
    lang_by = artist_language_batch(factory, aids)
    loc_by = artist_location_batch(factory, aids)
    published = 0
    for aid, mbid, name, source, embedding, _model, ratio in rows:
        akey = str(aid)
        urls = urls_by.get(akey, {})
        tags = tags_by.get(akey, {})
        perceptual = perc_by.get(akey)
        language = lang_by.get(akey)
        location = loc_by.get(akey)
        url_cols = "".join(f", {c} = %s" for c in urls)
        # Identity key (ADR-019): mbid when MB knows them; otherwise the
        # FACTORY artist id IS the app row id — provisional identity that an
        # accepted MB submission upgrades in place (mbid attaches via sync).
        if mbid:
            # ADR-019 loop close (review finding, critical): an artist first
            # published mbid-NULL has an app row keyed by factory id with
            # mbid NULL — ON CONFLICT (mbid) would MISS it and insert a
            # duplicate. Claim the provisional row first; then the conflict
            # fires and DO UPDATE refreshes in place (id + slug stable).
            app.execute(
                "UPDATE artists SET mbid = %s WHERE id = %s AND mbid IS NULL",
                (mbid, str(aid)),
            )
            conflict = "ON CONFLICT (mbid)"
            id_value, key = "gen_random_uuid()", mbid
        else:
            conflict = "ON CONFLICT (id)"
            id_value, key = "%s", str(aid)
        app.execute(
            f"""
            INSERT INTO artists (id, mbid, name, slug, tags, audio_embedding,
                                 signal_ratio, embedding_source, perceptual,
                                 language, location,
                                 audio_embedding_updated, created_at
                                 {"".join("," + c for c in urls)})
            VALUES ({id_value}, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), now()
                    {", %s" * len(urls)})
            {conflict} DO UPDATE SET
                name = EXCLUDED.name, slug = EXCLUDED.slug, tags = EXCLUDED.tags,
                -- Dirty-mark for incremental revector (#35): NULL the tag_vector
                -- ONLY when tags actually change (jsonb IS DISTINCT FROM is
                -- key-order-independent), so `rebuild-vectors --incremental`
                -- (onlyMissing) rebuilds exactly the changed artists — no full
                -- revector for routine tag changes.
                tag_vector = CASE WHEN artists.tags IS DISTINCT FROM EXCLUDED.tags
                                  THEN NULL ELSE artists.tag_vector END,
                audio_embedding = EXCLUDED.audio_embedding,
                signal_ratio = EXCLUDED.signal_ratio,
                embedding_source = EXCLUDED.embedding_source,
                perceptual = EXCLUDED.perceptual,
                language = coalesce(EXCLUDED.language, artists.language),
                location = coalesce(EXCLUDED.location, artists.location),
                audio_embedding_updated = now()
                {url_cols}
            """,
            ((*(() if mbid else (str(aid),)), mbid, name, resolve_slug(app, name, key),
              json.dumps(tags), embedding,
              ratio, source, json.dumps(perceptual) if perceptual else None,
              language, location,
              *urls.values(), *urls.values())),
        )
        published += 1
    return published


def main() -> None:
    import argparse
    import os

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="publish embedded artists to the serving DB")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--incremental", action="store_true", help="watermark mode (the hourly sync)")
    ap.add_argument("--republish-all", action="store_true",
                    help="re-derive EVERY embedded artist in committed batches "
                         "(for calibration/tag changes the watermark can't trigger)")
    args = ap.parse_args()
    app_dsn = os.environ.get("APP_DATABASE_URL")
    if not app_dsn:
        raise SystemExit("APP_DATABASE_URL not set — publishing is deliberate, no default")
    with psycopg.connect(Settings().database_url) as factory, psycopg.connect(app_dsn) as app:
        if args.incremental:
            n = publish_incremental(factory, app, args.limit)
            # COMMIT ORDER LAW (review finding, high): app rows FIRST, then
            # the watermark. A crash between them re-publishes idempotent
            # upserts next run; the reverse order silently never publishes
            # the lost window again.
            app.commit()
            factory.commit()
        elif args.republish_all:
            # capture the run-start BEFORE the loop; republish_all commits per
            # batch internally. Set the watermark to that start AFTER success so
            # the next hourly incremental catches anything embedded DURING the run
            # (publish-sync must stay stopped until this finishes).
            start = factory.execute("SELECT now()").fetchone()[0]
            n = republish_all(factory, app, batch=args.limit,
                              progress=lambda t, a: print(f"republished={t} after={a}", flush=True))
            factory.execute("UPDATE publish_watermark SET last_run = %s WHERE id = 'default'", (start,))
            factory.commit()
        else:
            n = publish_artists(factory, app, args.limit)
            app.commit()
    print(f"published={n}", flush=True)


if __name__ == "__main__":
    main()
