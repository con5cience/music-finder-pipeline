"""ADR-018: MusicBrainz refresh — shadow re-import with derive-diff.

Monthly fullexport → mb_raw_next (live mb_raw keeps serving) → fail-closed
sanity gates → diff applied through EXISTING machinery (derive_identities is
idempotent: against the shadow schema it lands only NEW artists/identities,
which the wave seeder then eats) → renames → merges via artist_gid_redirect
(both-embedded conflicts go to the Tier-C review queue, never auto-picked)
→ transactional schema swap (prior generation kept as mb_raw_old for one
cycle). DRY-RUN BY DEFAULT: gates + diff report only; --apply is deliberate.

Run:  uv run poe mb-refresh -- --dir <extracted-mbdump> [--apply]
"""

from __future__ import annotations

import json
from pathlib import Path

from psycopg import Connection

from pipeline.mb_bootstrap import EXPECTED_COLS, derive_identities, load_mbdump

REFRESH_TABLES = dict(EXPECTED_COLS)
REFRESH_TABLES["artist_gid_redirect"] = 3  # gid, new_id, created — merge map

ARTIST_GATE = 0.98   # new artist count must be >= this share of current
URLREL_GATE = 0.95


def prepare_shadow(conn: Connection) -> None:
    """mb_raw_next mirroring mb_raw's tables (LIKE keeps types/defaults)."""
    conn.execute("DROP SCHEMA IF EXISTS mb_raw_next CASCADE")
    conn.execute("CREATE SCHEMA mb_raw_next")
    for table in EXPECTED_COLS:
        conn.execute(
            f"CREATE UNLOGGED TABLE mb_raw_next.{table} (LIKE mb_raw.{table} INCLUDING ALL)"
        )
    conn.execute(
        """
        CREATE UNLOGGED TABLE mb_raw_next.artist_gid_redirect (
            gid uuid NOT NULL, new_id integer NOT NULL, created timestamptz
        )
        """
    )


def sanity_gates(conn: Connection) -> dict:
    """Fail-closed: a truncated/partial dump must abort before any apply."""
    cur_a = conn.execute("SELECT count(*) FROM mb_raw.artist").fetchone()[0]
    new_a = conn.execute("SELECT count(*) FROM mb_raw_next.artist").fetchone()[0]
    cur_u = conn.execute("SELECT count(*) FROM mb_raw.l_artist_url").fetchone()[0]
    new_u = conn.execute("SELECT count(*) FROM mb_raw_next.l_artist_url").fetchone()[0]
    gates = {
        "artists": {"current": cur_a, "next": new_a, "ok": new_a >= cur_a * ARTIST_GATE},
        "url_rels": {"current": cur_u, "next": new_u, "ok": new_u >= cur_u * URLREL_GATE},
    }
    gates["ok"] = all(g["ok"] for g in (gates["artists"], gates["url_rels"]))
    return gates


_MERGES_SQL = """
    SELECT x.id, x.mbid::text, nn.gid::text,
           (x.embedding_source IS NOT NULL) AS old_embedded,
           (t.id IS NOT NULL) AS target_exists,
           (t.embedding_source IS NOT NULL) AS target_embedded
    FROM artist x
    JOIN mb_raw_next.artist_gid_redirect r ON r.gid = x.mbid::uuid
    JOIN mb_raw_next.artist nn ON nn.id = r.new_id
    LEFT JOIN artist t ON t.mbid = nn.gid
    """


def diff_and_apply(conn: Connection, *, apply: bool) -> dict:
    """The derive-diff. Dry-run computes counts without mutating the derived
    layer; apply runs the (idempotent) derivation + renames + merges."""
    out: dict = {}
    # adds preview: derived candidates absent from the live artist table
    out["adds"] = conn.execute(
        """
        SELECT count(DISTINCT a.gid) FROM mb_raw_next.l_artist_url lau
        JOIN mb_raw_next.artist a ON a.id = lau.entity0
        WHERE NOT EXISTS (SELECT 1 FROM artist x WHERE x.mbid = a.gid)
        """
    ).fetchone()[0]
    out["renames"] = conn.execute(
        """
        SELECT count(*) FROM artist x JOIN mb_raw_next.artist n ON n.gid = x.mbid
        WHERE x.display_name != n.name
        """
    ).fetchone()[0]
    merges = conn.execute(_MERGES_SQL).fetchall()
    out["merges"] = len(merges)
    out["reviews"] = sum(1 for m in merges if m[3] and m[5])

    if not apply:
        return out

    before = conn.execute("SELECT count(*) FROM platform_identity").fetchone()[0]
    derive_identities(conn, schema="mb_raw_next")  # idempotent → only NEW lands
    out["new_identities"] = conn.execute(
        "SELECT count(*) FROM platform_identity"
    ).fetchone()[0] - before
    # RE-SNAPSHOT after derive (review finding): derive can INSERT a merge
    # TARGET (it inherits the merged artist's URLs), flipping target_exists.
    # The stale snapshot caused unique violations on artist.mbid.
    merges = conn.execute(_MERGES_SQL).fetchall()
    conn.execute(
        """
        UPDATE artist x SET display_name = n.name
        FROM mb_raw_next.artist n WHERE n.gid = x.mbid AND x.display_name != n.name
        """
    )
    for aid, old_mbid, new_mbid, old_emb, target_exists, target_emb in merges:
        if not target_exists:
            # per-row recheck (review finding): an EARLIER loop iteration may
            # have repointed another local artist to this same target —
            # two locals merging into one absent target must not both UPDATE.
            t = conn.execute(
                "SELECT embedding_source IS NOT NULL FROM artist WHERE mbid = %s AND id != %s",
                (new_mbid, aid),
            ).fetchone()
            if t is not None:
                target_exists, target_emb = True, t[0]
        if not target_exists:
            conn.execute("UPDATE artist SET mbid = %s WHERE id = %s", (new_mbid, aid))
        elif old_emb and target_emb:
            conn.execute(
                """
                INSERT INTO review_item (kind, subject_type, subject_id, reason, evidence, status)
                VALUES ('source_binding', 'artist', %s, 'mb_merge: both sides embedded', %s, 'pending')
                """,
                (aid, json.dumps({"mb_merge": {"old_mbid": old_mbid, "new_mbid": new_mbid}})),
            )
        else:
            # keep whichever side is embedded; move identities to it, retire the other
            keep_old = old_emb
            conn.execute(
                """
                UPDATE platform_identity pi SET artist_id = t.id
                FROM artist t WHERE t.mbid = %s AND pi.artist_id = (
                    SELECT id FROM artist WHERE mbid = %s)
                AND NOT EXISTS (SELECT 1 FROM platform_identity q
                    WHERE q.platform = pi.platform AND q.platform_id = pi.platform_id
                    AND q.artist_id = t.id)
                """,
                (old_mbid, new_mbid) if keep_old else (new_mbid, old_mbid),
            )
            loser_mbid = new_mbid if keep_old else old_mbid
            conn.execute("DELETE FROM artist WHERE mbid = %s AND embedding_source IS NULL",
                         (loser_mbid,))
            if keep_old:
                conn.execute("UPDATE artist SET mbid = %s WHERE id = %s", (new_mbid, aid))
    return out


def swap(conn: Connection) -> None:
    """Old generation survives one cycle as mb_raw_old."""
    conn.execute("DROP SCHEMA IF EXISTS mb_raw_old CASCADE")
    conn.execute("ALTER SCHEMA mb_raw RENAME TO mb_raw_old")
    conn.execute("ALTER SCHEMA mb_raw_next RENAME TO mb_raw")


def run_refresh(
    conn: Connection, dump_dir: Path | str, *, apply: bool, serial: str | None = None
) -> dict:
    """Every run — including gate-failure aborts — gets its OWN ledger row
    carrying its serial (review finding: the old post-hoc UPDATE-max(id)
    stamped a failed serial onto the previous APPLIED row, fail-open)."""
    prepare_shadow(conn)
    load_mbdump(conn, dump_dir, schema="mb_raw_next", tables=REFRESH_TABLES)
    gates = sanity_gates(conn)
    report: dict = {"gates": gates}
    if not gates["ok"]:
        conn.execute("DROP SCHEMA mb_raw_next CASCADE")
        report["aborted"] = "sanity gates failed — live state untouched"
    else:
        report.update(diff_and_apply(conn, apply=apply))
        if apply:
            swap(conn)
    conn.execute(
        """
        INSERT INTO mb_refresh_run (serial, gates, adds, new_identities, renames, merges, reviews, applied_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, CASE WHEN %s THEN now() END)
        """,
        (serial, json.dumps(gates), report.get("adds"), report.get("new_identities"),
         report.get("renames"), report.get("merges"), report.get("reviews"),
         apply and gates["ok"]),
    )
    return report


def main() -> None:
    import argparse

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="ADR-018 MB refresh (dry-run by default)")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    with psycopg.connect(Settings().database_url) as conn:
        report = run_refresh(conn, args.dir, apply=args.apply)
        conn.commit()
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
