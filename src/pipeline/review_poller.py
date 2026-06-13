"""Tier-C decision poller — the pipeline half of the decision-table pattern.

The admin panel (sibling repo) writes DECISIONS into review_item: status
'approved' with evidence.decision = {platform, platform_id} (the human-chosen
candidate), or 'rejected' (none match). This poller applies approved
decisions: creates a Tier-C platform_identity (scan_status pending → the
cascade ingests it like any other) and stamps resolved_at. The admin never
touches pipeline tables beyond review_item — single writer per table.

Run:  uv run poe review-poll          (cron/loop later; idempotent)
"""

from __future__ import annotations

import json

from psycopg import Connection


def apply_approved_bindings(conn: Connection, limit: int = 500) -> int:
    """Apply admin-approved source_binding decisions. Returns count applied."""
    rows = conn.execute(
        """
        SELECT id, subject_id, evidence FROM review_item
        WHERE kind = 'source_binding' AND status = 'approved' AND resolved_at IS NULL
          AND COALESCE(reason, '') <> 'artist_homonym'
        ORDER BY created_at
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    applied = 0
    for rid, artist_id, evidence in rows:
        decision = (evidence or {}).get("decision") or {}
        platform, platform_id = decision.get("platform"), decision.get("platform_id")
        if platform and platform_id:
            conn.execute(
                """
                INSERT INTO platform_identity (artist_id, platform, platform_id, page_type,
                                               binding_tier, binding_evidence)
                VALUES (%s, %s, %s, 'artist', 'C', %s)
                ON CONFLICT DO NOTHING
                """,
                (artist_id, platform, platform_id,
                 # decision.method distinguishes human approval from acoustic
                 # auto-adjudication (auto_coherence carries its cosine)
                 json.dumps({"method": decision.get("method", "admin_review"),
                             "review_item_id": str(rid),
                             **({"cosine": decision["cosine"]} if "cosine" in decision else {})})),
            )
            applied += 1
        # resolved either way: a decision without a usable candidate is closed
        # as actioned (the admin's note explains; nothing to bind)
        conn.execute("UPDATE review_item SET resolved_at = now() WHERE id = %s", (rid,))
    return applied


def take_approved_homonym_picks(conn: Connection, limit: int = 500) -> list[str]:
    """Resolve admin-picked homonym items and return the chosen artist ids to
    front-run. Marking resolved here (not after the workflow start) keeps the
    operator's decision recorded even if a start fails — the artist would then
    embed via the normal seeder, just not jumped to the front. Sync + DB-only
    so it's unit-testable without Temporal."""
    rows = conn.execute(
        """
        SELECT id, evidence FROM review_item
        WHERE kind = 'source_binding' AND status = 'approved' AND resolved_at IS NULL
          AND reason = 'artist_homonym'
        ORDER BY created_at
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    chosen: list[str] = []
    for rid, evidence in rows:
        pick = ((evidence or {}).get("decision") or {}).get("chosen_artist")
        if pick:
            chosen.append(str(pick))
        conn.execute("UPDATE review_item SET resolved_at = now() WHERE id = %s", (rid,))
    return chosen


async def front_run_artists(artist_ids: list[str]) -> int:
    """Start IngestArtistWorkflow for each picked homonym winner so it jumps
    ahead of the mbid-bound backlog. already-started is fine (idempotent id)."""
    if not artist_ids:
        return 0
    from temporalio.client import Client

    from pipeline.config import Settings
    from pipeline.seed_ingest import workflow_id
    from pipeline.workflows import IngestArtistInput, IngestArtistWorkflow

    s = Settings()
    client = await Client.connect(s.temporal_address, namespace=s.temporal_namespace)
    started = 0
    for aid in artist_ids:
        try:
            await client.start_workflow(
                IngestArtistWorkflow.run, IngestArtistInput(aid),
                id=workflow_id(aid), task_queue=s.temporal_task_queue)
            started += 1
        except Exception:  # noqa: BLE001 — already running is fine
            pass
    return started


def main() -> None:
    import asyncio
    import psycopg

    from pipeline.config import Settings

    with psycopg.connect(Settings().database_url) as conn:
        n = apply_approved_bindings(conn)
        picks = take_approved_homonym_picks(conn)
        conn.commit()  # decisions recorded before the (best-effort) front-run
        started = asyncio.run(front_run_artists(picks))
    print(f"applied={n} homonym_picks={len(picks)} front_run_started={started}", flush=True)


if __name__ == "__main__":
    main()
