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


def main() -> None:
    import psycopg

    from pipeline.config import Settings

    with psycopg.connect(Settings().database_url) as conn:
        n = apply_approved_bindings(conn)
        conn.commit()
    print(f"applied={n}", flush=True)


if __name__ == "__main__":
    main()
