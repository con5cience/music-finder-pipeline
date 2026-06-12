"""Binding integrity audit — the report that would have caught the typo tier.

One-shot (no-cron law): distribution of binding provenance, open integrity
flags, and a random evidence sample for eyeball verification. Run weekly-ish
or after any binder change:  uv run poe binding-audit
"""

from __future__ import annotations

from psycopg import Connection


def audit(conn: Connection, sample: int = 10) -> dict:
    methods = conn.execute(
        """
        SELECT binding_tier, COALESCE(binding_evidence->>'method', 'mb-dump/discovery'), count(*)
        FROM platform_identity GROUP BY 1, 2 ORDER BY 3 DESC
        """
    ).fetchall()
    flags = conn.execute(
        """
        SELECT reason, count(*) FROM review_item
        WHERE kind = 'source_binding' AND status = 'pending'
          AND reason IN ('source_coherence', 'mb_shared_url')
        GROUP BY 1
        """
    ).fetchall()
    recent = conn.execute(
        """
        SELECT a.display_name, pi.platform, pi.binding_tier,
               COALESCE(pi.binding_evidence->>'method', 'mb-dump/discovery'),
               COALESCE(pi.binding_evidence->>'candidate_name', pi.platform_id)
        FROM platform_identity pi JOIN artist a ON a.id = pi.artist_id
        WHERE pi.binding_tier <> 'A'
        ORDER BY pi.first_seen_at DESC LIMIT %s
        """,
        (sample,),
    ).fetchall()
    return {"methods": methods, "open_flags": flags, "recent_non_a": recent}


def main() -> None:
    import psycopg

    from pipeline.config import Settings

    with psycopg.connect(Settings().database_url) as conn:
        out = audit(conn)
        print("binding provenance:")
        for tier, method, n in out["methods"]:
            print(f"  {tier}  {method:24s} {n:>10,}")
        print("open integrity flags:")
        if not out["open_flags"]:
            print("  none")
        for reason, n in out["open_flags"]:
            print(f"  {reason:24s} {n:>6}")
        print(f"newest non-A bindings (eyeball these):")
        for name, platform, tier, method, cand in out["recent_non_a"]:
            print(f"  [{tier}/{method}] {name!r} -> {cand!r} on {platform}")


if __name__ == "__main__":
    main()
