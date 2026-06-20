"""Apply a JSON of AI tag verdicts ({approve:[...], block:[...]}) to the factory
tag tables with source='ai'. Idempotent: only acts on UNDECIDED tags (ON
CONFLICT DO NOTHING never overwrites an existing human/auto/ai decision), and
never deletes a human decision. Usage: python scripts/apply_tag_verdicts.py FILE
"""

from __future__ import annotations

import json
import sys

import psycopg

from pipeline.config import Settings


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tag_verdicts.json"
    data = json.load(open(path))
    approve = sorted({t.strip().lower() for t in data.get("approve", []) if t.strip()})
    block = sorted({t.strip().lower() for t in data.get("block", []) if t.strip()})
    overlap = set(approve) & set(block)
    if overlap:
        raise SystemExit(f"tag in BOTH approve and block: {sorted(overlap)}")

    with psycopg.connect(Settings().database_url) as conn:
        conn.cursor().executemany(
            "INSERT INTO tag_approved (tag, category, source) VALUES (%s,'genre','ai') ON CONFLICT (tag) DO NOTHING",
            [(t,) for t in approve],
        )
        conn.cursor().executemany(
            "INSERT INTO tag_manual_blocklist (tag, reason, source, category) "
            "VALUES (%s,'ai:non-genre','ai','non-genre') ON CONFLICT (tag) DO NOTHING",
            [(t,) for t in block],
        )
        conn.commit()
    print(f"applied: approve={len(approve)} block={len(block)}")


if __name__ == "__main__":
    main()
