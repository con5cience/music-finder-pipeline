"""`poe backup` one-shot: backs up BOTH stores and verifies each dump.

Static guards on the script + its poe wiring — the dump itself needs live DB
containers, exercised by running `poe backup` (see commit 2026-06-18). These
pins keep the script covering both unreconstructible stores (factory ledgers +
embeddings AND serving crates + accounts) and never shipping an unverified dump.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_backup_covers_both_stores():
    src = (ROOT / "scripts" / "backup.sh").read_text()
    # factory store
    assert "FACTORY_DB_CONTAINER" in src or "factory-db" in src
    assert "dump_db pipeline" in src
    # serving store (crates/accounts) — the gap the original factory-only script left
    assert "SERVING_DB_CONTAINER" in src or "music-finder-db-1" in src
    assert "dump_db serving" in src


def test_backup_excludes_reconstructible_factory_data():
    # fetch-cache + audio are re-fetchable; don't bloat the dump with them
    src = (ROOT / "scripts" / "backup.sh").read_text()
    assert "--exclude-table-data=fetch_cache" in src


def test_backup_verifies_each_dump():
    # never ship an unverified dump: read the archive TOC back and fail on empty
    src = (ROOT / "scripts" / "backup.sh").read_text()
    assert "pg_restore --list" in src
    assert "VERIFY FAILED" in src


def test_backup_is_safe_shell():
    src = (ROOT / "scripts" / "backup.sh").read_text()
    assert "set -euo pipefail" in src


def test_poe_backup_task_wired():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "backup = " in pyproject and "scripts/backup.sh" in pyproject
