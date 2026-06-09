"""Integration-test fixtures: apply migrations once, hand each test a rolled-back conn.

Tests that need Postgres take the `conn` fixture; if the dev DB (compose `db` on
5440) isn't up, they skip rather than fail, so unit tests still run bare.
"""

from __future__ import annotations

import os

import psycopg
import pytest
from alembic import command
from alembic.config import Config

DB_URL = os.environ.get("PIPELINE_DATABASE_URL", "postgresql://pipeline:pipeline@localhost:5440/pipeline")
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


@pytest.fixture(scope="session")
def migrated_db() -> str:
    try:
        psycopg.connect(DB_URL, connect_timeout=2).close()
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not available at {DB_URL} ({e}); run `docker compose up -d db`")
    cfg = Config(os.path.join(_PROJECT_ROOT, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(_PROJECT_ROOT, "alembic"))
    command.upgrade(cfg, "head")
    return DB_URL


@pytest.fixture
def conn(migrated_db: str):
    c = psycopg.connect(migrated_db)
    try:
        yield c
    finally:
        c.rollback()  # each test's writes are discarded
        c.close()
