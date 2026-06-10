"""Integration-test fixtures: apply migrations once, hand each test a rolled-back conn.

Tests run against a DEDICATED `pipeline_test` database (created on first run
inside the factory Postgres) — never the live corpus. Three hermeticity
collisions (real-capture ids, fetch-cache hits, backfill scooping corpus
rows) and one suite-vs-live-fleet hang earned this isolation; it mirrors the
sibling repo's db_test law. If Postgres itself is down, DB tests skip.
"""

from __future__ import annotations

import os

import psycopg
import pytest
from alembic import command
from alembic.config import Config

ADMIN_URL = os.environ.get("PIPELINE_DATABASE_URL", "postgresql://pipeline:pipeline@localhost:5440/pipeline")
DB_URL = os.environ.get(
    "PIPELINE_TEST_DATABASE_URL", "postgresql://pipeline:pipeline@localhost:5440/pipeline_test"
)
assert DB_URL.rsplit("/", 1)[-1].endswith("_test"), "test DB name must end with _test"
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


@pytest.fixture(scope="session")
def migrated_db() -> str:
    try:
        admin = psycopg.connect(ADMIN_URL, connect_timeout=2, autocommit=True)
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not available ({e}); run `docker compose up -d factory-db`")
    test_db = DB_URL.rsplit("/", 1)[-1]
    exists = admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (test_db,)).fetchone()
    if not exists:
        admin.execute(f'CREATE DATABASE "{test_db}"')
    admin.close()
    cfg = Config(os.path.join(_PROJECT_ROOT, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(_PROJECT_ROOT, "alembic"))
    prev = os.environ.get("PIPELINE_DATABASE_URL")
    os.environ["PIPELINE_DATABASE_URL"] = DB_URL  # alembic env.py reads Settings
    try:
        command.upgrade(cfg, "head")
    finally:
        if prev is None:
            os.environ.pop("PIPELINE_DATABASE_URL", None)
        else:
            os.environ["PIPELINE_DATABASE_URL"] = prev
    return DB_URL


@pytest.fixture
def conn(migrated_db: str):
    c = psycopg.connect(migrated_db)
    try:
        yield c
    finally:
        c.rollback()  # each test's writes are discarded
        c.close()
