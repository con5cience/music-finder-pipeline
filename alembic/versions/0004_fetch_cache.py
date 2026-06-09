"""Fetch cache index — every non-audio third-party fetch persists (ADR-017 §5).

The table indexes gzipped filesystem blobs (content-addressed by sha256, so
identical bodies share one file). One row per URL, latest fetch wins; audio
bytes are NEVER cached (fetch-embed-delete). The blobs make re-parsing free:
improved binding logic replays the cache instead of burning rate budget.

Revision ID: 0004_fetch_cache
Revises: 0003_mb_raw
"""

from alembic import op

revision = "0004_fetch_cache"
down_revision = "0003_mb_raw"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE fetch_cache (
            id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            platform     text NOT NULL,
            url          text NOT NULL UNIQUE,
            status       smallint NOT NULL,
            content_type text,
            content_hash text NOT NULL,
            content_path text NOT NULL,
            fetched_at   timestamptz NOT NULL DEFAULT now()
        );
        CREATE INDEX idx_fetch_cache_platform ON fetch_cache (platform);
        CREATE INDEX idx_fetch_cache_hash ON fetch_cache (content_hash);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS fetch_cache;")
