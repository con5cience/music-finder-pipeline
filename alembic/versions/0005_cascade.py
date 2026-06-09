"""Cascade state — scan verdicts, embedding source lock, signal ratio (ADR-017 §2/§3).

scan_status on platform_identity: terminal verdicts only (ADR-013 semantics) —
'scanned' (yield recorded), 'empty' (source had nothing), 'error' is NOT
terminal and is never written; transient failures leave 'pending' so the row
re-qualifies. artist.embedding_source locks centroid purity to one platform;
artist_embedding.signal_ratio records yield/floor at embed time so weak-signal
artists are queryable (publish gating, supersede targeting).

Revision ID: 0005_cascade
Revises: 0004_fetch_cache
"""

from alembic import op

revision = "0005_cascade"
down_revision = "0004_fetch_cache"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE platform_identity
            ADD COLUMN scan_status text NOT NULL DEFAULT 'pending'
                CHECK (scan_status IN ('pending','scanned','empty')),
            ADD COLUMN scanned_at timestamptz;

        ALTER TABLE artist ADD COLUMN embedding_source text;

        ALTER TABLE artist_embedding ADD COLUMN signal_ratio real;

        CREATE INDEX idx_platform_identity_scan_pending
            ON platform_identity (platform) WHERE scan_status = 'pending';
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_platform_identity_scan_pending;
        ALTER TABLE artist_embedding DROP COLUMN IF EXISTS signal_ratio;
        ALTER TABLE artist DROP COLUMN IF EXISTS embedding_source;
        ALTER TABLE platform_identity DROP COLUMN IF EXISTS scan_status, DROP COLUMN IF EXISTS scanned_at;
        """
    )
