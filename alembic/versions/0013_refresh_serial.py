"""mb_refresh_run.serial — the dump serial each run consumed, so mb-sync can
skip an already-applied fullexport (MetaBrainz publishes 2x/week; we consume
when the serial moves).

Revision ID: 0013_refresh_serial
Revises: 0012_mb_refresh
"""

from alembic import op

revision = "0013_refresh_serial"
down_revision = "0012_mb_refresh"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE mb_refresh_run ADD COLUMN serial text")


def downgrade() -> None:
    op.execute("ALTER TABLE mb_refresh_run DROP COLUMN IF EXISTS serial")
