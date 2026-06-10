"""mb_refresh_run: one ledger row per ADR-018 refresh (serial, gates, diff
counts, applied_at NULL = dry-run). The admin factory card can surface the
latest row; the runbook's monthly procedure reads like a checklist of these.

Revision ID: 0012_mb_refresh
Revises: 0011_artist_tags
"""

from alembic import op

revision = "0012_mb_refresh"
down_revision = "0011_artist_tags"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE mb_refresh_run (
            id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            started_at  timestamptz NOT NULL DEFAULT now(),
            gates       jsonb NOT NULL,
            adds        integer,
            new_identities integer,
            renames     integer,
            merges      integer,
            reviews     integer,
            applied_at  timestamptz
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS mb_refresh_run;")
