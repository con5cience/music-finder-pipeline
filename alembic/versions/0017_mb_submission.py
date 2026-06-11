"""mb_submission: the ADR-019 contribution-lane ledger. One row per artist
submitted to MusicBrainz; status transitions submitted → accepted (detected
by mb-sync matching our URLs) / failed. oauth_token holds the bot account's
refresh token (single-row table keyed 'default').

Revision ID: 0017_mb_submission
Revises: 0016_bc_candidate
"""

from alembic import op

revision = "0017_mb_submission"
down_revision = "0016_bc_candidate"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE mb_submission (
            id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            artist_id    uuid NOT NULL UNIQUE REFERENCES artist(id),
            payload      jsonb NOT NULL,
            edit_refs    jsonb,
            submitted_at timestamptz,
            status       text NOT NULL DEFAULT 'queued'
                         CHECK (status IN ('queued','spot_check','submitted','accepted','failed')),
            status_note  text,
            created_at   timestamptz NOT NULL DEFAULT now()
        );
        CREATE TABLE mb_oauth (
            id            text PRIMARY KEY DEFAULT 'default',
            refresh_token text NOT NULL,
            updated_at    timestamptz NOT NULL DEFAULT now()
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS mb_submission; DROP TABLE IF EXISTS mb_oauth;")
