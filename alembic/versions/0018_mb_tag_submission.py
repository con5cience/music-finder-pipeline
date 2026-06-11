"""mb_tag_submission: phase-2 tag-lane ledger (review finding: no ledger =
the same N artists re-submitted forever, zero coverage progress).

Revision ID: 0018_mb_tag_submission
Revises: 0017_mb_submission
"""

from alembic import op

revision = "0018_mb_tag_submission"
down_revision = "0017_mb_submission"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE TABLE mb_tag_submission (artist_id uuid PRIMARY KEY REFERENCES artist(id), "
        "submitted_at timestamptz NOT NULL DEFAULT now())"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS mb_tag_submission;")
