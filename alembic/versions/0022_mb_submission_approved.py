"""mb_submission gains 'approved' — the human blessing between spot_check
and live submission (live submits approved-only; test rehearses on
spot_check)."""

from alembic import op

revision = "0022_mb_submission_approved"
down_revision = "0021_mb_target_ledgers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE mb_submission DROP CONSTRAINT mb_submission_status_check")
    op.execute("""ALTER TABLE mb_submission ADD CONSTRAINT mb_submission_status_check
        CHECK (status = ANY (ARRAY['queued','spot_check','approved','submitted','accepted','failed']))""")


def downgrade() -> None:
    op.execute("ALTER TABLE mb_submission DROP CONSTRAINT mb_submission_status_check")
    op.execute("""ALTER TABLE mb_submission ADD CONSTRAINT mb_submission_status_check
        CHECK (status = ANY (ARRAY['queued','spot_check','submitted','accepted','failed']))""")
