"""mb_submission gains 'duplicate' — the URL-checked create-anyway policy
(ADR-019): when an artist's bandcamp/streaming URL already exists in MB the
artist is a true duplicate, so we record the existing MBID to LINK rather than
create a second entity in the commons. Name-only collisions are NOT duplicates
(they create-anyway with a disambiguation)."""

from alembic import op

revision = "0027_mb_submission_duplicate"
down_revision = "0026_tag_audio_blocklist"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE mb_submission DROP CONSTRAINT mb_submission_status_check")
    op.execute("""ALTER TABLE mb_submission ADD CONSTRAINT mb_submission_status_check
        CHECK (status = ANY (ARRAY['queued','spot_check','approved','submitted','accepted','failed','duplicate']))""")


def downgrade() -> None:
    op.execute("ALTER TABLE mb_submission DROP CONSTRAINT mb_submission_status_check")
    op.execute("""ALTER TABLE mb_submission ADD CONSTRAINT mb_submission_status_check
        CHECK (status = ANY (ARRAY['queued','spot_check','approved','submitted','accepted','failed']))""")
