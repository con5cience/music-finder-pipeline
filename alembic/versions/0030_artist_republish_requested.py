"""artist.republish_requested_at — admin-edit re-publish signal.

Admin field edits (name/mbid, later URLs/location/manual-tags) are written to the
FACTORY artist row (the source of truth), NOT to serving (a read-only projection).
This nullable timestamp is set to now() on such an edit so the incremental publish
sync (publishable_artists) re-projects the artist to serving.

Why a DEDICATED column and not artist.updated_at: updated_at DEFAULTs to now() at
creation, so it can't distinguish "freshly created" from "edited" and would make
every recently-created artist look re-publishable. republish_requested_at is NULL
by default (creation never sets it), so it matches ONLY genuinely-edited artists.
Additive + nullable → safe, no backfill.
"""

from alembic import op

revision = "0030_artist_republish_requested"
down_revision = "0029_tag_review"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE artist ADD COLUMN republish_requested_at timestamptz")


def downgrade() -> None:
    op.execute("ALTER TABLE artist DROP COLUMN republish_requested_at")
