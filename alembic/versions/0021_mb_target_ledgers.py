"""Per-target MB submission ledgers (2026-06-12).

test.musicbrainz.org and musicbrainz.org are SEPARATE servers with separate
accounts, OAuth apps, and databases. One untargeted ledger meant a rehearsal
on test would mark artists as submitted FOR LIVE (skipping them forever) and
one oauth row meant the wrong server's token got used silently. Every
submission surface now carries its target; oauth rows are keyed by target.
"""

from alembic import op

revision = "0021_mb_target_ledgers"
down_revision = "0020_slop_evaluation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE mb_tag_submission ADD COLUMN target text NOT NULL DEFAULT 'live'")
    op.execute("ALTER TABLE mb_tag_submission DROP CONSTRAINT mb_tag_submission_pkey")
    op.execute("ALTER TABLE mb_tag_submission ADD PRIMARY KEY (artist_id, target)")
    op.execute("ALTER TABLE mb_submission ADD COLUMN target text NOT NULL DEFAULT 'live'")
    op.execute("ALTER TABLE mb_submission ADD COLUMN created_mbid uuid")
    # oauth: one row per target ('live'/'test'), existing row was live
    op.execute("UPDATE mb_oauth SET id = 'live' WHERE id = 'default'")


def downgrade() -> None:
    op.execute("UPDATE mb_oauth SET id = 'default' WHERE id = 'live'")
    op.execute("ALTER TABLE mb_submission DROP COLUMN created_mbid")
    op.execute("ALTER TABLE mb_submission DROP COLUMN target")
    op.execute("ALTER TABLE mb_tag_submission DROP CONSTRAINT mb_tag_submission_pkey")
    op.execute("ALTER TABLE mb_tag_submission ADD PRIMARY KEY (artist_id)")
    op.execute("ALTER TABLE mb_tag_submission DROP COLUMN target")
