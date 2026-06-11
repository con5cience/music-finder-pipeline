"""Publish watermark (incremental sync) + the ban ledger.

publish_watermark: single-row high-water mark — incremental publish sends
only artists whose embedding/tags changed since. ban_ledger: the
do-not-rediscover list; banned artists are excluded from publish AND from
discovery re-admission (dedup/admit check platform ids + mbid + name).

Revision ID: 0019_publish_watermark_bans
Revises: 0018_mb_tag_submission
"""

from alembic import op

revision = "0019_publish_watermark_bans"
down_revision = "0018_mb_tag_submission"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE publish_watermark (
            id text PRIMARY KEY DEFAULT 'default',
            last_run timestamptz NOT NULL DEFAULT 'epoch'
        );
        INSERT INTO publish_watermark (id) VALUES ('default');

        CREATE TABLE ban_ledger (
            id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            artist_id uuid REFERENCES artist(id),
            mbid uuid,
            display_name text NOT NULL,
            platform_ids jsonb NOT NULL DEFAULT '[]',
            reason text,
            banned_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE INDEX idx_ban_mbid ON ban_ledger (mbid);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ban_ledger; DROP TABLE IF EXISTS publish_watermark;")
