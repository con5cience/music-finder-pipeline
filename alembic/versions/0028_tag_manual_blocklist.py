"""tag_manual_blocklist — a CURATED, durable tag blocklist (#35 follow-up). The
data-driven tag_audio_blocklist (0026) is auto-rebuilt and only prunes the AUDIO
tier; it can't hold manual entries and never touches the Bandcamp/MB tiers. This
table is for human-curated black-holes — mainly location-as-genre leaks that come
through the Bandcamp human tags (cdmx, mexico, oakland, san francisco, …). It is
applied at the final tag-assembly chokepoint in publish (every tier) and is
NEVER auto-wiped. Managed via `poe block-tag` / `poe unblock-tag`."""

from alembic import op

revision = "0028_tag_manual_blocklist"
down_revision = "0027_mb_submission_duplicate"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE tag_manual_blocklist (
            tag      text        PRIMARY KEY,
            reason   text,
            added_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE tag_manual_blocklist")
