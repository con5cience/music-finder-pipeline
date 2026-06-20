"""tag review model for the admin Tags tab (genre-only curation policy).

Three pieces:
  - tag_approved: the human/auto "keep" set (thumbs-up). Mirrors
    tag_manual_blocklist (the "block" set / thumbs-down); a tag in NEITHER is
    "undecided" and is what the review tab surfaces by default.
  - provenance columns on both tables (source human|auto, category) so the
    auto-classifier's decisions are auditable and a human decision can be
    distinguished from (and never overwritten by) an auto one.
  - tag_review_freq: a MATERIALIZED snapshot of per-tag corpus frequency
    (distinct lower(tag) -> #bc_candidate rows). A live GROUP BY over unnest of
    ~98k bc_candidate rows is ~400ms; the tab must not pay that per page load, so
    it reads this MV (refreshed by `poe refresh-tag-freq` / the classifier). The
    UNIQUE index enables REFRESH ... CONCURRENTLY.
"""

from alembic import op

revision = "0029_tag_review"
down_revision = "0028_tag_manual_blocklist"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE tag_approved (
            tag      text        PRIMARY KEY,
            category text,
            source   text        NOT NULL DEFAULT 'human',
            added_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("ALTER TABLE tag_manual_blocklist ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT 'human'")
    op.execute("ALTER TABLE tag_manual_blocklist ADD COLUMN IF NOT EXISTS category text")
    op.execute(
        """
        CREATE MATERIALIZED VIEW tag_review_freq AS
        SELECT lower(t) AS tag, count(*)::bigint AS df
        FROM bc_candidate, unnest(tags) t
        WHERE t <> ''
        GROUP BY 1
        WITH DATA
        """
    )
    op.execute("CREATE UNIQUE INDEX tag_review_freq_tag ON tag_review_freq (tag)")
    op.execute("CREATE INDEX tag_review_freq_df ON tag_review_freq (df DESC)")


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS tag_review_freq")
    op.execute("ALTER TABLE tag_manual_blocklist DROP COLUMN IF EXISTS source")
    op.execute("ALTER TABLE tag_manual_blocklist DROP COLUMN IF EXISTS category")
    op.execute("DROP TABLE IF EXISTS tag_approved")
