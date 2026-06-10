"""Slice 3d: B-tier search binding storage.

platform_identity grows binding_tier/binding_evidence (Tier-A url-rel rows
default 'A'; search-bound rows are 'B' with the search evidence). search_attempt
is the per-(artist, platform) ledger so searches are never repeated — verdicts:
bound / review / none. Ambiguous (multi-exact) candidates land in review_item
(kind 'tier_c_binding') for the admin Merge Review workflow.

Revision ID: 0008_search_bind
Revises: 0007_wave2
"""

from alembic import op

revision = "0008_search_bind"
down_revision = "0007_wave2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE platform_identity
            ADD COLUMN binding_tier text NOT NULL DEFAULT 'A'
                CHECK (binding_tier IN ('A','B','C')),
            ADD COLUMN binding_evidence jsonb;

        CREATE TABLE search_attempt (
            artist_id   uuid NOT NULL REFERENCES artist(id) ON DELETE CASCADE,
            platform    text NOT NULL,
            query       text NOT NULL,
            verdict     text NOT NULL CHECK (verdict IN ('bound','review','none')),
            candidates  integer NOT NULL DEFAULT 0,
            searched_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (artist_id, platform)
        );
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS search_attempt;
        ALTER TABLE platform_identity
            DROP COLUMN IF EXISTS binding_tier,
            DROP COLUMN IF EXISTS binding_evidence;
        """
    )
