"""bc_candidate: the Bandcamp discovery ledger (ADR-019).

Every band surfaced by the Discover API lands here exactly once. States:
candidate (awaiting admission) / dedup_existing (we already know them) /
admitted (artist row created, factory takes over) / rejected (quality gate,
with reason). Carries the MB-extraction stash (location, links) so the
submission lane never re-crawls.

Revision ID: 0016_bc_candidate
Revises: 0015_wave3_heavy
"""

from alembic import op

revision = "0016_bc_candidate"
down_revision = "0015_wave3_heavy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE bc_candidate (
            id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            platform_id   text NOT NULL UNIQUE,
            band_name     text NOT NULL,
            band_url      text NOT NULL,
            location      text,
            genre         text,
            tags          text[] NOT NULL DEFAULT '{}',
            links         jsonb,
            release_seen_at timestamptz,
            first_seen_at timestamptz NOT NULL DEFAULT now(),
            status        text NOT NULL DEFAULT 'candidate'
                          CHECK (status IN ('candidate','dedup_existing','admitted','rejected')),
            status_reason text,
            artist_id     uuid REFERENCES artist(id)
        );
        CREATE INDEX idx_bc_candidate_status ON bc_candidate (status, first_seen_at);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS bc_candidate;")
