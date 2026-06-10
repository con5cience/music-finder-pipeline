"""Artist-level tag scores — scored from the ARTIST-MEAN MuLan vector at
embed time. Per-track top-20 truncation made artist aggregation pathological
(empirical: an artist's twelve track lists can be nearly disjoint, so
consistent moderate signals lose to single-track flukes). Full-resolution
artist scoring has no truncation artifact; per-track rows remain for
track-level UX and calibration.

Revision ID: 0011_artist_tags
Revises: 0010_worker_heartbeat
"""

from alembic import op

revision = "0011_artist_tags"
down_revision = "0010_worker_heartbeat"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE artist_tag_scores (
            artist_id  uuid NOT NULL REFERENCES artist(id) ON DELETE CASCADE,
            tag        text NOT NULL,
            score      real NOT NULL,
            model      text NOT NULL,
            computed_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (artist_id, tag, model)
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS artist_tag_scores;")
