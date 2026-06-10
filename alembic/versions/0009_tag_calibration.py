"""Per-tag score calibration (the designed follow-up to Wave-1 tags).

Raw MuLan cosines are compressed (corpus: mean 0.471, sd 0.079) and carry
vocabulary priors — some tags score high for everything ('chap hop' top-
ranked 129 tracks). tag_calibration stores per-(tag, model) corpus moments;
consumers rank by z = (score - mean) / sd, which damps exactly the prior
noise. Raw scores in track_tag_scores stay untouched (immutable evidence;
recalibration is always possible).

Revision ID: 0009_tag_calibration
Revises: 0008_search_bind
"""

from alembic import op

revision = "0009_tag_calibration"
down_revision = "0008_search_bind"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE tag_calibration (
            tag         text NOT NULL,
            model       text NOT NULL,
            mean        real NOT NULL,
            stddev      real NOT NULL,
            n           integer NOT NULL,
            computed_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (tag, model)
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS tag_calibration;")
