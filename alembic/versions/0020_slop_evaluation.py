"""slop_evaluation: the continuous AI-slop gate's ledger (2026-06-12).

One row per evaluated artist: score + the catalog size it was judged at.
Publish and MB-queue evaluate lazily at their choke points — any artist
without a row (or whose catalog grew past the evaluated size) is scored
in the same cycle that would otherwise expose them. The future audio
ai_likelihood head writes the same ledger.
"""

from alembic import op

revision = "0020_slop_evaluation"
down_revision = "0019_publish_watermark_bans"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE slop_evaluation (
            artist_id uuid PRIMARY KEY REFERENCES artist(id) ON DELETE CASCADE,
            score real NOT NULL,
            n_tracks int NOT NULL,
            features jsonb,
            evaluated_at timestamptz NOT NULL DEFAULT now()
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE slop_evaluation")
