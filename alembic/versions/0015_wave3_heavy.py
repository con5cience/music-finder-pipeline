"""Wave-3 heavy-head tables: stem energy profile (Demucs) + language (ASR).

Revision ID: 0015_wave3_heavy
Revises: 0014_fingerprint_structure
"""

from alembic import op

revision = "0015_wave3_heavy"
down_revision = "0014_fingerprint_structure"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE track_stems (
            track_id     uuid PRIMARY KEY REFERENCES audio_track(id) ON DELETE CASCADE,
            vocal_ratio  real NOT NULL,
            drums_ratio  real NOT NULL,
            bass_ratio   real NOT NULL,
            other_ratio  real NOT NULL,
            computed_at  timestamptz NOT NULL DEFAULT now()
        );
        CREATE TABLE track_language (
            track_id    uuid PRIMARY KEY REFERENCES audio_track(id) ON DELETE CASCADE,
            language    text NOT NULL,
            confidence  real NOT NULL,
            computed_at timestamptz NOT NULL DEFAULT now()
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS track_language; DROP TABLE IF EXISTS track_stems;")
