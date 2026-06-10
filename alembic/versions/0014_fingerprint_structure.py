"""Fingerprint + Wave-3 structure tables.

track_fingerprint: compact spectral fingerprint per track (band-energy bit
matrix, hamming-comparable) + an exact-dup hash. Self-contained (no
chromaprint system dep); upgradeable later without schema change.
track_structure: Wave-3 librosa segmentation summary (gated head).
fingerprint findings are FLAG-ONLY (review_item evidence), consistent with
the integrity-gate law: no automated exclusion until corpus-calibrated.

Revision ID: 0014_fingerprint_structure
Revises: 0013_refresh_serial
"""

from alembic import op

revision = "0014_fingerprint_structure"
down_revision = "0013_refresh_serial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE track_fingerprint (
            track_id    uuid PRIMARY KEY REFERENCES audio_track(id) ON DELETE CASCADE,
            fp          bytea NOT NULL,
            fp_seconds  real NOT NULL,
            exact_hash  text NOT NULL,
            computed_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE INDEX idx_fp_exact ON track_fingerprint (exact_hash);

        CREATE TABLE track_structure (
            track_id         uuid PRIMARY KEY REFERENCES audio_track(id) ON DELETE CASCADE,
            n_sections       integer NOT NULL,
            avg_section_s    real NOT NULL,
            repetition_ratio real NOT NULL,
            boundaries_s     real[] NOT NULL,
            computed_at      timestamptz NOT NULL DEFAULT now()
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS track_structure; DROP TABLE IF EXISTS track_fingerprint;")
