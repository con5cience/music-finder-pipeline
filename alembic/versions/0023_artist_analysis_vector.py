"""artist_analysis_vector — persist the MuLan per-window + artist-mean audio
vectors (ADR-021 Tier A) so corpus re-analysis (re-score, centering, calibration
changes, vocabulary changes, new analysis heads, re-aggregation/window-weighting)
becomes a math pass over stored vectors instead of a re-fetch + re-decode of the
whole corpus.

These are the raw MuLan AUDIO embeddings — vocabulary-INDEPENDENT (the vocabulary
only enters at scoring time, vocab_matrix @ mean). So provenance is stamped with
the producing model + the window-selection config version only; a vocab change
re-scores these vectors without touching them. A model swap or a window-scheme
change is what invalidates a row (detectable via model / window_version).

Stored as an unbounded pgvector with a vector_dims CHECK, mirroring
artist_embedding; no ANN index — these are read by offline batch re-analysis, not
online similarity search, so an HNSW build would be pure cost."""

from alembic import op

revision = "0023_artist_analysis_vector"
down_revision = "0022_mb_submission_approved"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE artist_analysis_vector (
            artist_id      uuid        NOT NULL REFERENCES artist(id) ON DELETE CASCADE,
            model          text        NOT NULL,
            kind           text        NOT NULL,   -- 'mean' (idx 0) | 'window' (idx 0..M-1)
            idx            integer     NOT NULL,
            dim            smallint    NOT NULL,
            embedding      vector      NOT NULL,
            window_version text        NOT NULL,
            computed_at    timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (artist_id, model, kind, idx),
            CONSTRAINT artist_analysis_vector_kind_check
                CHECK (kind IN ('mean', 'window')),
            CONSTRAINT artist_analysis_vector_dim_check
                CHECK (vector_dims(embedding) = dim)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE artist_analysis_vector")
