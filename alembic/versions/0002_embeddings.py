"""Embedding tables — model-stamped clip vectors + artist centroid cache (ADR-016).

Design (locked after the model bench):
- `embedding` columns are UNTYPED `vector` so one schema stores any model's dims
  (MuQ/MusicFM = 1024 today; 512-dim text-capable models later). A `dim` stamp +
  CHECK keeps rows honest; ANN indexing uses pgvector's documented pattern for
  mixed dims: an expression + partial index per active model.
- `model` is the embedder registry name (e.g. 'muq-large-msd') — the ADR-016
  version stamp that makes a future model swap an additive re-embed, not an
  archaeology project. UNIQUE keys include it so re-embeds coexist.
- `vector_ip_ops` (inner product) because AudioEmbedder L2-normalizes every
  vector — same ranking as cosine, cheaper to compute.

Revision ID: 0002_embeddings
Revises: 0001_baseline
"""

from alembic import op

revision = "0002_embeddings"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None

# The registry's DEFAULT_EMBEDDER at the time this migration was written; the
# partial ANN indexes below cover it. A model swap adds indexes in a new
# migration — it does not edit this one (append-only, like ADRs).
_DEFAULT_MODEL = "muq-large-msd"
_DEFAULT_DIM = 1024


def upgrade() -> None:
    op.execute(
        f"""
        CREATE EXTENSION IF NOT EXISTS vector;

        -- One row per (clip x model). A clip is an inline segment of a track
        -- (ADR-011 precedent) — no separate clip table until clips carry
        -- independent state.
        CREATE TABLE clip_embedding (
            id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            track_id        uuid NOT NULL REFERENCES audio_track(id) ON DELETE CASCADE,
            segment_start_s integer NOT NULL,
            segment_end_s   integer NOT NULL,
            model           text NOT NULL,
            dim             smallint NOT NULL,
            embedding       vector NOT NULL,
            created_at      timestamptz NOT NULL DEFAULT now(),
            CHECK (segment_end_s > segment_start_s),
            CHECK (vector_dims(embedding) = dim),
            UNIQUE (track_id, segment_start_s, model)
        );
        CREATE INDEX idx_clip_embedding_track ON clip_embedding (track_id);
        CREATE INDEX idx_clip_embedding_muq_ann ON clip_embedding
            USING hnsw ((embedding::vector({_DEFAULT_DIM})) vector_ip_ops)
            WHERE model = '{_DEFAULT_MODEL}';

        -- Materialized per-artist centroid (mean of that artist's clip
        -- embeddings, re-normalized). The artist-similarity query surface.
        CREATE TABLE artist_embedding (
            artist_id   uuid NOT NULL REFERENCES artist(id) ON DELETE CASCADE,
            model       text NOT NULL,
            dim         smallint NOT NULL,
            embedding   vector NOT NULL,
            clip_count  integer NOT NULL,
            computed_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (artist_id, model),
            CHECK (vector_dims(embedding) = dim)
        );
        CREATE INDEX idx_artist_embedding_muq_ann ON artist_embedding
            USING hnsw ((embedding::vector({_DEFAULT_DIM})) vector_ip_ops)
            WHERE model = '{_DEFAULT_MODEL}';
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS artist_embedding;
        DROP TABLE IF EXISTS clip_embedding;
        """
    )
