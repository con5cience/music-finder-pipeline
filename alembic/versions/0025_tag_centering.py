"""tag_centering — per-tag dot product d_i = (genre text embedding) . (corpus-mean
audio direction), the ADR-020 Phase-5 "centering" demotion computed offline once.

At publish, the artist tag ranking becomes `score - C*d_i` instead of the
z-score: the z divided by each tag's spread, which INFLATED low-spread tags
(scattered/magnet genres); subtracting C*d_i instead demotes tags aligned with
the dominant (anisotropy) audio direction. Validated to recover ~2/3 of full
re-embed centering and to sharpen per-artist tags (a punk band -> punk, a metal
band -> metal) — using ONLY stored scores, so it covers the whole corpus with no
re-embed. d_i needs the MuLan model (vocab text embeddings) so it is precomputed
here and read at publish (publish-sync has no model)."""

from alembic import op

revision = "0025_tag_centering"
down_revision = "0024_audio_clip_archive"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE tag_centering (
            tag         text        NOT NULL,
            model       text        NOT NULL,
            d           real        NOT NULL,
            mu_version  text        NOT NULL,
            n_sample    integer     NOT NULL,
            computed_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (tag, model)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE tag_centering")
