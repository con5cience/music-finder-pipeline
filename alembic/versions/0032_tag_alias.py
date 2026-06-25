"""tag_alias — a CURATED garbage/misspelling -> canonical tag map (factory
source-of-truth). Unlike the blocklist, which DROPS a junk tag (losing its
signal), an alias FOLDS the tag's weight onto a real canonical tag at publish, so
a misspelling ('eletronica') or punctuation-variant ('#folk', 'e.l.e.c.t.r.o')
still contributes to recommendations under the right name. Applied at the
final tag-assembly chokepoint in publish, BEFORE the manual blocklist drop, so a
folded alias survives as its (un-blocked) canonical. Complements the MB
genre_alias merge (synth-pop->synthpop) for the manual/folksonomy layer."""

from alembic import op

revision = "0032_tag_alias"
down_revision = "0031_pi_artist_id_idx"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE tag_alias (
            alias     text PRIMARY KEY,
            canonical text NOT NULL,
            source    text NOT NULL DEFAULT 'ai',
            added_at  timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT tag_alias_no_self CHECK (alias <> canonical)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE tag_alias")
