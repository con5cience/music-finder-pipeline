"""tag_audio_blocklist — the data-driven magnet prune (ADR-020 Phase 4). Tags the
AUDIO model sprays on a large share of artists but MB editors essentially never
use are anisotropy artifacts (kilapanga 21.9% audio / 0 MB; orthodox pop 19.4% /
57 MB; pumpcore, fm synthesis, geek rock, …). They are excluded from the
audio-tag tier at publish (MB + Bandcamp tiers are untouched — if MB genuinely
uses a tag, it stays). Refreshed by tags.refresh_audio_blocklist."""

from alembic import op

revision = "0026_tag_audio_blocklist"
down_revision = "0025_tag_centering"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE tag_audio_blocklist (
            tag         text        PRIMARY KEY,
            audio_pct   real        NOT NULL,
            mb_n        integer     NOT NULL,
            computed_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE tag_audio_blocklist")
