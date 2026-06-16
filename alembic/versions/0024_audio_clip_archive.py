"""audio_clip_archive — ledger for the retained COMPRESSED source clips (ADR-021
Tier B). The embed pass fetches a track's compressed audio, decodes it, embeds,
and throws the audio away; this ledger records the one compressed clip we now
keep per track (on the archive volume) so an embedding-model swap or a windowing
change is a local decode+embed, never a trip back through the slow, rate-limited,
dead-URL-prone acquisition pipeline.

One row per track (the selected source clip). rel_path is relative to
PIPELINE_ARCHIVE_DIR so the archive can be relocated/restored without rewriting
rows. The ledger answers coverage ("which tracks can re-analyze without a fetch")
and is the backup manifest for the irreplaceable clip store."""

from alembic import op

revision = "0024_audio_clip_archive"
down_revision = "0023_artist_analysis_vector"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE audio_clip_archive (
            track_id    uuid        PRIMARY KEY REFERENCES audio_track(id) ON DELETE CASCADE,
            artist_id   uuid        NOT NULL REFERENCES artist(id) ON DELETE CASCADE,
            platform    text        NOT NULL,
            rel_path    text        NOT NULL,
            bytes       bigint      NOT NULL,
            archived_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    # coverage / backfill queries scan by artist ("is this artist re-analyzable
    # without re-fetching any track?")
    op.execute("CREATE INDEX idx_audio_clip_archive_artist ON audio_clip_archive (artist_id)")


def downgrade() -> None:
    op.execute("DROP TABLE audio_clip_archive")
