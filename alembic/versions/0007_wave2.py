"""Wave-2 perceptual head storage + per-head completion tracking.

track_perceptual: MuLan zero-shot ANCHOR-PAIR axis scores (raw cosine
differences, uncalibrated — corpus distributions calibrate later, same
posture as tags) + top instruments. track_head_runs: which head versions ran
per track — the embed pass and backfill key per-head idempotency on THIS
(review-era gap: keying on track_analysis alone hid tag-less tracks).

Revision ID: 0007_wave2
Revises: 0006_analysis
"""

from alembic import op

revision = "0007_wave2"
down_revision = "0006_analysis"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE track_perceptual (
            track_id      uuid PRIMARY KEY REFERENCES audio_track(id) ON DELETE CASCADE,
            danceability  real,
            valence       real,
            arousal       real,
            speechiness   real,
            liveness      real,
            vocalness     real,
            instruments   jsonb,           -- top-k [{"name": ..., "score": ...}]
            model         text NOT NULL,
            computed_at   timestamptz NOT NULL DEFAULT now()
        );

        CREATE TABLE track_head_runs (
            track_id    uuid NOT NULL REFERENCES audio_track(id) ON DELETE CASCADE,
            head        text NOT NULL,
            version     integer NOT NULL,
            computed_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (track_id, head)
        );
        """
    )
    # Wave-1 rows predate head tracking: backfill their completion records so
    # the per-head idempotency doesn't re-run the whole corpus.
    op.execute(
        """
        INSERT INTO track_head_runs (track_id, head, version, computed_at)
        SELECT track_id, 'cpu_analysis', 1, computed_at FROM track_analysis
        ON CONFLICT DO NOTHING;
        INSERT INTO track_head_runs (track_id, head, version, computed_at)
        SELECT DISTINCT track_id, 'tags', 1, min(computed_at) FROM track_tag_scores
        GROUP BY track_id
        ON CONFLICT DO NOTHING;
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS track_head_runs; DROP TABLE IF EXISTS track_perceptual;")
