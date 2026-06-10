"""Wave-1 analysis storage (ADR-015 decode-once multi-head) + genre vocabulary.

track_analysis: one row per analyzed track — fingerprint (chromaprint, for 3d
dedup/verification), MIR features, integrity verdict (FLAG-ONLY in v1: gates
are recorded, never enforced, until corpus-calibrated). analysis_version
allows head upgrades to re-run selectively. track_tag_scores: per-track
zero-shot tag scores (PER-TRACK per user decision; artist-level aggregates
derive at publish). mb_raw.genre/genre_alias mirror MB's editor-curated
vocabulary — canonical names + alias merges ("synth punk" → "synth-punk").

Revision ID: 0006_analysis
Revises: 0005_cascade
"""

from alembic import op

revision = "0006_analysis"
down_revision = "0005_cascade"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE UNLOGGED TABLE mb_raw.genre (
            id integer PRIMARY KEY,
            gid uuid NOT NULL,
            name text NOT NULL,
            comment text NOT NULL DEFAULT '',
            edits_pending integer NOT NULL DEFAULT 0,
            last_updated timestamptz
        );
        CREATE UNLOGGED TABLE mb_raw.genre_alias (
            id integer PRIMARY KEY,
            genre integer NOT NULL,
            name text NOT NULL,
            locale text,
            edits_pending integer NOT NULL DEFAULT 0,
            last_updated timestamptz,
            type integer,
            sort_name text NOT NULL,
            begin_date_year smallint, begin_date_month smallint, begin_date_day smallint,
            end_date_year smallint, end_date_month smallint, end_date_day smallint,
            primary_for_locale boolean NOT NULL DEFAULT false,
            ended boolean NOT NULL DEFAULT false
        );

        CREATE TABLE track_analysis (
            track_id          uuid PRIMARY KEY REFERENCES audio_track(id) ON DELETE CASCADE,
            analysis_version  integer NOT NULL DEFAULT 1,
            fingerprint       text,
            analyzed_s        real,
            tempo_bpm         real,
            key               text,
            mode              text,
            spectral_centroid_hz real,
            rms               real,
            loudness_lufs     real,
            silence_frac      real,
            clip_frac         real,
            integrity         text NOT NULL DEFAULT 'ok'
                               CHECK (integrity IN ('ok','silent','clipped','nonmusic','short')),
            computed_at       timestamptz NOT NULL DEFAULT now()
        );

        CREATE TABLE track_tag_scores (
            track_id   uuid NOT NULL REFERENCES audio_track(id) ON DELETE CASCADE,
            tag        text NOT NULL,
            score      real NOT NULL,
            model      text NOT NULL,
            computed_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (track_id, tag, model)
        );
        CREATE INDEX idx_track_tag_scores_tag ON track_tag_scores (tag);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS track_tag_scores;
        DROP TABLE IF EXISTS track_analysis;
        DROP TABLE IF EXISTS mb_raw.genre_alias;
        DROP TABLE IF EXISTS mb_raw.genre;
        """
    )
