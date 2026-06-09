"""Baseline schema — artist / platform_identity / audio_track / review_item + fanout guard.

Greenfield consolidated baseline (ADR-015). Embedding tables are intentionally a
*later* migration: pgvector needs a fixed dimension, decided by the model bench.

Revision ID: 0001_baseline
Revises:
"""

from alembic import op

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        -- Canonical, one-record-per-artist entity (UX-level identity).
        CREATE TABLE artist (
            id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            display_name text NOT NULL,
            status       text NOT NULL DEFAULT 'active'
                          CHECK (status IN ('active','merged','hidden')),
            created_at   timestamptz NOT NULL DEFAULT now(),
            updated_at   timestamptz NOT NULL DEFAULT now()
        );

        -- An artist's (or a source's) presence on a platform, keyed by the
        -- platform's IMMUTABLE numeric id. artist_id is NULL for non-artist
        -- source pages (labels/compilations); set for artist/topic pages.
        CREATE TABLE platform_identity (
            id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            artist_id         uuid REFERENCES artist(id) ON DELETE CASCADE,
            platform          text NOT NULL,
            platform_id       text NOT NULL,
            vanity_url        text,
            page_type         text NOT NULL DEFAULT 'unknown'
                               CHECK (page_type IN ('artist','label','compilation','topic','unknown')),
            first_seen_at     timestamptz NOT NULL DEFAULT now(),
            last_validated_at timestamptz,
            UNIQUE (platform, platform_id),
            -- artist/topic pages name one artist; label/compilation/unknown do not.
            CHECK ((page_type IN ('artist','topic')) = (artist_id IS NOT NULL))
        );

        -- An embeddable audio item attributed to an artist, with verification
        -- tier + provenance. from_identity_id may point at a *label* page.
        CREATE TABLE audio_track (
            id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            artist_id           uuid NOT NULL REFERENCES artist(id) ON DELETE CASCADE,
            platform            text NOT NULL,
            platform_track_id   text NOT NULL,
            audio_url           text,
            duration_s          integer,
            from_identity_id    uuid REFERENCES platform_identity(id) ON DELETE SET NULL,
            binding_tier        text NOT NULL CHECK (binding_tier IN ('A','B1','B2','C')),
            binding_evidence    jsonb NOT NULL DEFAULT '{}'::jsonb,
            verification_status text NOT NULL DEFAULT 'pending_review'
                                 CHECK (verification_status IN ('verified','pending_review','rejected','quarantined')),
            discovered_at       timestamptz NOT NULL DEFAULT now(),
            UNIQUE (platform, platform_track_id)
        );
        CREATE INDEX idx_audio_track_artist ON audio_track (artist_id);
        CREATE INDEX idx_audio_track_from_identity ON audio_track (from_identity_id);

        -- The human-review queue (Tier C, B2 disagreements, gate quarantines).
        CREATE TABLE review_item (
            id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            kind         text NOT NULL CHECK (kind IN ('source_binding','dedup','quality_gate')),
            subject_type text NOT NULL,
            subject_id   uuid NOT NULL,
            reason       text NOT NULL,
            evidence     jsonb NOT NULL DEFAULT '{}'::jsonb,
            status       text NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending','approved','rejected','deferred')),
            created_at   timestamptz NOT NULL DEFAULT now(),
            resolved_at  timestamptz,
            note         text
        );
        CREATE INDEX idx_review_item_status ON review_item (status) WHERE status = 'pending';

        -- Fanout guard (DB building block): an artist/topic page belongs to ONE
        -- artist, so its tracks must credit that artist. A page cannot spray
        -- tracks for other artists (the exact contamination class). Labels and
        -- compilations are exempt — they legitimately credit many artists.
        CREATE FUNCTION enforce_artist_page_binding() RETURNS trigger AS $$
        DECLARE
            pt  text;
            pid uuid;
        BEGIN
            IF NEW.from_identity_id IS NULL THEN
                RETURN NEW;
            END IF;
            SELECT page_type, artist_id INTO pt, pid
            FROM platform_identity WHERE id = NEW.from_identity_id;
            IF pt IN ('artist','topic') AND NEW.artist_id <> pid THEN
                RAISE EXCEPTION USING
                    ERRCODE = 'check_violation',
                    MESSAGE = 'fanout guard: track ' || NEW.platform_track_id
                              || ' from ' || pt || ' page ' || NEW.from_identity_id::text
                              || ' credits a different artist than the page owner';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        CREATE TRIGGER trg_enforce_artist_page_binding
            BEFORE INSERT OR UPDATE OF artist_id, from_identity_id ON audio_track
            FOR EACH ROW EXECUTE FUNCTION enforce_artist_page_binding();
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TRIGGER IF EXISTS trg_enforce_artist_page_binding ON audio_track;
        DROP FUNCTION IF EXISTS enforce_artist_page_binding();
        DROP TABLE IF EXISTS review_item;
        DROP TABLE IF EXISTS audio_track;
        DROP TABLE IF EXISTS platform_identity;
        DROP TABLE IF EXISTS artist;
        """
    )
