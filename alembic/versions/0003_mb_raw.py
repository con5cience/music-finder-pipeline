"""MB bootstrap tables — mb_raw dump mirror + artist.mbid linkage (ADR-017).

`mb_raw` mirrors the 8 MusicBrainz fullexport tables the pipeline consumes,
in the exact column layout of the upstream schema (verified against
musicbrainz-server CreateTables.sql and the 20260606 dump). UNLOGGED: this is
a refreshable cache reloaded from dumps, and the diff base for the periodic
refresh — losing it on crash costs a reload, not data.

artist.mbid is the MusicBrainz linkage key (NULL for non-MB artists, if any
ever exist).

Revision ID: 0003_mb_raw
Revises: 0002_embeddings
"""

from alembic import op

revision = "0003_mb_raw"
down_revision = "0002_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE artist ADD COLUMN mbid uuid;
        CREATE UNIQUE INDEX idx_artist_mbid ON artist (mbid) WHERE mbid IS NOT NULL;

        CREATE SCHEMA mb_raw;

        CREATE UNLOGGED TABLE mb_raw.artist (
            id integer PRIMARY KEY,
            gid uuid NOT NULL,
            name text NOT NULL,
            sort_name text NOT NULL,
            begin_date_year smallint, begin_date_month smallint, begin_date_day smallint,
            end_date_year smallint, end_date_month smallint, end_date_day smallint,
            type integer, area integer, gender integer,
            comment text NOT NULL DEFAULT '',
            edits_pending integer NOT NULL DEFAULT 0,
            last_updated timestamptz,
            ended boolean NOT NULL DEFAULT false,
            begin_area integer, end_area integer
        );

        CREATE UNLOGGED TABLE mb_raw.artist_alias (
            id integer PRIMARY KEY,
            artist integer NOT NULL,
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

        CREATE UNLOGGED TABLE mb_raw.artist_tag (
            artist integer NOT NULL,
            tag integer NOT NULL,
            count integer NOT NULL,
            last_updated timestamptz
        );

        CREATE UNLOGGED TABLE mb_raw.tag (
            id integer PRIMARY KEY,
            name text NOT NULL,
            ref_count integer NOT NULL DEFAULT 0
        );

        CREATE UNLOGGED TABLE mb_raw.url (
            id integer PRIMARY KEY,
            gid uuid NOT NULL,
            url text NOT NULL,
            edits_pending integer NOT NULL DEFAULT 0,
            last_updated timestamptz
        );

        CREATE UNLOGGED TABLE mb_raw.l_artist_url (
            id integer PRIMARY KEY,
            link integer NOT NULL,
            entity0 integer NOT NULL,
            entity1 integer NOT NULL,
            edits_pending integer NOT NULL DEFAULT 0,
            last_updated timestamptz,
            link_order integer NOT NULL DEFAULT 0,
            entity0_credit text NOT NULL DEFAULT '',
            entity1_credit text NOT NULL DEFAULT ''
        );

        CREATE UNLOGGED TABLE mb_raw.link (
            id integer PRIMARY KEY,
            link_type integer NOT NULL,
            begin_date_year smallint, begin_date_month smallint, begin_date_day smallint,
            end_date_year smallint, end_date_month smallint, end_date_day smallint,
            attribute_count integer NOT NULL DEFAULT 0,
            created timestamptz,
            ended boolean NOT NULL DEFAULT false
        );

        CREATE UNLOGGED TABLE mb_raw.link_type (
            id integer PRIMARY KEY,
            parent integer,
            child_order integer NOT NULL DEFAULT 0,
            gid uuid NOT NULL,
            entity_type0 text NOT NULL,
            entity_type1 text NOT NULL,
            name text NOT NULL,
            description text,
            link_phrase text NOT NULL,
            reverse_link_phrase text NOT NULL,
            long_link_phrase text NOT NULL,
            last_updated timestamptz,
            is_deprecated boolean NOT NULL DEFAULT false,
            has_dates boolean NOT NULL DEFAULT true,
            entity0_cardinality integer NOT NULL DEFAULT 0,
            entity1_cardinality integer NOT NULL DEFAULT 0
        );

        -- Join paths used by derive_identities (and the future refresh diff).
        CREATE INDEX idx_mb_raw_lau_entity0 ON mb_raw.l_artist_url (entity0);
        CREATE INDEX idx_mb_raw_lau_entity1 ON mb_raw.l_artist_url (entity1);
        CREATE INDEX idx_mb_raw_lau_link ON mb_raw.l_artist_url (link);
        CREATE INDEX idx_mb_raw_artist_tag_artist ON mb_raw.artist_tag (artist);
        CREATE INDEX idx_mb_raw_artist_alias_artist ON mb_raw.artist_alias (artist);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP SCHEMA mb_raw CASCADE;
        DROP INDEX IF EXISTS idx_artist_mbid;
        ALTER TABLE artist DROP COLUMN IF EXISTS mbid;
        """
    )
