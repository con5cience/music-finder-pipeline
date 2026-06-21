"""platform_identity(artist_id) index — admin dashboard / Tier-C perf.

Postgres does not auto-index FK columns, so platform_identity.artist_id was
unindexed. The admin Tier-C list builds an `identities` JSON per row with a
correlated `SELECT ... FROM platform_identity WHERE artist_id = a.id`, which
seq-scanned the whole table PER ROW (×50) — ~4.3s, run twice (pending+deferred),
the bulk of the ~12s admin-status load. This index makes each lookup an index
scan (<1ms).

IF NOT EXISTS: on the live DB the index is created out-of-band via
`CREATE INDEX CONCURRENTLY` (no ACCESS EXCLUSIVE lock on the cascade-written
table), so this migration is a no-op there; on a fresh DB it builds the index
normally. (The migration runner wraps each migration in a transaction, so
CONCURRENTLY can't run here.)
"""

from alembic import op

revision = "0031_pi_artist_id_idx"  # <=32 chars (alembic_version.version_num)
down_revision = "0030_artist_republish_requested"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE INDEX IF NOT EXISTS ix_platform_identity_artist_id ON platform_identity (artist_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_platform_identity_artist_id")
