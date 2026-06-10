"""worker_heartbeat: fleet liveness for the admin Workers card (no Redis,
no docker introspection — pure DB rows upserted by each worker process).

Revision ID: 0010_worker_heartbeat
Revises: 0009_tag_calibration
"""

from alembic import op

revision = "0010_worker_heartbeat"
down_revision = "0009_tag_calibration"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE worker_heartbeat (
            role      text NOT NULL,
            hostname  text NOT NULL,
            queues    text NOT NULL,
            last_seen timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (role, hostname)
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS worker_heartbeat;")
