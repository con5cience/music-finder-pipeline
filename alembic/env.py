"""Alembic environment — URL driven by pipeline.config.Settings (PIPELINE_DATABASE_URL).

SQL-first: revisions carry raw DDL, so there is no ORM metadata to autogenerate
against (target_metadata = None).
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from pipeline.config import Settings

config = context.config

# Drive the connection from pipeline settings; SQLAlchemy needs the psycopg driver.
_url = Settings().database_url
if _url.startswith("postgresql://"):
    _url = _url.replace("postgresql://", "postgresql+psycopg://", 1)
config.set_main_option("sqlalchemy.url", _url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
