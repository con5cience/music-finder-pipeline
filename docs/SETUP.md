# Setup & runbook

How to stand up the pipeline stack — Postgres (the local "factory" DB), Temporal,
and the worker — on the Mac for dev and on the CUDA box for real runs.

## Ports

| Service | Address |
|---|---|
| Postgres (factory DB) | `localhost:5440` |
| Temporal gRPC | `localhost:7233` |
| Temporal Web UI | `localhost:8233` |

## Prerequisites

```sh
# uv (Python toolchain — installs/pins Python 3.12 itself)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Docker (for the Postgres container)  — already present on the Mac

# Temporal CLI (dev server + tctl-style tooling)
brew install temporal                              # macOS
curl -sSf https://temporal.download/cli.sh | sh    # Linux (adds ~/.temporalio/bin to PATH)
```

## Tasks (the "npm scripts" — `uv run poe <task>`)

Frequent actions live in `[tool.poe.tasks]` in `pyproject.toml`, so they're
stable, named commands (easy to allow-list):

| Task | Runs |
|---|---|
| `uv run poe test` | `pytest -q` |
| `uv run poe lint` / `fix` / `fmt` | `ruff check .` / `--fix` / `ruff format .` |
| `uv run poe check` | lint then test |
| `uv run poe db-up` / `db-down` | `docker compose up -d db` / `down` |
| `uv run poe migrate` | `alembic upgrade head` |
| `uv run poe migrate-down` | `alembic downgrade -1` |
| `uv run poe migrate-new "msg"` | `alembic revision -m "msg"` |
| `uv run poe migrate-status` | `alembic current` |
| `uv run poe worker` | `python -m pipeline.worker` |
| `uv run poe temporal` | `temporal server start-dev` |

## Local dev (Mac)

```sh
cd ~/Documents/g/music-finder-pipeline
uv sync                            # venv + deps (managed Python 3.12)
docker compose up -d db            # Postgres 16 on 5440
uv run alembic upgrade head        # apply the schema

# in a second terminal — Temporal dev server (Web UI at http://localhost:8233):
temporal server start-dev

# in a third terminal — the worker (prints device=mps/cpu on the Mac):
uv run python -m pipeline.worker

# checks
uv run pytest -q                   # unit + integration (needs db + temporal test server)
uv run ruff check .                # lint
```

Tests are resilient: they **skip** (not fail) if Postgres or the Temporal test
server isn't reachable, so `uv run pytest -q` works on a bare checkout too.

## The CUDA box (Ubuntu 26.04) — tomorrow

```sh
# uv, Docker, Temporal CLI via the Linux installers above.
# NVIDIA driver + CUDA toolkit (12.x) for the RTX 4070 Ti SUPER.

git clone <repo> && cd music-finder-pipeline
uv sync
docker compose up -d db
uv run alembic upgrade head
temporal server start-dev --ip 0.0.0.0   # or run under systemd for persistence
uv run python -m pipeline.worker         # prints device=cuda automatically
```

GPU lights up by default — `device.select_device()` returns `cuda` when torch
sees the card. Force with `PIPELINE_DEVICE=cpu|mps|cuda`. (torch + the CLAP/audio
models are added in the embedding slice; the foundation runs without them.)

## Configuration (env, `PIPELINE_` prefix — defaults in `src/pipeline/config.py`)

| Var | Default | Notes |
|---|---|---|
| `PIPELINE_DATABASE_URL` | `postgresql://pipeline:pipeline@localhost:5440/pipeline` | the factory DB |
| `PIPELINE_TEMPORAL_ADDRESS` | `localhost:7233` | |
| `PIPELINE_TEMPORAL_NAMESPACE` | `default` | |
| `PIPELINE_TEMPORAL_TASK_QUEUE` | `pipeline` | |
| `PIPELINE_DEVICE` | _(auto)_ | `cpu` / `mps` / `cuda` override |

## Migrations (Alembic, SQL-first)

```sh
uv run alembic revision -m "add embedding tables"   # then write raw DDL in upgrade()/downgrade()
uv run alembic upgrade head        # apply
uv run alembic downgrade -1        # roll back one
uv run alembic current             # what's applied
uv run alembic history             # full history
```

## Data bootstrap (forthcoming slices)

- The MusicBrainz JSON artist dump is parked at
  `~/g/db-backups/musicbrainz-json-artist-20260606.tar.xz`.
- Import → per-artist `IngestArtistWorkflow` runs land in the next slices; this
  doc gets a "bootstrap" section then.
