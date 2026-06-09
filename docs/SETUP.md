# Setup & runbook

How to stand up the pipeline stack — Postgres (the local "factory" DB), Temporal,
and the worker — on the Mac for dev and on the CUDA box for real runs.

**Temporal is self-hosted** — the CLI **dev server** locally (`temporal server
start-dev`), and a single-node self-hosted server on the box — **not Temporal
Cloud**. The worker connects to `localhost:7233`; tests use the in-process
time-skipping server. No SaaS, no API keys.

**One command** (after the system prerequisites below): `./scripts/bootstrap.sh`
— `uv sync` → Postgres → migrate. On the box add MuQ: `WITH_MUQ=1
./scripts/bootstrap.sh`. Everything Python is reproducible from `uv.lock`; only
the prereqs (uv, Docker, Temporal CLI, NVIDIA driver) are system installs.

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
| `./scripts/bootstrap.sh` | full: `uv sync` → Postgres → migrate (`WITH_MUQ=1` on the box) |
| `uv run poe setup` | post-sync: `db-up` then `migrate` |
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
| `uv run poe bench` | `python -m pipeline.bench` (model-eval + clap-cost demo) |

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
# NVIDIA driver only — the pinned CUDA torch wheel bundles the CUDA runtime, so
# NO separate CUDA toolkit is required.

git clone <repo> && cd music-finder-pipeline
WITH_MUQ=1 ./scripts/bootstrap.sh        # uv sync (+ muq) → Postgres → migrate
temporal server start-dev --ip 0.0.0.0   # self-hosted; or run under systemd
uv run python -m pipeline.worker         # prints device=cuda automatically
```

GPU lights up by default — `device.select_device()` returns `cuda` when torch
sees the card. Force with `PIPELINE_DEVICE=cpu|mps|cuda`. torch is **pinned per
platform** in `pyproject.toml`/`uv.lock`: the CUDA wheel (cu128) on Linux, the
default CPU/MPS build on the Mac — `uv sync --group models` is deterministic on
both. (The `muq` group is heavier; install it on the box via the bootstrap above.)

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

## Benchmarks (`src/pipeline/bench/`)

Model-agnostic harness comparing audio embedders on **throughput** (O4: clips/s,
ms/clip) and **quality** (O1: same-artist clips cluster tighter than cross-artist
— `separation` and `precision@1`). `uv run poe bench` runs a mock demo; a real
model is added by implementing the `Embedder` protocol (`name` + `embed(clips)`)
and registering it in `bench/__main__.py`. The candidate model shortlist
(LAION-CLAP variants vs MERT/MusiCNN/…) is a tomorrow decision, researched before
the box run.

## Data bootstrap (forthcoming slices)

- The MusicBrainz JSON artist dump is parked at
  `~/g/db-backups/musicbrainz-json-artist-20260606.tar.xz`.
- Import → per-artist `IngestArtistWorkflow` runs land in the next slices; this
  doc gets a "bootstrap" section then.
