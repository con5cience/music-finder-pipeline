# music-finder-pipeline

Audio-first corpus pipeline — the data **factory** for music-finder, per
[ADR-015](../music-finder/docs/adr/ADR-015-audio-first-rebuild.md).

It bootstraps from a MusicBrainz dump, sources platform audio under a
source-correctness law (no blind name-matching), CLAP-embeds it on the GPU box,
and publishes serve-ready records to the cloud app. Identity is keyed on the
immutable platform numeric ID; quality gates fail closed.

This repo is **Python + Temporal**, separate from the Node/TS serving app
(`music-finder`). The clean boundary is the database.

## Layout

```
src/pipeline/      # the pipeline package
  device.py        # cuda/mps/cpu selection (PIPELINE_DEVICE override)
  config.py        # env-driven settings (PIPELINE_ prefix)
tests/             # pytest
compose.yaml       # local Postgres 16 (port 5440)
```

## Dev

```sh
uv sync                 # create venv (managed Python 3.12) + install deps
uv run pytest -q        # tests (run on the Mac; GPU steps are mocked / CPU-fallback)
uv run ruff check .     # lint
docker compose up -d db # local Postgres for schema/integration work
```

Common actions are `poe` tasks (the "npm scripts" of this repo): `uv run poe
test | lint | check | db-up | migrate | worker | temporal | …`. Full setup +
box runbook in [`docs/SETUP.md`](docs/SETUP.md).

Device model: code is device-agnostic — mocked in unit tests, CPU/MPS fallback
on the Mac for integration/E2E, CUDA by default on the box. Force with
`PIPELINE_DEVICE=cpu|mps|cuda`.
