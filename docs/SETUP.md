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

## Infrastructure prerequisites

The only non-`uv` installs. The box (Ubuntu) needs all four; the Mac needs the
first three (no GPU). Everything else comes from `uv.lock`.

### 1. uv — both

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"        # or restart the shell
uv --version
```

### 2. Docker — both (only runs the Postgres container)

macOS: install Docker Desktop (`brew install --cask docker`) and launch it.

Ubuntu (official apt repo — [docs](https://docs.docker.com/engine/install/ubuntu/)):

```sh
sudo apt-get update && sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"      # then log out/in so `docker` works without sudo
docker compose version
```

(quick alternative: `curl -fsSL https://get.docker.com | sh`)

### 3. Temporal CLI — both

macOS:

```sh
brew install temporal
```

Linux (official CDN binary — the archive contains just `LICENSE` + the `temporal`
binary; use `arch=arm64` on ARM):

```sh
curl -fsSL "https://temporal.download/cli/archive/latest?platform=linux&arch=amd64" -o /tmp/temporal.tar.gz
tar -xzf /tmp/temporal.tar.gz -C /tmp temporal
sudo install /tmp/temporal /usr/local/bin/temporal
temporal --version
```

(simpler alternative on Ubuntu: `sudo snap install temporal`.)
Docs: [Set up the Temporal CLI](https://docs.temporal.io/cli/setup-cli).

### 4. NVIDIA driver — the box only

```sh
sudo ubuntu-drivers install          # picks the recommended driver for the 4070 Ti SUPER
sudo reboot
nvidia-smi                           # verify: shows driver version, CUDA version, the GPU
```

- If Secure Boot prompts for **MOK enrollment** on reboot, complete it or the
  kernel module won't load.
- **No CUDA *toolkit* and no NVIDIA Container Toolkit are needed:** the pinned
  torch wheel bundles the CUDA runtime, and Docker here only runs Postgres (CPU)
  — all GPU work is host-side via uv.
- Docs: [Ubuntu NVIDIA driver guide](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/).

With the four in place: `WITH_MUQ=1 ./scripts/bootstrap.sh` (box) or
`./scripts/bootstrap.sh` (Mac).

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
| `uv run poe fetch-clips` | download labeled benchmark clips from Deezer (`seeds/benchmark-artists.txt` → `clips/`) |

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

## Running the benchmarks (`src/pipeline/bench/`)

The harness compares audio embedders on **throughput** (O4) and **embedding
quality** (O1: same-artist clips should cluster tighter than cross-artist).

**Step 1 — fetch a labeled clip set (no manual prep).** The fetcher resolves
each artist *name* to a Deezer **artist entity** and downloads ~12 of that
entity's own top-track preview MP3s — provably the named artist (kept only when
that artist is the track's `artist.id` **and** a `role=="Main"` contributor; an
unresolvable/ambiguous name is skipped, never guessed):

```sh
uv run poe fetch-clips                      # uses seeds/benchmark-artists.txt
# or an ad-hoc list:
uv run python -m pipeline.bench.fetch_clips "Aphex Twin" "Burial" -n 12 --out clips
```

This writes the `clips/<artist>/<trackid>.mp3` layout Step 2 reads, plus a
per-artist `manifest.json` (resolved id + per-clip provenance) for auditability.
`seeds/benchmark-artists.txt` is 15 deep-catalog acts, each verified to return
≥12 qualifying tracks — pull ≥10–12/artist for a stable intra-artist cosine.
(Deezer previews are a fixed 30s, 128 kbps; `/top` is the artist's most-popular
tracks so they cluster tightly — ideal for the same-vs-cross separation metric.)

**Step 2 — run it** (first run downloads weights: CLAP ~600 MB, MERT ~400 MB,
MuQ ~2 GB):

```sh
uv run --group models --group muq python -m pipeline.bench --clips clips/
```

Uses CUDA automatically on the box; force with `PIPELINE_DEVICE=cuda|cpu`.

**Step 3 — read the table:**

```
model            clips/s   ms/clip   intra   inter     sep    p@1
laion-clap-music    ...
mert-v1-95M         ...
muq-mulan-large     ...
```

- **O4 throughput** — `clips/s` / `ms/clip`. This is the cost number that decides
  verify-then-clap vs clap-then-verify (ADR-015 O4).
- **O1 quality** — `sep` (`intra − inter`; higher = same-artist clusters tighter)
  and `p@1` (fraction whose nearest neighbour is the same artist). Higher is
  better. Pick the best `sep`/`p@1` at acceptable throughput; joint audio-text
  models (CLAP, MuQ) also give tags for free — the tie-breaker.

(`uv run poe bench` with no args runs a mock demo — shows the harness without any
models installed.)

## Data bootstrap (MusicBrainz)

The corpus bootstraps from the MusicBrainz **Postgres fullexport** (NOT the JSON
dumps — those exclude URL relationships entirely). Two archives are needed:

- `mbdump.tar.bz2` (~7 GB) — core tables: artist, artist_alias, url,
  l_artist_url, link, link_type.
- `mbdump-derived.tar.bz2` (~0.5 GB) — user-generated tables: **artist_tag and
  tag live here**, not in the core dump.

Steps (the periodic refresh repeats these with a newer dump):

```bash
# 1. Find the latest export (published Wed + Sat)
curl -s https://data.metabrainz.org/pub/musicbrainz/data/fullexport/LATEST
# 2. Download both archives + MD5SUMS into ~/g/db-backups/ and verify md5sum
# 3. Extract the 10 tables (single pass per archive)
tar -xjf mbdump.tar.bz2 -C mbdump-extract \
  mbdump/artist mbdump/artist_alias mbdump/url mbdump/l_artist_url \
  mbdump/link mbdump/link_type
tar -xjf mbdump.tar.bz2 -C mbdump-extract mbdump/genre mbdump/genre_alias  # tag vocabulary
tar -xjf mbdump-derived.tar.bz2 -C mbdump-extract mbdump/artist_tag mbdump/tag
# 4. Load + derive Tier-A identities (truncate-and-reload; ~3 min)
uv run poe mb-bootstrap --dir ~/g/db-backups/mbdump-extract/mbdump
```

The loader fails fast if a table's column count drifts from the verified layout
(see `EXPECTED_COLS` in `pipeline/mb_bootstrap.py`). 20260606 export yields
~543k artists / ~1.03M Tier-A platform identities. Known coverage gap: YouTube
`/user/...` and `/@handle` url-rels (~74k) are skipped until the YouTube slice
adds channel-id resolution.
