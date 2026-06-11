# Pipeline worker image — one image, two roles (--role io | gpu).
# Models (torch cu128, MuQ, MuLan) install in the image but import lazily:
# the io role never loads them at runtime; the gpu role does, with weights
# cached on the hf-cache volume so restarts don't re-download ~4GB.
FROM python:3.12-slim

# libsndfile: decode (mp3 support needs >=1.1); libchromaprint: fingerprint
# head (loaded via ctypes); ca-certs for CDN fetches; ffmpeg: yt-dlp's
# m4a→wav postprocessor (libsndfile cannot decode m4a — 2026-06-11 lesson).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 libchromaprint1 ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
# Locked, reproducible env — same resolution the suite runs against.
RUN uv sync --frozen --group models --group muq --no-install-project

COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./
RUN uv sync --frozen --group models --group muq

ENTRYPOINT [".venv/bin/python", "-m", "pipeline.worker"]
CMD ["--role", "io"]
