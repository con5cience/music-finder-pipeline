"""Pipeline settings — env-driven (prefix PIPELINE_), with Mac-safe local defaults.

The box overrides these via env; locally they point at the dev compose. Device
defaults to auto-detection (see device.select_device) unless pinned.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from pipeline.device import select_device


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PIPELINE_", env_file=".env", extra="ignore")

    # Local-first Postgres (the "factory" DB). Cloud publish is a later phase.
    database_url: str = "postgresql://pipeline:pipeline@localhost:5440/pipeline"

    # Temporal dev server defaults.
    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "default"
    temporal_task_queue: str = "pipeline"

    # None → auto-detect (cuda/mps/cpu); pin via PIPELINE_DEVICE.
    device: str | None = None

    # Analysis head. None → registry default (MuQ audio-only, ADR-016). Swap via
    # PIPELINE_EMBEDDING_MODEL (e.g. "musicfm-msd" for the MIT/commercial-safe model).
    # Whatever runs gets stamped on stored embeddings so a swap is a clean re-embed.
    embedding_model: str | None = None

    @property
    def effective_device(self) -> str:
        return self.device or select_device()
