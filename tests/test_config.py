"""Settings load Mac-safe defaults and honor PIPELINE_-prefixed env overrides."""

from __future__ import annotations

from pipeline.config import Settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("PIPELINE_DEVICE", raising=False)
    monkeypatch.delenv("PIPELINE_DATABASE_URL", raising=False)
    s = Settings()
    assert s.temporal_namespace == "default"
    assert s.temporal_task_queue == "pipeline"
    assert s.database_url.startswith("postgresql://")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("PIPELINE_DATABASE_URL", "postgresql://u@h:5440/db")
    monkeypatch.setenv("PIPELINE_DEVICE", "cuda")
    s = Settings()
    assert s.database_url == "postgresql://u@h:5440/db"
    assert s.effective_device == "cuda"
