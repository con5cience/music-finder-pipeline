"""start_ingest_workflow: the single owner of how an ingest run is launched.

Centralizing all three producers (mass seeder, calibration seed, homonym
front-run) onto one helper guarantees every run carries an execution_timeout —
without it a run wedged on a stuck activity stays Running forever, holding a
window slot so the seeder's low-water gate never drains and pinning its history
past the 24h retention (the 4-day zombie runs seen 2026-06-15).
"""

from __future__ import annotations

from datetime import timedelta

from temporalio.exceptions import TemporalError

from pipeline.seed_ingest import (
    INGEST_EXECUTION_TIMEOUT,
    start_ingest_workflow,
    workflow_id,
)


class _Settings:
    temporal_task_queue = "pipeline"


class _FakeClient:
    def __init__(self, raise_exc: Exception | None = None) -> None:
        self.raise_exc = raise_exc
        self.calls: list[tuple[tuple, dict]] = []

    async def start_workflow(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls.append((args, kwargs))
        if self.raise_exc:
            raise self.raise_exc


async def test_start_ingest_passes_execution_timeout_and_canonical_args():
    client = _FakeClient()
    assert await start_ingest_workflow(client, "abc", _Settings()) is True
    _, kwargs = client.calls[0]
    assert kwargs["execution_timeout"] == INGEST_EXECUTION_TIMEOUT
    assert kwargs["id"] == workflow_id("abc")
    assert kwargs["task_queue"] == "pipeline"


async def test_start_ingest_skips_already_started():
    # already-started surfaces as a TemporalError subclass → idempotent skip
    client = _FakeClient(raise_exc=TemporalError("already started"))
    assert await start_ingest_workflow(client, "abc", _Settings()) is False


async def test_execution_timeout_bounded_and_under_retention():
    # finite (kills zombies) but well under the 24h namespace retention so
    # terminal runs still self-purge
    assert timedelta(0) < INGEST_EXECUTION_TIMEOUT < timedelta(hours=24)
