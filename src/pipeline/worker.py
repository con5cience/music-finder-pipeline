"""Temporal worker entrypoint — hosts IngestArtistWorkflow + activities.

Run locally against `temporal server start-dev`:  uv run python -m pipeline.worker
On the box the same command picks up CUDA automatically (see device.select_device).
"""

from __future__ import annotations

import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from pipeline import activities
from pipeline.config import Settings
from pipeline.workflows import IngestArtistWorkflow


async def main() -> None:
    settings = Settings()
    client = await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)
    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[IngestArtistWorkflow],
        activities=[activities.classify_page, activities.bind_source, activities.embed_artist],
    )
    print(f"worker up — queue={settings.temporal_task_queue} device={settings.effective_device}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
