"""Temporal worker entrypoint — workflow worker + per-platform IO workers.

The main queue hosts workflows + DB/GPU activities. Each platform gets its own
task queue whose dispatch rate is enforced SERVER-SIDE
(max_task_queue_activities_per_second, ADR-017 §4) — platform IO activities
register there as the discovery slices land.

Run locally against `temporal server start-dev`:  uv run python -m pipeline.worker
On the box the same command picks up CUDA automatically (see device.select_device).
"""

from __future__ import annotations

import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from pipeline import activities
from pipeline.config import Settings
from pipeline.queues import PLATFORM_QUEUES
from pipeline.workflows import IngestArtistWorkflow

# platform → IO activities (filled by the per-platform discovery slices).
PLATFORM_ACTIVITIES: dict[str, list] = {platform: [] for platform in PLATFORM_QUEUES}
PLATFORM_ACTIVITIES["deezer"] = [activities.discover_deezer_tracks]


def build_workers(client: Client, settings: Settings) -> list[Worker]:
    workers = [
        Worker(
            client,
            task_queue=settings.temporal_task_queue,
            workflows=[IngestArtistWorkflow],
            activities=[activities.classify_page, activities.bind_source, activities.embed_artist],
        )
    ]
    for platform, cfg in PLATFORM_QUEUES.items():
        acts = PLATFORM_ACTIVITIES[platform]
        if not acts:
            continue  # no IO activities for this platform yet
        workers.append(
            Worker(
                client,
                task_queue=cfg.name,
                activities=acts,
                max_task_queue_activities_per_second=cfg.max_per_second,
            )
        )
    return workers


async def main() -> None:
    settings = Settings()
    client = await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)
    workers = build_workers(client, settings)
    queues = ", ".join(w.config()["task_queue"] for w in workers)
    print(f"workers up — queues=[{queues}] device={settings.effective_device}")
    await asyncio.gather(*(w.run() for w in workers))


if __name__ == "__main__":
    asyncio.run(main())
