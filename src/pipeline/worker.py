"""Temporal worker entrypoint — workflow worker + per-platform IO workers.

Roles (--role, for containerized split): `io` runs the workflow + DB + IO
queues (CPU image); `gpu` runs only the embed queue (GPU reservation); `all`
(default) runs everything in one process — the pre-compose layout. Every
role heartbeats into worker_heartbeat so the admin Workers card reflects
fleet liveness without Redis or docker introspection.

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
from pipeline.queues import DISCOVERY_ACTIVITIES, GPU_QUEUE, PLATFORM_QUEUES, PLATFORMS
from pipeline.workflows import IngestArtistWorkflow

# DERIVED from the PLATFORMS descriptor — never hand-edit (review finding:
# parallel registries drift; test_queues asserts this wiring stays coherent).
PLATFORM_ACTIVITIES: dict[str, list] = {platform: [] for platform in PLATFORM_QUEUES}
for _platform, _activity_name in DISCOVERY_ACTIVITIES.items():
    PLATFORM_ACTIVITIES[_platform].append(getattr(activities, _activity_name))


def build_workers(client: Client, settings: Settings, role: str = "all") -> list[Worker]:
    workers: list[Worker] = []
    if role in ("all", "io"):
        workers.append(Worker(
            client,
            task_queue=settings.temporal_task_queue,
            workflows=[IngestArtistWorkflow],
            # GPU work lives ONLY on the gpu queue (concurrency-capped). The
            # legacy embed_artist registration here is gone: zero pre-cascade
            # workflows remain in flight (verified before removal), and an
            # uncapped queue must never run GPU activities (review finding).
            activities=[
                activities.cascade_plan,
                activities.record_scan,
                activities.choose_embed_source,
            ],
        ))
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
                    # orthogonal to the rate budget: bounds in-flight fetches
                    max_concurrent_activities=PLATFORMS[platform].io_concurrency,
                )
            )
    if role in ("all", "gpu"):
        import os

        # Embed wall-time is download/decode-heavy (measured: GPU idle 60-70%
        # at concurrency 2) — concurrency hides fetch latency. Env-tunable so
        # VRAM tuning needs no rebuild; peak observed 10.8GB at 2, so step via
        # 3 and watch nvidia-smi before 4.
        gpu_conc = int(os.environ.get("PIPELINE_GPU_CONCURRENCY", "3"))
        workers.append(Worker(
            client,
            task_queue=GPU_QUEUE,
            activities=[activities.embed_artist],
            max_concurrent_activities=gpu_conc,
        ))
    return workers


async def _heartbeat_loop(settings: Settings, role: str, queues: str) -> None:
    """Liveness for the admin Workers card — a row per (role, host), upserted
    every 30s. Pure DB: works identically native or containerized."""
    import socket

    import psycopg

    host = socket.gethostname()
    while True:
        try:
            with psycopg.connect(settings.database_url) as conn:
                conn.execute(
                    """
                    INSERT INTO worker_heartbeat (role, hostname, queues, last_seen)
                    VALUES (%s, %s, %s, now())
                    ON CONFLICT (role, hostname) DO UPDATE
                        SET queues = EXCLUDED.queues, last_seen = now()
                    """,
                    (role, host, queues),
                )
                conn.commit()
        except Exception:  # noqa: BLE001 — heartbeat must never kill the fleet
            pass
        await asyncio.sleep(30)


async def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("all", "io", "gpu"), default="all")
    args = ap.parse_args()
    settings = Settings()
    client = await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)
    workers = build_workers(client, settings, args.role)
    queues = ", ".join(w.config()["task_queue"] for w in workers)
    print(f"workers up — role={args.role} queues=[{queues}] device={settings.effective_device}", flush=True)
    await asyncio.gather(
        _heartbeat_loop(settings, args.role, queues),
        *(w.run() for w in workers),
    )


if __name__ == "__main__":
    asyncio.run(main())
