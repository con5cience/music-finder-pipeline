"""Per-artist ingest workflow: the audio-source cascade (ADR-017 §2).

One run per ARTIST (not per identity): walk audio-role identities in
EMBED_PRIORITY order, discover each pending source on its rate-capped queue,
record terminal scan verdicts, stop early when a source meets its floor, then
choose the winner (floor-met by priority, else best floor-ratio) and embed
from exactly that source. Tier-B/C binding and its review park return in the
B-tier slice; today's identities are all Tier-A (MB url-rels).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from pipeline import activities
    from pipeline.queues import DISCOVERY_ACTIVITIES, EMBED_FLOORS, GPU_QUEUE, queue_for

_ACTIVITY_TIMEOUT = timedelta(seconds=30)
_DISCOVERY_TIMEOUT = timedelta(minutes=5)  # platform IO behind a rate-capped queue
# First embed on a fresh worker loads model weights (~30s) before downloading
# ~12 previews + GPU inference; 30s would timeout-loop the weight load forever.
_EMBED_TIMEOUT = timedelta(minutes=10)
# Bounded: the calibration run proved unlimited default retries turn a
# persistent failure (expired signed URLs → 403) into a forever-stall.
_IO_RETRY = RetryPolicy(maximum_attempts=5)


@dataclass
class IngestArtistInput:
    artist_id: str


def _meets_floor(platform: str, total_yield: int) -> bool:
    floor = EMBED_FLOORS.get(platform)
    return floor is not None and total_yield >= floor


@workflow.defn
class IngestArtistWorkflow:
    @workflow.run
    async def run(self, inp: IngestArtistInput) -> dict:
        plan = await workflow.execute_activity(
            activities.cascade_plan, inp.artist_id, start_to_close_timeout=_ACTIVITY_TIMEOUT
        )
        if not plan["pending"] and not plan["has_audio_identities"]:
            return {"status": "unbound"}  # no audio-role identity; crawler binding is design-gated

        scanned = 0
        for platform, _platform_id in plan["pending"]:
            discovery = DISCOVERY_ACTIVITIES.get(platform)
            if discovery is None:
                continue  # no ingestion flow built for this source yet — stays pending
            await workflow.execute_activity(
                discovery,
                inp.artist_id,
                task_queue=queue_for(platform),
                start_to_close_timeout=_DISCOVERY_TIMEOUT,
                retry_policy=_IO_RETRY,
            )
            total_yield = await workflow.execute_activity(
                activities.record_scan,
                args=[inp.artist_id, platform, _platform_id],
                start_to_close_timeout=_ACTIVITY_TIMEOUT,
            )
            scanned += 1
            if _meets_floor(platform, total_yield):
                break  # cascade satisfied — don't burn budget on lower sources

        choice = await workflow.execute_activity(
            activities.choose_embed_source, inp.artist_id, start_to_close_timeout=_ACTIVITY_TIMEOUT
        )
        if choice is None:
            return {"status": "no_signal", "scanned": scanned}

        embedded = await workflow.execute_activity(
            activities.embed_artist,
            args=[inp.artist_id, choice["source"], choice["ratio"]],
            task_queue=GPU_QUEUE,
            start_to_close_timeout=_EMBED_TIMEOUT,
            retry_policy=_IO_RETRY,
        )
        return {
            "status": "embedded",
            "source": choice["source"],
            "ratio": choice["ratio"],
            "scanned": scanned,
            "embedded": embedded,
        }
