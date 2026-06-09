"""Per-artist ingest workflow.

Durable orchestration: classify the page → bind a source under a verification
tier → (Tier C blocks on a human-review SIGNAL) → embed. The human-in-the-
loop gate is the reason for Temporal: a Tier-C binding parks the workflow,
crash-safe, for as long as it takes a reviewer to decide.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from pipeline import activities
    from pipeline.queues import DISCOVERY_ACTIVITIES, GPU_QUEUE, queue_for

_ACTIVITY_TIMEOUT = timedelta(seconds=30)
_DISCOVERY_TIMEOUT = timedelta(minutes=5)  # platform IO behind a rate-capped queue
# First embed on a fresh worker loads model weights (~30s) before downloading
# ~12 previews + GPU inference; 30s would timeout-loop the weight load forever.
_EMBED_TIMEOUT = timedelta(minutes=10)


@dataclass
class IngestArtistInput:
    artist_id: str
    platform: str
    platform_id: str


@workflow.defn
class IngestArtistWorkflow:
    def __init__(self) -> None:
        self._review_decision: str | None = None

    @workflow.signal
    def submit_review_decision(self, decision: str) -> None:
        """Reviewer's verdict on a Tier-C binding: 'approved' | 'rejected'."""
        self._review_decision = decision

    @workflow.query
    def status(self) -> str:
        return self._review_decision or "running"

    @workflow.run
    async def run(self, inp: IngestArtistInput) -> dict:
        page_type = await workflow.execute_activity(
            activities.classify_page,
            args=[inp.platform, inp.platform_id],
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
        )
        binding = await workflow.execute_activity(
            activities.bind_source,
            args=[inp.artist_id, inp.platform, inp.platform_id],
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
        )
        if binding is None:
            # No authoritative link and no search path yet (B-tier slice).
            return {"status": "unbound", "page_type": page_type}

        if binding["tier"] == "C":
            # Park until a reviewer signals; Temporal persists this wait.
            await workflow.wait_condition(lambda: self._review_decision is not None)
            if self._review_decision != "approved":
                return {"status": "rejected_by_review", "page_type": page_type}

        discovered = 0
        discovery = DISCOVERY_ACTIVITIES.get(inp.platform)
        if discovery is not None:
            discovered = await workflow.execute_activity(
                discovery,
                inp.artist_id,
                task_queue=queue_for(inp.platform),
                start_to_close_timeout=_DISCOVERY_TIMEOUT,
            )

        embedded = await workflow.execute_activity(
            activities.embed_artist,
            inp.artist_id,
            task_queue=GPU_QUEUE,
            start_to_close_timeout=_EMBED_TIMEOUT,
        )
        return {
            "status": "embedded",
            "tier": binding["tier"],
            "page_type": page_type,
            "discovered": discovered,
            "embedded": embedded,
        }
