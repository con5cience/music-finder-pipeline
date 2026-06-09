"""IngestArtistWorkflow orchestration + the human-review signal gate.

Uses Temporal's time-skipping test environment with mocked activities. Skips
cleanly if the test server can't be fetched (offline), like the DB tests.
"""

from __future__ import annotations

import uuid

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from pipeline.workflows import IngestArtistInput, IngestArtistWorkflow


@activity.defn(name="classify_page")
async def mock_classify(platform_id: str) -> str:
    return "artist"


def _mock_bind(tier: str):
    @activity.defn(name="bind_source")
    async def bind(artist_id: str, platform: str, platform_id: str) -> dict:
        return {"tier": tier, "track_count": 3}

    return bind


@activity.defn(name="embed_artist")
async def mock_embed(artist_id: str) -> int:
    return 3


async def _env() -> WorkflowEnvironment:
    try:
        return await WorkflowEnvironment.start_time_skipping()
    except Exception as exc:  # noqa: BLE001 — any failure to fetch/boot the test server → skip
        pytest.skip(f"Temporal test server unavailable: {exc}")


async def _run(env: WorkflowEnvironment, bind, signal: str | None = None) -> dict:
    tq = "test-" + uuid.uuid4().hex
    async with Worker(
        env.client,
        task_queue=tq,
        workflows=[IngestArtistWorkflow],
        activities=[mock_classify, bind, mock_embed],
    ):
        handle = await env.client.start_workflow(
            IngestArtistWorkflow.run,
            IngestArtistInput("a1", "soundcloud", "111"),
            id="wf-" + uuid.uuid4().hex,
            task_queue=tq,
        )
        if signal is not None:
            await handle.signal(IngestArtistWorkflow.submit_review_decision, signal)
        return await handle.result()


async def test_tier_a_auto_embeds():
    env = await _env()
    async with env:
        res = await _run(env, _mock_bind("A"))
    assert res["status"] == "embedded"
    assert res["tier"] == "A"
    assert res["embedded"] == 3


async def test_tier_c_blocks_until_approved():
    env = await _env()
    async with env:
        res = await _run(env, _mock_bind("C"), signal="approved")
    assert res["status"] == "embedded"


async def test_tier_c_rejected_by_review():
    env = await _env()
    async with env:
        res = await _run(env, _mock_bind("C"), signal="rejected")
    assert res["status"] == "rejected_by_review"
