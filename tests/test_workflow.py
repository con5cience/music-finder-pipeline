"""Cascade IngestArtistWorkflow orchestration (mocked activities).

Time-skipping Temporal env; skips cleanly when the test server is unavailable.
Covers: floor-met early exit, thin-source fallback embed, unbound, no_signal,
and the no-flow-yet platform skip.
"""

from __future__ import annotations

import uuid

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from pipeline.workflows import IngestArtistInput, IngestArtistWorkflow


def _mock_plan(pending, has_audio=True):
    @activity.defn(name="cascade_plan")
    async def plan(artist_id: str) -> dict:
        return {"has_audio_identities": has_audio, "pending": pending}

    return plan


def _mock_record_scan(total_yield: int, calls: list | None = None):
    @activity.defn(name="record_scan")
    async def record(artist_id: str, platform: str, platform_id: str) -> int:
        if calls is not None:
            calls.append(platform)
        return total_yield

    return record


def _mock_choose(result):
    @activity.defn(name="choose_embed_source")
    async def choose(artist_id: str) -> dict | None:
        return result

    return choose


@activity.defn(name="embed_artist")
async def mock_embed(artist_id: str, source: str | None = None, ratio: float | None = None) -> int:
    return 12


@activity.defn(name="discover_deezer_tracks")
async def mock_discover(artist_id: str) -> int:
    return 12


async def _env() -> WorkflowEnvironment:
    try:
        return await WorkflowEnvironment.start_time_skipping()
    except Exception as exc:  # noqa: BLE001 — any failure to fetch/boot the test server → skip
        pytest.skip(f"Temporal test server unavailable: {exc}")


async def _run(env: WorkflowEnvironment, plan, record, choose) -> dict:
    tq = "test-" + uuid.uuid4().hex
    async with (
        Worker(env.client, task_queue=tq, workflows=[IngestArtistWorkflow],
               activities=[plan, record, choose, mock_embed]),
        Worker(env.client, task_queue="deezer-io", activities=[mock_discover]),
        Worker(env.client, task_queue="gpu", activities=[mock_embed]),
    ):
        return await env.client.execute_workflow(
            IngestArtistWorkflow.run,
            IngestArtistInput("a1"),
            id="wf-" + uuid.uuid4().hex,
            task_queue=tq,
        )


async def test_floor_met_embeds_from_winner():
    env = await _env()
    async with env:
        res = await _run(
            env,
            _mock_plan([["deezer", "d1"]]),
            _mock_record_scan(12),  # >= deezer floor 10
            _mock_choose({"source": "deezer", "ratio": 1.2}),
        )
    assert res["status"] == "embedded"
    assert (res["source"], res["scanned"], res["embedded"]) == ("deezer", 1, 12)


async def test_thin_source_still_embeds_with_ratio():
    env = await _env()
    async with env:
        res = await _run(
            env,
            _mock_plan([["deezer", "d1"]]),
            _mock_record_scan(2),  # under floor — cascade exhausts, thin fallback
            _mock_choose({"source": "deezer", "ratio": 0.2}),
        )
    assert res["status"] == "embedded"
    assert res["ratio"] == 0.2


async def test_no_audio_identities_is_unbound():
    env = await _env()
    async with env:
        res = await _run(env, _mock_plan([], has_audio=False), _mock_record_scan(0), _mock_choose(None))
    assert res == {"status": "unbound"}


async def test_nothing_usable_is_no_signal():
    env = await _env()
    async with env:
        res = await _run(env, _mock_plan([["deezer", "d1"]]), _mock_record_scan(0), _mock_choose(None))
    assert res["status"] == "no_signal"


async def test_platform_without_flow_is_skipped_not_fatal():
    # soundcloud has no discovery activity yet: identity stays pending, the
    # cascade moves on, and choose still runs (an earlier-scanned source or
    # nothing may win). NB: must use a genuinely flow-less platform — this
    # test hung for 30 minutes when bandcamp gained a flow and the dispatch
    # went to a queue no test worker polls.
    env = await _env()
    calls: list = []
    async with env:
        res = await _run(
            env,
            _mock_plan([["soundcloud", "s1"], ["deezer", "d1"]]),
            _mock_record_scan(12, calls),
            _mock_choose({"source": "deezer", "ratio": 1.2}),
        )
    assert res["status"] == "embedded"
    assert calls == ["deezer"]  # soundcloud never scanned (no flow), deezer was
