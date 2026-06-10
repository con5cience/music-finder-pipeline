"""Per-platform task queues + rate budgets (ADR-017 §4) + descriptor coherence.

Review finding: five hand-synced registries across four files meant a new
platform could be half-registered silently. PLATFORMS is now the single
source; these tests assert every derived view + the worker/activity wiring
stay coherent BY CONSTRUCTION."""

from __future__ import annotations

import importlib

from pipeline.queues import (
    DISCOVERY_ACTIVITIES,
    EMBED_FLOORS,
    EMBED_PRIORITY,
    GPU_QUEUE,
    PLATFORM_QUEUES,
    PLATFORMS,
    REFRESHERS,
    WORKFLOW_QUEUE,
    queue_for,
)


def test_locked_platforms_have_queues():
    assert {"deezer", "bandcamp", "soundcloud", "tidal", "youtube", "musicbrainz"} <= set(PLATFORM_QUEUES)


def test_budgets_match_adr017():
    # Incident-derived ceilings: violating these caused real bans/cooldowns.
    assert PLATFORM_QUEUES["musicbrainz"].max_per_second == 1.0  # MB TOS, never deviate
    assert PLATFORM_QUEUES["tidal"].max_per_second == 0.2
    assert PLATFORM_QUEUES["youtube"].max_per_second == 0.1
    # start-conservative budgets stay at or under the sibling's observed ceilings
    assert PLATFORM_QUEUES["deezer"].max_per_second <= 50
    assert PLATFORM_QUEUES["bandcamp"].max_per_second <= 50
    assert PLATFORM_QUEUES["soundcloud"].max_per_second <= 50


def test_queue_names_are_namespaced():
    for platform, cfg in PLATFORM_QUEUES.items():
        assert cfg.name == f"{platform}-io"
        assert cfg.name != WORKFLOW_QUEUE
        assert cfg.name != GPU_QUEUE


def test_gpu_queue_is_distinct():
    assert GPU_QUEUE not in (WORKFLOW_QUEUE, *(c.name for c in PLATFORM_QUEUES.values()))


def test_queue_for_lookup():
    assert queue_for("deezer") == "deezer-io"


# ---- descriptor coherence: adding a platform CANNOT half-register it -------


def test_every_discovery_activity_exists_and_is_registered():
    from pipeline import activities
    from pipeline.worker import PLATFORM_ACTIVITIES

    for platform, activity_name in DISCOVERY_ACTIVITIES.items():
        fn = getattr(activities, activity_name, None)
        assert fn is not None, f"{platform}: activity {activity_name} missing from activities.py"
        assert fn in PLATFORM_ACTIVITIES[platform], (
            f"{platform}: {activity_name} not registered on its IO queue worker"
        )


def test_every_refresher_resolves():
    for platform, dotted in REFRESHERS.items():
        module, _, fn_name = dotted.partition(":")
        fn = getattr(importlib.import_module(module), fn_name, None)
        assert callable(fn), f"{platform}: refresher {dotted} does not resolve"


def test_audio_platforms_have_coherent_descriptors():
    for p in PLATFORMS.values():
        if p.audio_priority is not None:
            assert p.name in EMBED_FLOORS
            assert p.name in EMBED_PRIORITY
            # any audio platform WITH a discovery flow must also refresh
            # (signed URLs rot on every platform we've met so far)
            if p.discovery_activity:
                assert p.refresher, f"{p.name}: discovery without a refresher"
        else:
            assert p.name not in EMBED_PRIORITY  # playback/identity assets never cascade


def test_priority_order_is_dense_and_unique():
    prios = [p.audio_priority for p in PLATFORMS.values() if p.audio_priority is not None]
    assert sorted(prios) == list(range(1, len(prios) + 1))
