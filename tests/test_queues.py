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
    assert PLATFORM_QUEUES["bandcamp"].max_per_second <= 5.0  # proxied; 429s monitored
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
            # any NON-EXPERIMENTAL audio platform with a discovery flow must
            # also refresh (signed URLs rot everywhere we fetch audio).
            # Experimental (floor None) stores audio_url=NULL — nothing rots.
            if p.discovery_activity and p.floor is not None:
                assert p.refresher or p.stable_audio_urls, (
                f"{p.name}: floored platform needs a refresher or a stable url scheme"
            )
        else:
            assert p.name not in EMBED_PRIORITY  # playback/identity assets never cascade


def test_priority_order_is_dense_and_unique():
    prios = [p.audio_priority for p in PLATFORMS.values() if p.audio_priority is not None]
    assert sorted(prios) == list(range(1, len(prios) + 1))


def test_io_concurrency_caps_are_sane():
    # Mined from the old fleet: concurrency is orthogonal to rate budgets.
    for p in PLATFORMS.values():
        assert 1 <= p.io_concurrency <= 16
    assert PLATFORMS["youtube"].io_concurrency == 1  # most fragile platform


def test_every_workflow_dispatched_activity_is_registered():
    # The unregistered-activity hang, 3rd occurrence (embed_artist_staged
    # parked 981 staged artists): every activities.X the workflow source
    # dispatches must be registered on SOME worker queue.
    import re
    from pathlib import Path

    import pipeline.workflows as wf
    from pipeline.worker import GPU_ACTIVITIES, PLATFORM_ACTIVITIES, PREP_ACTIVITIES

    src = Path(wf.__file__).read_text()
    dispatched = set(re.findall(r"activities\.(\w+)", src))
    dispatched |= set(DISCOVERY_ACTIVITIES.values())
    registered = {f.__name__ for f in GPU_ACTIVITIES + PREP_ACTIVITIES}
    registered |= {f.__name__ for fns in PLATFORM_ACTIVITIES.values() for f in fns}
    # pipeline-queue activities (workflow + DB) are registered inline:
    registered |= {"cascade_plan", "record_scan", "choose_embed_source"}
    missing = dispatched - registered
    assert not missing, f"workflow dispatches unregistered activities: {missing}"
