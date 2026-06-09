"""Per-platform task queues + rate budgets (ADR-017 §4)."""

from __future__ import annotations

from pipeline.queues import PLATFORM_QUEUES, WORKFLOW_QUEUE, queue_for


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


def test_queue_for_lookup():
    assert queue_for("deezer") == "deezer-io"
