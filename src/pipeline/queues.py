"""Platform descriptors — THE single registry (ADR-017 §4 + consolidation).

One PlatformSource per platform; everything else (queues, floors, priority,
discovery dispatch, windowing, refresher wiring) DERIVES from this dict.
Review finding: five hand-synced registries across four files meant adding a
platform could half-register it silently — test_queues now asserts coherence.

This module stays sandbox-safe (pure data + strings): workflows import it via
passthrough; refreshers are dotted strings resolved lazily in embed_job.

Budgets are the sibling fleet's incident-derived ceilings, enforced
SERVER-SIDE via max_task_queue_activities_per_second. Floors double as
equal-signal normalizers (10 previews ≈ 3 full tracks); floor None =
experimental (scanned + recorded, never auto-embeds).
"""

from __future__ import annotations

from dataclasses import dataclass

WORKFLOW_QUEUE = "pipeline"  # workflows + fast DB activities (cascade plan/scan/choose)
# GPU work is concurrency-capped (VRAM), not rate-capped: its own queue so slow
# embeds never occupy the pipeline queue's slots (observed live: 9h ETA stall).
GPU_QUEUE = "gpu"


@dataclass(frozen=True)
class PlatformSource:
    name: str
    io_rate: float            # req/s budget for {name}-io (server-enforced)
    audio_priority: int | None = None  # cascade order; None = playback/identity only
    floor: int | None = None           # tracks; None = experimental, never auto-embeds
    windowed: bool = False             # full tracks → RMS-peak windows
    discovery_activity: str | None = None  # activity name (string: sandbox-safe dispatch)
    refresher: str | None = None           # "module:function" for expired audio URLs


PLATFORMS: dict[str, PlatformSource] = {
    # ~50/s observed ceiling (proxy-limited upstream); sustained-presence polite.
    "deezer": PlatformSource(
        "deezer", 10.0, audio_priority=1, floor=10, windowed=False,
        discovery_activity="discover_deezer_tracks",
        refresher="pipeline.sources.deezer:refresh_preview",
    ),
    # "ample headroom in practice" at 50/s; HTML scraping, be polite.
    "bandcamp": PlatformSource(
        "bandcamp", 5.0, audio_priority=2, floor=3, windowed=True,
        discovery_activity="discover_bandcamp_tracks",
        refresher="pipeline.sources.bandcamp:refresh_bandcamp",
    ),
    # OAuth headroom at 50/s; discovery flow not built yet.
    "soundcloud": PlatformSource("soundcloud", 5.0, audio_priority=3, floor=3, windowed=True),
    # Fragile scraping; url-rel-only scope; floor experimental.
    "youtube": PlatformSource("youtube", 0.1, audio_priority=4, floor=None, windowed=True),
    # Community-confirmed ≈0.2/s; playback/URL asset only — never audio.
    "tidal": PlatformSource("tidal", 0.2),
    # MB TOS: 1 req/s, never deviate. UA must carry a real contact email.
    "musicbrainz": PlatformSource("musicbrainz", 1.0),
}


# ---- derived views (the only names other modules should consume) -----------


@dataclass(frozen=True)
class QueueConfig:
    name: str
    max_per_second: float


PLATFORM_QUEUES: dict[str, QueueConfig] = {
    p.name: QueueConfig(f"{p.name}-io", p.io_rate) for p in PLATFORMS.values()
}

EMBED_PRIORITY: list[str] = [
    p.name for p in sorted(
        (p for p in PLATFORMS.values() if p.audio_priority is not None),
        key=lambda p: p.audio_priority,
    )
]

EMBED_FLOORS: dict[str, int | None] = {
    p.name: p.floor for p in PLATFORMS.values() if p.audio_priority is not None
}

WINDOWED_PLATFORMS: set[str] = {p.name for p in PLATFORMS.values() if p.windowed}

DISCOVERY_ACTIVITIES: dict[str, str] = {
    p.name: p.discovery_activity for p in PLATFORMS.values() if p.discovery_activity
}

REFRESHERS: dict[str, str] = {p.name: p.refresher for p in PLATFORMS.values() if p.refresher}


def queue_for(platform: str) -> str:
    return PLATFORM_QUEUES[platform].name
