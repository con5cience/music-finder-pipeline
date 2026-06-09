"""Per-platform Temporal task queues + rate budgets (ADR-017 §4).

One task queue per platform; the budget is enforced SERVER-SIDE via the task
queue's max_task_queue_activities_per_second, so every worker on the queue
shares one global budget — no Redis token buckets. Numbers are the sibling
fleet's incident-derived ceilings, started conservative where headroom exists.
Raising a budget is a config change here, nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

WORKFLOW_QUEUE = "pipeline"  # workflows + non-IO activities (embed, db)


@dataclass(frozen=True)
class QueueConfig:
    name: str
    max_per_second: float


PLATFORM_QUEUES: dict[str, QueueConfig] = {
    # ~50/s observed ceiling (proxy-limited); start at 10/s — we are a sustained
    # presence at 1M scale, not a burst.
    "deezer": QueueConfig("deezer-io", 10.0),
    # "ample headroom in practice" at 50/s; start 5/s (HTML scraping, be polite).
    "bandcamp": QueueConfig("bandcamp-io", 5.0),
    # OAuth headroom at 50/s; start 5/s. oEmbed is a separate envelope upstream.
    "soundcloud": QueueConfig("soundcloud-io", 5.0),
    # Community-confirmed bucket ≈ 0.2/s sustained; 10× over caused 22h cooldowns.
    "tidal": QueueConfig("tidal-io", 0.2),
    # Fragile scraping; url-rel-only scope. 0.1/s.
    "youtube": QueueConfig("youtube-io", 0.1),
    # MB TOS: 1 req/s, never deviate. UA must carry a real contact email.
    "musicbrainz": QueueConfig("musicbrainz-io", 1.0),
}


def queue_for(platform: str) -> str:
    return PLATFORM_QUEUES[platform].name
