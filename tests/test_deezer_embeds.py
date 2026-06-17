"""Deezer embed sync (#26): the pure embed-row builder. The DB sync
(populate_deezer_embeds) is a thin factory-read -> serving-write loop verified
live against both DBs."""

from __future__ import annotations

from pipeline.deezer_embeds import build_deezer_top_tracks


def test_builds_widget_rows_in_serving_snake_case_shape():
    rows = build_deezer_top_tracks(["3013685181", "3013685191"])
    assert rows == [
        {"name": "Deezer", "embed_url": "https://widget.deezer.com/widget/dark/track/3013685181"},
        {"name": "Deezer", "embed_url": "https://widget.deezer.com/widget/dark/track/3013685191"},
    ]


def test_caps_at_max_and_dedups_preserving_order():
    rows = build_deezer_top_tracks(["a", "a", "b", "c", "d"], max_n=3)
    assert [r["embed_url"].rsplit("/", 1)[1] for r in rows] == ["a", "b", "c"]


def test_skips_empty_ids_and_handles_empty_input():
    assert build_deezer_top_tracks([]) == []
    assert build_deezer_top_tracks(["", None, "x"]) == [  # type: ignore[list-item]
        {"name": "Deezer", "embed_url": "https://widget.deezer.com/widget/dark/track/x"},
    ]
