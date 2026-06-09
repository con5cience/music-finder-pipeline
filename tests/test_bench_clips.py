"""Labeled-clip loader: <root>/<artist>/<file>.<ext> → Clips."""

from __future__ import annotations

import os

from pipeline.bench.clips import load_clip_dir


def test_loads_labeled_clips(tmp_path):
    layout = {"x": ["a.wav", "b.flac"], "y": ["c.mp3"]}
    for artist, files in layout.items():
        d = tmp_path / artist
        d.mkdir()
        for f in files:
            (d / f).write_bytes(b"")
        (d / "notes.txt").write_text("ignored")  # non-audio is skipped
    clips = load_clip_dir(str(tmp_path))
    assert len(clips) == 3
    assert {c.artist_id for c in clips} == {"x", "y"}
    assert all(c.path and os.path.exists(c.path) for c in clips)
    assert all(c.id.split("/")[0] == c.artist_id for c in clips)


def test_ignores_loose_files(tmp_path):
    (tmp_path / "loose.wav").write_bytes(b"")  # not under an artist subdir
    assert load_clip_dir(str(tmp_path)) == []
