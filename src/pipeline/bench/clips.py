"""Load a labeled clip set from disk for the benchmark.

Layout: ``<root>/<artist>/<file>.<ext>`` — each sub-directory is an artist label,
each audio file under it is a clip. Supported extensions are the common
soundfile-readable ones (wav/flac/mp3/ogg/aiff/...).
"""

from __future__ import annotations

import os

from pipeline.bench.types import Clip

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".oga", ".aiff", ".aif", ".au", ".w64", ".caf"}


def load_clip_dir(root: str) -> list[Clip]:
    clips: list[Clip] = []
    for artist in sorted(os.listdir(root)):
        adir = os.path.join(root, artist)
        if not os.path.isdir(adir):
            continue
        for fn in sorted(os.listdir(adir)):
            if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                clips.append(Clip(id=f"{artist}/{fn}", artist_id=artist, path=os.path.join(adir, fn)))
    return clips
