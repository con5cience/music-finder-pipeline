"""Pure-function tests for the deterministic tag sweep — the genre-signal block
gate and the microgenre KEEP gate (the two halves of 'what to keep')."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "sweep_junk_tags", Path(__file__).resolve().parent.parent / "scripts" / "sweep_junk_tags.py"
)
swp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(swp)


def test_has_genre_signal():
    gw = frozenset({"rock", "punk", "house", "ambient"})
    # token match, root-substring match (no-space compound), and the junk tail
    assert swp.has_genre_signal("dark rock", gw)
    assert swp.has_genre_signal("afropunk", gw)  # substring root, no token
    assert swp.has_genre_signal("witchgaze", gw)  # 'gaze' root via regex
    assert not swp.has_genre_signal("alan turing", gw)
    assert not swp.has_genre_signal("528 hz", gw)


def test_microgenre_keep_accepts_real_compounds():
    for t in ("witchstep", "vaporrock", "ambienttechno", "afrofolk", "southernrock", "blackgaze"):
        assert swp.microgenre_keep(t), t


def test_microgenre_keep_rejects_junk_prefixes_and_non_compounds():
    # NON_GENRE_PREFIX guard (fish/pun/object/slur), spaced/hyphenated, too short
    for t in ("albacore", "applecore", "horsecore", "guitarsynth", "tardcore"):
        assert not swp.microgenre_keep(t), t
    assert not swp.microgenre_keep("dark rock")  # has a separator -> not a no-space compound
    assert not swp.microgenre_keep("post-punk")  # hyphen
    assert not swp.microgenre_keep("rock")  # prefix too short / no compound
    assert not swp.microgenre_keep("alan turing")  # no genre suffix
