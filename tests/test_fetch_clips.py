"""Auto-clip fetcher: artist name -> Deezer artist entity -> own /top previews.

HTTP is injected (get_json / get_bytes), so these run offline and assert the
*correctness safeguard*: a saved clip is always provably the named artist
(resolved by artist ID, kept only when that artist is the Main contributor).
"""

from __future__ import annotations

import json

from pipeline.bench.fetch_clips import (
    deezer_top_main_tracks,
    fetch_artist_clips,
    normalize_name,
    resolve_deezer_artist,
    slugify,
)


def test_slugify_filesystem_safe():
    assert slugify("Aphex Twin") == "aphex-twin"
    assert slugify("Björk!") == "bjork"
    assert slugify("  A$AP   Rocky  ") == "a-ap-rocky"  # $ and runs of space collapse to one dash
    assert slugify("Sigur Rós") == "sigur-ros"


def test_normalize_name_matches_across_punctuation_and_diacritics():
    assert normalize_name("Björk") == normalize_name("bjork")
    assert normalize_name("Sigur Rós") == normalize_name("sigur ros")
    assert normalize_name("M.I.A.") == normalize_name("mia")


def test_resolve_picks_normalized_match_then_max_fans():
    def get_json(url):
        assert "search/artist" in url
        return {"data": [
            {"id": 1, "name": "Burial UK", "nb_fan": 999},      # not a name match
            {"id": 2, "name": "Burial", "nb_fan": 10},          # match
            {"id": 3, "name": "burial", "nb_fan": 5000},        # match, more fans
        ]}

    assert resolve_deezer_artist("Burial", get_json=get_json) == 3


def test_resolve_unresolved_when_no_name_match():
    def get_json(url):
        return {"data": [{"id": 1, "name": "Someone Else", "nb_fan": 1}]}

    assert resolve_deezer_artist("Aphex Twin", get_json=get_json) is None


def test_top_tracks_drops_wrong_artist_features_and_no_preview():
    artist_id = 42

    def get_json(url):
        assert f"artist/{artist_id}/top" in url
        return {"data": [
            # kept: this artist is the track artist AND Main contributor, has preview
            {"id": 100, "title": "Real", "preview": "https://p/100.mp3",
             "artist": {"id": 42},
             "contributors": [{"id": 42, "role": "Main"}]},
            # dropped: a feature (Main is someone else)
            {"id": 101, "title": "Feat", "preview": "https://p/101.mp3",
             "artist": {"id": 99},
             "contributors": [{"id": 99, "role": "Main"}, {"id": 42, "role": "Featured"}]},
            # dropped: no preview url
            {"id": 102, "title": "NoPrev", "preview": "",
             "artist": {"id": 42},
             "contributors": [{"id": 42, "role": "Main"}]},
            # dropped: track.artist.id mismatch (compilation misattribution)
            {"id": 103, "title": "Comp", "preview": "https://p/103.mp3",
             "artist": {"id": 7},
             "contributors": [{"id": 42, "role": "Main"}]},
        ]}

    tracks = deezer_top_main_tracks(artist_id, 5, get_json=get_json)
    assert [t["id"] for t in tracks] == [100]


def test_fetch_artist_clips_writes_files_and_manifest(tmp_path):
    calls = {"json": [], "bytes": []}

    def get_json(url):
        calls["json"].append(url)
        if "search/artist" in url:
            return {"data": [{"id": 42, "name": "Aphex Twin", "nb_fan": 100}]}
        return {"data": [
            {"id": 100, "title": "Xtal", "preview": "https://p/100.mp3",
             "artist": {"id": 42}, "contributors": [{"id": 42, "role": "Main"}]},
            {"id": 101, "title": "Tha", "preview": "https://p/101.mp3",
             "artist": {"id": 42}, "contributors": [{"id": 42, "role": "Main"}]},
        ]}

    def get_bytes(url):
        calls["bytes"].append(url)
        return b"FAKEMP3" + url.encode()

    manifest = fetch_artist_clips("Aphex Twin", str(tmp_path), n=5,
                                  get_json=get_json, get_bytes=get_bytes)

    adir = tmp_path / "aphex-twin"
    assert (adir / "100.mp3").read_bytes().startswith(b"FAKEMP3")
    assert (adir / "101.mp3").exists()
    assert manifest["status"] == "ok"
    assert manifest["source"] == "deezer"
    assert manifest["artist_id"] == 42
    assert {c["track_id"] for c in manifest["clips"]} == {100, 101}
    # manifest persisted to disk for auditability
    on_disk = json.loads((adir / "manifest.json").read_text())
    assert on_disk["artist_id"] == 42
    assert all(c["url"].startswith("https://p/") for c in on_disk["clips"])


def test_fetch_artist_clips_skips_unresolved_without_writing(tmp_path):
    def get_json(url):
        return {"data": []}  # no search hit

    def get_bytes(url):  # must never be called for an unresolved artist
        raise AssertionError("should not download for an unresolved artist")

    manifest = fetch_artist_clips("Nonexistent Act", str(tmp_path),
                                  get_json=get_json, get_bytes=get_bytes)
    assert manifest["status"] == "unresolved"
    assert manifest["clips"] == []
    assert not (tmp_path / "nonexistent-act").exists()


def test_fetch_stops_at_n_clips(tmp_path):
    def get_json(url):
        if "search/artist" in url:
            return {"data": [{"id": 1, "name": "X", "nb_fan": 1}]}
        return {"data": [
            {"id": i, "title": f"t{i}", "preview": f"https://p/{i}.mp3",
             "artist": {"id": 1}, "contributors": [{"id": 1, "role": "Main"}]}
            for i in range(10)
        ]}

    manifest = fetch_artist_clips("X", str(tmp_path), n=3,
                                  get_json=get_json, get_bytes=lambda u: b"x")
    assert len(manifest["clips"]) == 3
