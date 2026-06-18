"""parse_bandcamp_tags: human genre tags from a Bandcamp release page, with the
artist location (same <a class="tag">) dropped. Fixture mirrors the real markup
(multi-line href, &amp; entity, location both as a span and a tag) confirmed
against Houses of Heaven's cached /album/remnant page (2026-06-18)."""

from __future__ import annotations

from pipeline.sources.bandcamp import parse_bandcamp_tags

SAMPLE = b"""
<span class="location secondaryText">Oakland, California</span>
<div class="tralbumData tralbum-tags">
  <a class="tag" href="https://bandcamp.com/discover/alternative?from=tralbum&artist=1"
                >alternative</a>
  <a class="tag" href="https://bandcamp.com/discover/drum-bass?from=tralbum&artist=1"
                >drum &amp; bass</a>
  <a class="tag" href="https://bandcamp.com/discover/electronic?from=tralbum&artist=1"
                >electronic</a>
  <a class="tag" href="https://bandcamp.com/discover/industrial?from=tralbum&artist=1"
                >industrial</a>
  <a class="tag" href="https://bandcamp.com/discover/post-punk?from=tralbum&artist=1"
                >post-punk</a>
  <a class="tag" href="https://bandcamp.com/discover/techno?from=tralbum&artist=1"
                >techno</a>
  <a class="tag" href="https://bandcamp.com/discover/oakland?from=tralbum&artist=1"
                >Oakland</a>
</div>
"""


def test_extracts_genres_drops_location_and_unescapes():
    assert parse_bandcamp_tags(SAMPLE) == [
        "alternative",
        "drum & bass",
        "electronic",
        "industrial",
        "post-punk",
        "techno",
    ]


def test_no_tag_block_returns_empty():
    assert parse_bandcamp_tags(b"<html><body>no tags here</body></html>") == []


def test_dedupes_preserving_order():
    html = (
        b'<a class="tag" href="/discover/techno">techno</a>'
        b'<a class="tag" href="/discover/ebm">EBM</a>'
        b'<a class="tag" href="/discover/techno">Techno</a>'
    )
    assert parse_bandcamp_tags(html) == ["techno", "ebm"]


def test_drops_location_even_without_comma():
    html = (
        b'<span class="location">Berlin</span>'
        b'<a class="tag" href="/discover/coldwave">coldwave</a>'
        b'<a class="tag" href="/discover/berlin">Berlin</a>'
    )
    assert parse_bandcamp_tags(html) == ["coldwave"]
