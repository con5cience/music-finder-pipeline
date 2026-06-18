from __future__ import annotations

from pipeline.harvest_bc_tags import subdomain_of


def test_subdomain_extraction():
    assert subdomain_of("https://housesofheaven.bandcamp.com/album/remnant") == "housesofheaven"
    assert subdomain_of("https://HousesOfHeaven.bandcamp.com/music") == "housesofheaven"  # lowercased
    assert subdomain_of("http://foo.bandcamp.com/") == "foo"


def test_subdomain_non_bandcamp_is_none():
    assert subdomain_of("https://bandcamp.com/tag/ambient") is None  # no subdomain
    assert subdomain_of("https://example.com/x") is None
