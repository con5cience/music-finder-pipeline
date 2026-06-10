"""Fetch cache (ADR-017 §5): every non-audio third-party fetch persists to disk;
re-parsing never re-crawls. DB index + gzipped filesystem blobs.

The HTTP fetcher is injected so tests never touch the network.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from pipeline.fetch_cache import CachedResponse, cached_fetch

URL = "https://api.example.test/artist/1/top"


def _fetcher(payload: bytes = b'{"data": []}', status: int = 200, content_type: str = "application/json"):
    calls = []

    def fetch(url: str) -> tuple[int, str, bytes]:
        calls.append(url)
        return status, content_type, payload

    fetch.calls = calls
    return fetch


def test_miss_fetches_stores_and_returns(conn, tmp_path: Path):
    f = _fetcher(b'{"data": [1]}')
    r = cached_fetch(conn, "deezer", URL, fetcher=f, cache_dir=tmp_path)
    assert isinstance(r, CachedResponse)
    assert (r.status, r.from_cache, r.body) == (200, False, b'{"data": [1]}')
    assert f.calls == [URL]
    row = conn.execute(
        "SELECT platform, status, content_path FROM fetch_cache WHERE url = %s", (URL,)
    ).fetchone()
    assert row[0] == "deezer" and row[1] == 200
    blob = tmp_path / row[2]
    assert blob.exists()
    assert gzip.decompress(blob.read_bytes()) == b'{"data": [1]}'


def test_hit_never_refetches(conn, tmp_path: Path):
    f = _fetcher(b"payload-1")
    r1 = cached_fetch(conn, "deezer", URL, fetcher=f, cache_dir=tmp_path)
    r2 = cached_fetch(conn, "deezer", URL, fetcher=f, cache_dir=tmp_path)
    assert (r1.from_cache, r2.from_cache) == (False, True)
    assert r2.body == b"payload-1"
    assert len(f.calls) == 1  # the law: never refetch


def test_404_is_negative_cached(conn, tmp_path: Path):
    f = _fetcher(b'{"error": "no such artist"}', status=404)
    r1 = cached_fetch(conn, "deezer", URL, fetcher=f, cache_dir=tmp_path)
    r2 = cached_fetch(conn, "deezer", URL, fetcher=f, cache_dir=tmp_path)
    assert (r1.status, r2.status) == (404, 404)
    assert r2.from_cache is True
    assert len(f.calls) == 1  # dead pages are never re-crawled either


def test_transient_4xx_raises_and_is_not_cached(conn, tmp_path: Path):
    # 403/429 are rate-limit/auth transients: caching them would poison the
    # URL forever (and terminal scan verdicts would lock the artist out).
    for status in (401, 403, 429):
        f = _fetcher(b"rate limited", status=status)
        with pytest.raises(RuntimeError, match=str(status)):
            cached_fetch(conn, "bandcamp", URL + f"?s{status}", fetcher=f, cache_dir=tmp_path)
        n = conn.execute(
            "SELECT count(*) FROM fetch_cache WHERE url = %s", (URL + f"?s{status}",)
        ).fetchone()[0]
        assert n == 0  # nothing cached — a later attempt refetches


def test_5xx_raises_and_is_not_cached(conn, tmp_path: Path):
    f = _fetcher(b"upstream sad", status=503)
    with pytest.raises(RuntimeError, match="503"):
        cached_fetch(conn, "deezer", URL, fetcher=f, cache_dir=tmp_path)
    assert conn.execute("SELECT count(*) FROM fetch_cache WHERE url = %s", (URL,)).fetchone()[0] == 0
    # next attempt fetches again (transient failures must not poison the cache)
    f2 = _fetcher(b"recovered")
    r = cached_fetch(conn, "deezer", URL, fetcher=f2, cache_dir=tmp_path)
    assert (r.status, r.body) == (200, b"recovered")


def test_identical_bodies_share_one_blob(conn, tmp_path: Path):
    body = b"same-bytes-everywhere"
    cached_fetch(conn, "deezer", URL + "?a", fetcher=_fetcher(body), cache_dir=tmp_path)
    cached_fetch(conn, "deezer", URL + "?b", fetcher=_fetcher(body), cache_dir=tmp_path)
    paths = {
        r[0] for r in conn.execute(
            "SELECT content_path FROM fetch_cache WHERE url LIKE %s", (URL + "%",)
        ).fetchall()
    }
    assert len(paths) == 1  # content-addressed: two rows, one blob
    blobs = list(tmp_path.rglob("*.gz"))
    assert len(blobs) == 1


def test_blobs_are_namespaced_by_platform(conn, tmp_path: Path):
    cached_fetch(conn, "bandcamp", URL, fetcher=_fetcher(b"x"), cache_dir=tmp_path)
    row = conn.execute("SELECT content_path FROM fetch_cache WHERE url = %s", (URL,)).fetchone()
    assert row[0].startswith("bandcamp/")


def test_refresh_bypasses_cache_read_but_still_writes(conn, tmp_path: Path):
    cached_fetch(conn, "deezer", URL, fetcher=_fetcher(b"stale-signed-url"), cache_dir=tmp_path)
    f2 = _fetcher(b"fresh-signed-url")
    r = cached_fetch(conn, "deezer", URL, fetcher=f2, cache_dir=tmp_path, refresh=True)
    assert (r.from_cache, r.body) == (False, b"fresh-signed-url")
    assert len(f2.calls) == 1
    # the cache row now holds the fresh payload for future readers
    r3 = cached_fetch(conn, "deezer", URL, fetcher=_fetcher(b"never-called"), cache_dir=tmp_path)
    assert (r3.from_cache, r3.body) == (True, b"fresh-signed-url")
