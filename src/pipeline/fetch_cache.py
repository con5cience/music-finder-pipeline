"""Fetch cache (ADR-017 §5): never re-crawl what we already fetched.

`cached_fetch` checks the DB index; on miss it fetches (injectable fetcher;
default urllib with our UA and optional proxy), gzips the body to a
content-addressed blob ({platform}/{hh}/{hash}.gz under the cache dir), and
upserts the index row. 2xx and 404 are cached (404 = negative cache: dead
pages are dead); 5xx/network errors raise and cache nothing, so transient
failures never poison the cache. Audio bytes must never go through here.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from psycopg import Connection

from pipeline.config import Settings

_UA = "music-finder-pipeline/0.1 (wstiern@gmail.com)"


@dataclass(frozen=True)
class CachedResponse:
    status: int
    body: bytes
    content_type: str | None
    from_cache: bool


def _http_get(url: str) -> tuple[int, str, bytes]:
    settings = Settings()
    handlers: list[urllib.request.BaseHandler] = []
    if settings.proxy_url:
        handlers.append(urllib.request.ProxyHandler({"http": settings.proxy_url, "https": settings.proxy_url}))
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with opener.open(req, timeout=30) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as e:  # non-2xx still has a body we may cache
        return e.code, e.headers.get("Content-Type", "") if e.headers else "", e.read()


def _http_post_json(url: str, payload: dict) -> tuple[int, str, bytes]:
    settings = Settings()
    handlers: list[urllib.request.BaseHandler] = []
    if settings.proxy_url:
        handlers.append(urllib.request.ProxyHandler({"http": settings.proxy_url, "https": settings.proxy_url}))
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
        method="POST", headers={"User-Agent": _UA, "Content-Type": "application/json"})
    try:
        with opener.open(req, timeout=30) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as e:  # non-2xx still has a body we may cache
        return e.code, e.headers.get("Content-Type", "") if e.headers else "", e.read()


def _blob_path(platform: str, content_hash: str) -> str:
    return f"{platform}/{content_hash[:2]}/{content_hash}.gz"


def cached_fetch(
    conn: Connection,
    platform: str,
    url: str,
    *,
    fetcher=None,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
    post_json: dict | None = None,
) -> CachedResponse:
    """`refresh=True` skips the cache READ (still writes): for payloads carrying
    expiring signed URLs that must be re-resolved live (Deezer previews)."""
    cache_dir = Path(cache_dir if cache_dir is not None else Settings().fetch_cache_dir).expanduser()

    # POST responses cache under a synthetic key: url + body hash (the BC
    # Discover API is POST-JSON; same 2xx/404-only law applies).
    cache_key = url
    if post_json is not None:
        body_bytes = json.dumps(post_json, sort_keys=True).encode()
        cache_key = f"{url}#post:{hashlib.sha256(body_bytes).hexdigest()[:24]}"

    row = (
        None
        if refresh
        else conn.execute(
            "SELECT status, content_type, content_path FROM fetch_cache WHERE url = %s", (cache_key,)
        ).fetchone()
    )
    if row is not None:
        status, content_type, content_path = row
        body = gzip.decompress((cache_dir / content_path).read_bytes())
        return CachedResponse(status, body, content_type, from_cache=True)

    if post_json is not None and fetcher is None:
        status, content_type, body = _http_post_json(url, post_json)
    else:
        status, content_type, body = (fetcher or _http_get)(url)
    if not (200 <= status < 300 or status == 404):
        # Only 2xx and 404 (negative cache: dead is dead) are cacheable.
        # Everything else — 5xx, 403/429 rate-limits, auth walls — is
        # transient: caching it would poison the URL permanently, and a
        # terminal scan verdict on a poisoned read locks the artist out.
        raise RuntimeError(f"fetch failed upstream ({status}) for {url}")

    content_hash = hashlib.sha256(body).hexdigest()
    rel = _blob_path(platform, content_hash)
    blob = cache_dir / rel
    if not blob.exists():  # content-addressed: identical bodies share one file
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(gzip.compress(body))
    conn.execute(
        """
        INSERT INTO fetch_cache (platform, url, status, content_type, content_hash, content_path)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (url) DO UPDATE SET
            status = EXCLUDED.status, content_type = EXCLUDED.content_type,
            content_hash = EXCLUDED.content_hash, content_path = EXCLUDED.content_path,
            fetched_at = now()
        """,
        (platform, cache_key, status, content_type or None, content_hash, rel),
    )
    return CachedResponse(status, body, content_type or None, from_cache=False)
