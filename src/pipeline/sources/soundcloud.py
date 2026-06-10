"""SoundCloud discovery via the OFFICIAL API (registered app, client_credentials).

Empirical findings this module encodes (probed 2026-06-09):
- App-only tokens stream 30-SECOND INTRO PREVIEWS for every track, regardless
  of `access` ('playable' included — a 58-min mix streamed 30s from
  cf-preview-media). SC is therefore a PREVIEW-grade source: floor 10,
  non-windowed, like Deezer — see the PLATFORMS descriptor. Previews are
  track INTROS (offset 0-30), weaker than Deezer's label-chosen hooks; the
  real track duration is kept in evidence, duration_s stores the actual 30s.
- The SC-only corpus population (96k artists) is overwhelmingly `playable`
  self-uploads — the preview limit is the API tier, not catalog absence.
- /users/{id}/tracks returns newest-first; we record walk order as
  release_index/track_index (uuid pks make row order a lottery).
- Stream URLs are CloudFront-signed and rot; refresh_soundcloud re-resolves
  the /tracks/{id}/stream redirect with a fresh token.
- Tokens live ~1h; grants are limited (50/12h per docs) → process-level cache
  with expiry. JSON endpoints go through the fetch cache (ADR-017 §5) with an
  OAuth fetcher; signed stream URLs are never cached (they rot by design).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from psycopg import Connection

from pipeline.config import Settings
from pipeline.fetch_cache import cached_fetch
from pipeline.sources.common import identity_row, insert_audio_track, store_refreshed_url

_API = "https://api.soundcloud.com"
_PREVIEW_S = 30           # what app-only tokens actually stream, always
_TRACK_PAGE_LIMIT = 50    # newest N tracks per artist (one page)

_token_cache: dict = {}   # {"access": str, "exp": epoch} — process-level


class SoundcloudAuthError(RuntimeError):
    """Credentials missing or the token grant failed."""


def _token() -> str:
    now = time.time()
    if _token_cache.get("exp", 0) - 60 > now:
        return _token_cache["access"]
    s = Settings()
    if not (s.soundcloud_client_id and s.soundcloud_client_secret):
        raise SoundcloudAuthError("SOUNDCLOUD_CLIENT_ID/SECRET not configured")
    req = urllib.request.Request(
        f"{_API}/oauth2/token",
        data=urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": s.soundcloud_client_id,
            "client_secret": s.soundcloud_client_secret,
        }).encode(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            tok = json.load(r)
    except urllib.error.HTTPError as e:
        raise SoundcloudAuthError(f"token grant failed: HTTP {e.code}") from e
    _token_cache.update(access=tok["access_token"], exp=now + int(tok.get("expires_in", 3599)))
    return _token_cache["access"]


def _oauth_fetcher(url: str) -> tuple[int, str, bytes]:
    """fetch_cache-compatible fetcher carrying the OAuth header."""
    req = urllib.request.Request(url, headers={"Authorization": f"OAuth {_token()}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: ARG002
        return None


def resolve_stream_url(track_api_id: str) -> str | None:
    """The signed CDN URL behind /tracks/{id}/stream (302 capture). Rots."""
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(
        f"{_API}/tracks/{track_api_id}/stream",
        headers={"Authorization": f"OAuth {_token()}"},
    )
    try:
        opener.open(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return e.headers.get("Location")
        return None
    return None


def parse_tracks(body: bytes) -> list[dict]:
    """Streamable tracks from a /users/{id}/tracks response, walk order kept."""
    collection = json.loads(body).get("collection", [])
    return [t for t in collection if t.get("streamable") and t.get("stream_url")]


def discover_soundcloud(
    conn: Connection,
    artist_id: str,
    permalink: str,
    *,
    fetcher=None,
    stream_resolver=resolve_stream_url,
) -> int:
    """Resolve the artist page, list newest tracks, store ALL streamable ones
    with resolved (signed, rotting) preview URLs. Returns NEW rows written."""
    identity_id = identity_row(conn, "soundcloud", artist_id, permalink)
    fetcher = fetcher or _oauth_fetcher

    resolve_url = f"{_API}/resolve?url=" + urllib.parse.quote(
        f"https://soundcloud.com/{permalink}", safe=""
    )
    res = cached_fetch(conn, "soundcloud", resolve_url, fetcher=fetcher)
    if res.status == 404:
        return 0  # account gone — negative-cached, terminal verdict upstream
    user = json.loads(res.body)
    tracks_url = f"{_API}/users/{user['id']}/tracks?limit={_TRACK_PAGE_LIMIT}&linked_partitioning=true"
    res = cached_fetch(conn, "soundcloud", tracks_url, fetcher=fetcher)
    if res.status == 404:
        return 0

    written = 0
    for track_index, t in enumerate(parse_tracks(res.body)):
        stream = stream_resolver(str(t["id"]))
        if not stream:
            continue
        evidence = {
            "source": "soundcloud_api",
            "access": t.get("access"),
            "full_duration_s": (t.get("duration") or 0) // 1000,
            "title": t.get("title"),
            # newest-first walk order IS the selection order (no releases on
            # SC: each track is its own "release" for the selection pass)
            "release_index": track_index,
            "track_index": track_index,
            "preview_only": True,  # app-only API: 30s intro previews, always
        }
        if insert_audio_track(
            conn, artist_id, "soundcloud", str(t["id"]), stream,
            _PREVIEW_S,  # ACTUAL audio length, not the metadata duration
            identity_id, evidence,
        ):
            written += 1
    return written


def refresh_soundcloud(conn: Connection, platform_track_id: str) -> str | None:
    """Self-heal a rotted signed URL: re-resolve the stream redirect."""
    fresh = resolve_stream_url(platform_track_id)
    if fresh:
        store_refreshed_url(conn, "soundcloud", platform_track_id, fresh)
    return fresh
