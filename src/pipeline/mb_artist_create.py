"""MB artist-creation driver — phase 1b (2026-06-12).

MusicBrainz has NO ws/2 endpoint for creating artists: creation is the
website's edit system, driven bot-style — session login, then a form POST
to /artist/create. Transport is injected everywhere so the full flow is
testable offline; real-world form quirks get ironed against
test.musicbrainz.org (which is the entire reason rehearsal exists).

Safety model:
  - target is EXPLICIT; CLI defaults to test. Live requires --target live.
  - live submits ONLY payloads a human blessed (status='approved');
    test rehearsal may consume staged spot_check payloads directly.
  - artists with any open integrity flag (coherence / slop) never submit
    — the same freezer as publish, re-checked here at the door.
  - created MBIDs are recorded per target; TEST mbids are fake-world and
    are never attached to our artists (live attach happens via mb-sync).

URL relationship link-type ids are read from mb_raw.link_type by NAME —
the dump is the source of truth, never hardcoded ids.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar

from psycopg import Connection

from pipeline.mb_submit import UA, base_for

_CSRF_RE = re.compile(r'name="csrf_token"[^>]*value="([^"]+)"')
_MBID_RE = re.compile(r"/artist/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
# platform -> mb link_type NAME (resolved to ids via mb_raw.link_type)
_URL_REL_NAMES = {"bandcamp": "bandcamp", "soundcloud": "soundcloud",
                  "youtube": "youtube", "deezer": "free streaming"}


def default_transport():
    """Cookie-keeping opener; returns (status, headers, body) and never
    follows redirects (the created-artist MBID rides the 302 Location)."""
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):  # noqa: D102
            return None

    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(CookieJar()), _NoRedirect())

    def fetch(url: str, data: dict | None = None) -> tuple[int, dict, bytes]:
        body = urllib.parse.urlencode(data).encode() if data is not None else None
        req = urllib.request.Request(url, data=body, headers={"User-Agent": UA})
        try:
            with opener.open(req, timeout=60) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers or {}), e.read()

    return fetch


def login(fetch, target: str, username: str, password: str) -> None:
    base = base_for(target)
    status, _h, body = fetch(f"{base}/login")
    m = _CSRF_RE.search(body.decode("utf-8", "replace"))
    form = {"username": username, "password": password, "remember_me": "1"}
    if m:
        form["csrf_token"] = m.group(1)
    status, headers, body = fetch(f"{base}/login", form)
    if status not in (302, 303):
        raise SystemExit(f"MB login failed (HTTP {status}) — check MB_BOT_USER/MB_BOT_PASSWORD"
                         f" for {target}: {body[:200]!r}")


def link_type_ids(conn: Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT name, id FROM mb_raw.link_type WHERE name = ANY(%s)",
        (list(set(_URL_REL_NAMES.values())),),
    ).fetchall()
    return dict(rows)


def create_artist(fetch, conn: Connection, target: str, payload: dict,
                  *, edit_note: str) -> str:
    base = base_for(target)
    status, _h, body = fetch(f"{base}/artist/create")
    m = _CSRF_RE.search(body.decode("utf-8", "replace"))
    form: dict[str, str] = {
        "edit-artist.name": payload["name"],
        "edit-artist.sort_name": payload.get("sort_name") or payload["name"],
        "edit-artist.edit_note": edit_note,
    }
    if m:
        form["csrf_token"] = m.group(1)
    lt = link_type_ids(conn)
    n = 0
    for u in payload.get("urls", []):
        name = _URL_REL_NAMES.get(u.get("platform"))
        if not name or name not in lt:
            continue
        form[f"edit-artist.url.{n}.text"] = u["url"]
        form[f"edit-artist.url.{n}.link_type_id"] = str(lt[name])
        n += 1
    status, headers, body = fetch(f"{base}/artist/create", form)
    if status not in (302, 303):
        raise RuntimeError(f"artist create rejected (HTTP {status}): {body[:300]!r}")
    m = _MBID_RE.search(headers.get("Location", "") or headers.get("location", ""))
    if not m:
        raise RuntimeError(f"created but no MBID in redirect: {headers!r}")
    return m.group(1)


def submit_artists(conn: Connection, *, target: str, limit: int = 5,
                   fetch=None, username: str | None = None,
                   password: str | None = None, pace_s: float = 6.0) -> dict:
    """Submit staged payloads as new MB artists. Live: approved-only.
    Test rehearsal: spot_check payloads allowed."""
    import os

    statuses = ("approved",) if target == "live" else ("approved", "spot_check")
    rows = conn.execute(
        """
        SELECT s.id, s.artist_id, s.payload FROM mb_submission s
        WHERE s.status = ANY(%s) AND s.created_mbid IS NULL
          AND NOT EXISTS (SELECT 1 FROM review_item ri WHERE ri.subject_id = s.artist_id
                          AND ri.reason IN ('source_coherence', 'ai_slop')
                          AND ri.status = 'pending')
        ORDER BY s.id LIMIT %s
        """,
        (list(statuses), limit),
    ).fetchall()
    if not rows:
        return {"submitted": 0}
    if fetch is None:
        fetch = default_transport()
        username = username or os.environ.get("MB_BOT_USER")
        password = password or os.environ.get("MB_BOT_PASSWORD")
        if not username or not password:
            raise SystemExit("MB_BOT_USER / MB_BOT_PASSWORD not set — the edit "
                             "system needs the bot's website session, not OAuth")
        login(fetch, target, username, password)
    out = {"submitted": 0, "failed": 0}
    for sid, artist_id, payload in rows:
        try:
            mbid = create_artist(
                fetch, conn, target, payload,
                edit_note=("crates.ltd underground-discovery bot — announced and "
                           "blessed on the community forum; full analysis + human "
                           "spot-check behind every submission."))
            conn.execute(
                "UPDATE mb_submission SET created_mbid = %s, target = %s, status = 'submitted' "
                "WHERE id = %s", (mbid, target, sid))
            out["submitted"] += 1
        except RuntimeError as exc:
            conn.execute(
                "UPDATE mb_submission SET status = 'failed', target = %s WHERE id = %s",
                (target, sid))
            print(f"submission failed for {artist_id}: {exc}")
            out["failed"] += 1
        conn.commit()
        time.sleep(pace_s)  # bot-account pacing: slower than any human reviewer
    return out
