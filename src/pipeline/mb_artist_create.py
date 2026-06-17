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
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar

from psycopg import Connection

from pipeline.mb_submit import UA, _env_fallback, base_for

_CSRF_RE = re.compile(r'name="csrf_token"[^>]*value="([^"]+)"')
_MBID_RE = re.compile(r"/artist/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
# The required name input is our authenticated-and-editable signal: modern MB
# forms carry NO csrf_token field (verified against the live /artist/create form
# 2026-06-17), so we must detect the edit form itself, not a csrf token, to know
# the session is usable. An anonymous session 302s away; an unverified-email
# account 401s — neither renders this field.
_FORM_RE = re.compile(r'name="edit-artist\.name"')


def _edit_form_present(html: str) -> bool:
    return bool(_FORM_RE.search(html))


def _is_duplicate_page(html: str) -> bool:
    # A name collision re-renders the create form (HTTP 200) with a "possible
    # duplicates" panel and demands a disambiguation comment before it will
    # create. (The URL was already cleared as unique upstream, so this is a
    # same-name-different-artist case, not a true duplicate.)
    return "possible-duplicates" in html


# platform -> mb link_type NAME (resolved to ids via mb_raw.link_type)
_URL_REL_NAMES = {"bandcamp": "bandcamp", "soundcloud": "soundcloud",
                  "youtube": "youtube", "deezer": "free streaming"}


def default_transport(session_cookie: str | None = None, host: str | None = None):
    """Cookie-keeping opener; returns (status, headers, body) and never
    follows redirects (the created-artist MBID rides the 302 Location).

    When session_cookie is given it is pre-seeded as the MB website session
    (musicbrainz_server_session) for `host` — the only viable auth now that MB
    login is MetaBrainz SSO behind MTCaptcha (no programmatic form login). Grab
    the cookie from a browser logged into the target server."""
    from http.cookiejar import Cookie

    jar = CookieJar()
    if session_cookie and host:
        jar.set_cookie(Cookie(
            version=0, name="musicbrainz_server_session", value=session_cookie,
            port=None, port_specified=False, domain=host, domain_specified=True,
            domain_initial_dot=False, path="/", path_specified=True, secure=True,
            expires=None, discard=False, comment=None, comment_url=None, rest={}))

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):  # noqa: D102
            return None

    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar), _NoRedirect())

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
    # MB reuses a link-type NAME across entity types (e.g. 'bandcamp' exists as
    # artist-url 718, label-url 719, genre-url 1092). We attach URLs to ARTISTS,
    # so filter to the artist<->url link type — matching by name alone picked an
    # arbitrary (often label/genre) id and would mis-type or reject the link.
    rows = conn.execute(
        "SELECT name, id FROM mb_raw.link_type WHERE name = ANY(%s) "
        "AND entity_type0 = 'artist' AND entity_type1 = 'url'",
        (list(set(_URL_REL_NAMES.values())),),
    ).fetchall()
    return dict(rows)


def mb_artist_url_owner(target: str, url: str, *, pace_s: float = 1.0) -> str | None:
    """MBID of an existing MB artist already linked to `url`, else None.

    The URL is the reliable duplicate discriminator: if MB already has this
    bandcamp/streaming URL attached to an artist, OUR artist IS that artist —
    we link the existing entity rather than create a second one in the commons.
    A 404 means MB has never seen the URL (genuinely new artist). Read via ws/2
    (not the website), which is not behind the browser-verification wall."""
    base = base_for(target)
    q = (f"{base}/ws/2/url?resource={urllib.parse.quote(url, safe='')}"
         "&inc=artist-rels&fmt=json")
    req = urllib.request.Request(q, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # URL unknown to MB
        raise
    finally:
        time.sleep(pace_s)  # ws/2 anonymous rate limit (~1 req/s)
    for rel in data.get("relations", []):
        artist = rel.get("artist") or {}
        if artist.get("id"):
            return artist["id"]
    return None


def _disambiguation(payload: dict) -> str:
    """A name collision requires a disambiguation comment. Location is the most
    natural human discriminator for same-named artists; fall back to a minimal
    honest descriptor (the human approval gate refines weak ones)."""
    area = (payload.get("area_hint") or "").strip()
    return area or "Bandcamp artist"


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
    # Name collision: MB re-renders (HTTP 200, possible-duplicates) and requires
    # a disambiguation. The URL was already cleared as unique upstream, so this
    # is a distinct same-named artist — supply a disambiguation and create anyway.
    if status == 200 and _is_duplicate_page(body.decode("utf-8", "replace")):
        form["edit-artist.comment"] = _disambiguation(payload)
        status, headers, body = fetch(f"{base}/artist/create", form)
    if status not in (302, 303):
        raise RuntimeError(f"artist create rejected (HTTP {status}): {body[:300]!r}")
    m = _MBID_RE.search(headers.get("Location", "") or headers.get("location", ""))
    if not m:
        raise RuntimeError(f"created but no MBID in redirect: {headers!r}")
    return m.group(1)


def submit_artists(conn: Connection, *, target: str, limit: int = 5,
                   fetch=None, username: str | None = None,
                   password: str | None = None, pace_s: float = 6.0,
                   url_owner=None) -> dict:
    """Submit staged payloads as new MB artists. Live: approved-only.
    Test rehearsal: spot_check payloads allowed, and does NOT consume rows (no
    ledger writes) so they stay available for the eventual live run.

    URL-checked create-anyway: before creating, each artist's URLs are checked
    against MB — if any URL already belongs to an existing artist the row is a
    true duplicate (recorded 'duplicate' with the existing MBID to link, never
    re-created). Name-only collisions are NOT duplicates: create_artist supplies
    a disambiguation and creates anyway. `url_owner` is injectable for tests."""
    url_owner = url_owner or mb_artist_url_owner
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
        base = base_for(target)
        host = urllib.parse.urlsplit(base).hostname
        # Primary auth: a browser-obtained session cookie. MB login is MetaBrainz
        # SSO behind MTCaptcha — there is NO programmatic form login. Grab the
        # musicbrainz_server_session cookie from a browser logged into the target
        # server: MB_BOT_SESSION_TEST (test) / MB_BOT_SESSION (live).
        session = _env_fallback("MB_BOT_SESSION_TEST") if target == "test" else _env_fallback("MB_BOT_SESSION")
        if session:
            fetch = default_transport(session_cookie=session, host=host)
        else:
            # Fallback: classic username/password form login — works ONLY where MB
            # is not behind SSO/captcha (e.g. a local musicbrainz-docker). Against
            # the public servers it fails the preflight below.
            fetch = default_transport()
            if target == "test":
                username = username or _env_fallback("MB_BOT_USER_TEST")
                password = password or _env_fallback("MB_BOT_PASSWORD_TEST")
            else:
                username = username or _env_fallback("MB_BOT_USER")
                password = password or _env_fallback("MB_BOT_PASSWORD")
            sess_var = "MB_BOT_SESSION_TEST" if target == "test" else "MB_BOT_SESSION"
            if not username or not password:
                raise SystemExit(f"no MB auth — set {sess_var} (browser session cookie) for the "
                                 "SSO servers, or MB_BOT_USER[_TEST]/MB_BOT_PASSWORD[_TEST] for a "
                                 "non-SSO (local) server")
            login(fetch, target, username, password)
        # Preflight: confirm the session reaches the edit form before submitting
        # anything — a stale/missing cookie otherwise silently 302s to login and
        # every create fails with a confusing "no MBID in redirect".
        sess_var = "MB_BOT_SESSION_TEST" if target == "test" else "MB_BOT_SESSION"
        st, _h, body = fetch(f"{base}/artist/create")
        if st != 200 or not _edit_form_present(body.decode("utf-8", "replace")):
            raise SystemExit(
                f"MB session not usable for edits (GET /artist/create -> HTTP {st}, no edit form). "
                f"HTTP 302 -> {sess_var} is anonymous/expired: re-copy musicbrainz_server_session "
                f"from a browser logged into {host}. HTTP 401 -> the {host} account needs a "
                "verified email address first (account/edit).")
    out = {"submitted": 0, "failed": 0, "duplicate": 0}
    # A TEST rehearsal validates the flow but MUST NOT consume staged rows: only
    # a live submission writes created_mbid/status back, so the rows remain
    # available for the eventual live run.
    persist = target == "live"
    for sid, artist_id, payload in rows:
        # URL-checked dedup: a URL already in MB means this artist already
        # exists there — link the existing entity, never create a duplicate.
        existing = None
        for u in payload.get("urls", []):
            if u.get("url"):
                existing = url_owner(target, u["url"])
                if existing:
                    break
        if existing:
            if persist:
                conn.execute(
                    "UPDATE mb_submission SET status='duplicate', created_mbid=%s, target=%s "
                    "WHERE id=%s", (existing, target, sid))
                conn.commit()
            out["duplicate"] += 1
            print(f"duplicate (URL already in MB) for {artist_id} -> {existing}")
            continue
        try:
            mbid = create_artist(
                fetch, conn, target, payload,
                edit_note=("crates.ltd underground-discovery bot — announced and "
                           "blessed on the community forum; full analysis + human "
                           "spot-check behind every submission."))
            if persist:
                conn.execute(
                    "UPDATE mb_submission SET created_mbid = %s, target = %s, status = 'submitted' "
                    "WHERE id = %s", (mbid, target, sid))
            out["submitted"] += 1
        except RuntimeError as exc:
            if persist:
                conn.execute(
                    "UPDATE mb_submission SET status = 'failed', target = %s WHERE id = %s",
                    (target, sid))
            print(f"submission failed for {artist_id}: {exc}")
            out["failed"] += 1
        if persist:
            conn.commit()
        time.sleep(pace_s)  # bot-account pacing: slower than any human reviewer
    return out
