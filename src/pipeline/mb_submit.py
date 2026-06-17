"""ADR-019 MusicBrainz contribution lane (poe mb-submit).

OAuth2: MB supports the authorization-code grant only — a ONE-TIME human
consent mints the refresh token (mb-submit --consent prints the URL;
--code exchanges and stores it in mb_oauth). After that the lane is
autonomous: queue eligible artists → spot-check (admin-visible statuses)
→ submit in small batches at bot-polite pace.

Submission reality: MB has NO create-artist REST endpoint. Until the bot
account is community-announced and the edit-driver is built (phase 1b),
`submit` runs in BUILD mode: it assembles and validates the maximal
in-vocabulary payload (name, sort name, area from BC location, url-rels
from the candidate stash + identities) and ledgers it as spot_check. The
ws/2 TAG submission (phase 2) IS implemented — tags post via the official
API for artists that already carry mbids. Base URL is configurable:
MB_SUBMIT_BASE=https://test.musicbrainz.org for rehearsal.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from psycopg import Connection

# Explicit targets: test.musicbrainz.org and musicbrainz.org are SEPARATE
# servers (accounts, OAuth apps, databases). Every flow takes target
# explicitly; CLI defaults to TEST — live requires saying --target live.
BASES = {"live": "https://musicbrainz.org", "test": "https://test.musicbrainz.org"}
MB_BASE = os.environ.get("MB_SUBMIT_BASE", BASES["live"])  # legacy alias


def base_for(target: str) -> str:
    if target not in BASES:
        raise SystemExit(f"unknown MB target {target!r} (use test|live)")
    return BASES[target]
UA = "crates.ltd-contributor/0.1 (wstiern@gmail.com)"


def _env_fallback(key: str) -> str | None:
    """os.environ first; else the pipeline .env (pydantic loads it into
    Settings, not the process env — MB_ keys live outside its prefix)."""
    if v := os.environ.get(key):
        return v
    try:
        for line in open(".env"):
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass
    return None


def _creds() -> tuple[str, str]:
    cid = _env_fallback("MB_CLIENT_ID")
    sec = _env_fallback("MB_CLIENT_SECRET")
    if not cid or not sec:
        raise SystemExit("MB_CLIENT_ID / MB_CLIENT_SECRET not set (env or pipeline .env)")
    return cid, sec


def _redirect_uri() -> str:
    """Must EXACTLY match the MB application registration. Default is the
    out-of-band URN (MB 'installed application' type); web-type apps carry
    a real callback — set MB_REDIRECT_URI to it (a localhost one is caught
    automatically by --consent's listener)."""
    return _env_fallback("MB_REDIRECT_URI") or "urn:ietf:wg:oauth:2.0:oob"


def consent_url() -> str:
    cid, _ = _creds()
    q = urllib.parse.urlencode({
        "response_type": "code", "client_id": cid,
        "redirect_uri": _redirect_uri(),
        "scope": "profile tag",  # only what phase 2 uses (review finding: over-broad consent)
        "access_type": "offline",  # REQUIRED for a refresh token (MB follows the Google convention)
        "approval_prompt": "force",  # re-consent re-issues the refresh token
    })
    return f"{MB_BASE}/oauth2/authorize?{q}"


def catch_code_locally(port: int) -> str:
    """One-shot HTTP listener for a localhost redirect URI: prints the URL,
    waits for MB's redirect, returns the ?code=."""
    import http.server
    import urllib.parse as up

    captured: dict = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            q = up.parse_qs(up.urlsplit(self.path).query)
            captured["code"] = (q.get("code") or [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h2>crates: code received - return to the terminal.</h2>")

        def log_message(self, *a):  # silence
            pass

    with http.server.HTTPServer(("127.0.0.1", port), H) as srv:
        print(consent_url())
        print(f"(listening on 127.0.0.1:{port} for the redirect...)", flush=True)
        while "code" not in captured:
            srv.handle_request()
    if not captured["code"]:
        raise SystemExit("redirect arrived without a code — check the MB app registration")
    return captured["code"]


def exchange_code(conn: Connection, code: str, *, target: str = "live") -> None:
    cid, sec = _creds()
    data = urllib.parse.urlencode({
        "grant_type": "authorization_code", "code": code,
        "client_id": cid, "client_secret": sec,
        "redirect_uri": _redirect_uri(),
    }).encode()
    req = urllib.request.Request(f"{base_for(target)}/oauth2/token", data=data,
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        tok = json.loads(r.read())
    conn.execute(
        "INSERT INTO mb_oauth (id, refresh_token) VALUES (%s, %s) "
        "ON CONFLICT (id) DO UPDATE SET refresh_token = EXCLUDED.refresh_token, updated_at = now()",
        (target, tok["refresh_token"]),
    )


def access_token(conn: Connection, *, target: str = "live") -> str:
    cid, sec = _creds()
    row = conn.execute("SELECT refresh_token FROM mb_oauth WHERE id = %s", (target,)).fetchone()
    if row is None:
        raise SystemExit(f"no MB refresh token for target {target!r} — run: "
                         f"poe mb-submit --target {target} --consent (then --code <code>)")
    data = urllib.parse.urlencode({
        "grant_type": "refresh_token", "refresh_token": row[0],
        "client_id": cid, "client_secret": sec,
    }).encode()
    req = urllib.request.Request(f"{base_for(target)}/oauth2/token", data=data, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["access_token"]
    except urllib.error.HTTPError as e:
        detail = e.read()[:200].decode("utf-8", "replace")
        raise SystemExit(
            f"MB token refresh failed ({e.code}): {detail}\n"
            "The refresh token is likely expired/revoked — re-run: "
            "poe mb-submit -- --consent  (then --code <code>)"
        ) from e


def build_payload(conn: Connection, artist_id) -> dict:
    """Maximal in-MB-vocabulary artist profile: name/sort/area/url-rels."""
    name, = conn.execute("SELECT display_name FROM artist WHERE id = %s", (artist_id,)).fetchone()
    cand = conn.execute(
        "SELECT location, links FROM bc_candidate WHERE artist_id = %s", (artist_id,)
    ).fetchone()
    # Provenance gate: only MB-declared/own-page (A) or human-confirmed (C)
    # bindings ride upstream. Tier-B is a machine guess — the typo tier
    # shipped 191 wrong artists before it was caught (2026-06-12), and a
    # B-tier URL in an MB edit would push OUR mistake into the commons.
    urls = [
        {"url": vanity or _default_url(p, pid), "platform": p}
        for p, pid, vanity in conn.execute(
            "SELECT platform, platform_id, vanity_url FROM platform_identity "
            "WHERE artist_id = %s AND binding_tier IN ('A', 'C')",
            (artist_id,),
        ).fetchall()
    ]
    return {
        "name": name,
        "sort_name": name,
        "area_hint": (cand[0] if cand else None),
        "urls": urls,
        "extra_links": (cand[1] if cand else None),
    }


def _default_url(platform: str, pid: str) -> str:
    from pipeline.publish import _URL_BUILDERS

    return _URL_BUILDERS[platform](pid)


def queue_eligible(conn: Connection, limit: int = 50) -> int:
    """Admitted + EMBEDDED (full analysis passed) + not yet queued."""
    rows = conn.execute(
        """
        SELECT bc.artist_id FROM bc_candidate bc
        JOIN artist a ON a.id = bc.artist_id AND a.embedding_source IS NOT NULL
        WHERE bc.status = 'admitted' AND a.mbid IS NULL
          AND NOT EXISTS (SELECT 1 FROM mb_submission s WHERE s.artist_id = bc.artist_id)
          -- integrity freezer: never submit a suspect artist to MB
          AND NOT EXISTS (SELECT 1 FROM review_item ri WHERE ri.subject_id = bc.artist_id
                          AND ri.reason IN ('source_coherence', 'ai_slop')
                          AND ri.status = 'pending')
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    from pipeline.slop_detect import gate_unevaluated

    gate_unevaluated(conn, [r[0] for r in rows])
    queued = 0
    for (aid,) in rows:
        if conn.execute(
            "SELECT 1 FROM review_item WHERE reason='ai_slop' AND status='pending' "
            "AND subject_id = %s", (aid,),
        ).fetchone():
            continue  # flagged in THIS cycle — never reaches the commons
        conn.execute(
            "INSERT INTO mb_submission (artist_id, payload, status) VALUES (%s, %s, 'spot_check')",
            (aid, json.dumps(build_payload(conn, aid))),
        )
        queued += 1
    return queued


def eligible_tag_rows(conn: Connection, target: str, limit: int) -> list[tuple]:
    """Artists to contribute tags for, with their CLEAN merged tag list.

    Contribution-quality gate (the test rehearsal on 2026-06-17 caught the old
    code submitting garbage): unlike publish, a contribution must NEVER include a
    guess — so audio tags are filtered to POSITIVE, non-NaN, non-magnet-blocklist
    scores (no min-1 floor), Bandcamp human tags lead, and an artist with nothing
    clean to add is SKIPPED entirely (not submitted, not marked — retried later if
    it earns clean tags). Returns [(artist_id, mbid, [tags<=5])] for non-empty
    contributions only. Pure read — testable without the network."""
    model_row = conn.execute("SELECT model FROM artist_tag_scores LIMIT 1").fetchone()
    model = model_row[0] if model_row else None
    rows = conn.execute(
        """
        SELECT a.id, a.mbid::text,
               coalesce((SELECT array_agg(s.tag ORDER BY s.score DESC) FROM (
                   SELECT ats.tag, ats.score FROM artist_tag_scores ats
                   WHERE ats.artist_id = a.id AND ats.model = %(model)s
                     AND ats.score > 0 AND ats.score != 'NaN'::real
                     AND NOT EXISTS (SELECT 1 FROM tag_audio_blocklist bl WHERE bl.tag = ats.tag)
                   ORDER BY ats.score DESC LIMIT 5) s), '{}') AS audio_tags,
               coalesce((SELECT array_agg(DISTINCT lower(bt))
                         FROM bc_candidate bc, unnest(bc.tags) bt
                         WHERE bc.artist_id = a.id AND bt IS NOT NULL AND bt <> ''),
                        '{}') AS bc_tags
        FROM artist a
        WHERE a.mbid IS NOT NULL AND a.embedding_source IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM mb_tag_submission ts
                          WHERE ts.artist_id = a.id AND ts.target = %(target)s)
          AND ( EXISTS (SELECT 1 FROM artist_tag_scores ats
                        WHERE ats.artist_id = a.id AND ats.model = %(model)s
                          AND ats.score > 0 AND ats.score != 'NaN'::real
                          AND NOT EXISTS (SELECT 1 FROM tag_audio_blocklist bl WHERE bl.tag = ats.tag))
             OR EXISTS (SELECT 1 FROM bc_candidate bc, unnest(bc.tags) bt
                        WHERE bc.artist_id = a.id AND bt IS NOT NULL AND bt <> '') )
        ORDER BY a.id LIMIT %(limit)s
        """,
        {"model": model, "target": target, "limit": limit},
    ).fetchall()
    out: list[tuple] = []
    for aid, mbid, audio_tags, bc_tags in rows:
        # Bandcamp human tags lead (higher quality), audio fills the rest;
        # dedup case-insensitively, cap at 5.
        merged, seen = [], set()
        for t in [*(bc_tags or []), *(audio_tags or [])]:
            if t and t.lower() not in seen:
                seen.add(t.lower())
                merged.append(t)
        if merged:
            out.append((aid, mbid, merged[:5]))
    return out


def submit_tags(conn: Connection, limit: int = 20, *, target: str = "live") -> int:
    """Phase 2 (LIVE API): upvote our CLEAN genre tags on artists that HAVE mbids
    — the contribution we can make today, fully programmatic. See
    eligible_tag_rows for the contribution-quality gate."""
    import time
    import xml.sax.saxutils as sx

    rows = eligible_tag_rows(conn, target, limit)
    if not rows:
        return 0
    tok = access_token(conn, target=target)
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<metadata xmlns="http://musicbrainz.org/ns/mmd-2.0#"><artist-list>']
    for _aid, mbid, merged in rows:
        parts.append(f'<artist id="{mbid}"><user-tag-list>')
        parts.extend(f"<user-tag><name>{sx.escape(t)}</name></user-tag>" for t in merged)
        parts.append("</user-tag-list></artist>")
    parts.append("</artist-list></metadata>")
    req = urllib.request.Request(
        f"{base_for(target)}/ws/2/tag?client=crates.ltd-0.1",
        data="".join(parts).encode(),
        headers={"User-Agent": UA, "Content-Type": "application/xml; charset=utf-8",
                 "Authorization": f"Bearer {tok}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"MB tag submission failed ({e.code}): {e.read()[:200]!r}") from e
    for aid, _mbid, _merged in rows:
        conn.execute(
            "INSERT INTO mb_tag_submission (artist_id, target) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (aid, target))
    time.sleep(1.1)  # bot-polite even in batch mode
    return len(rows)


def main() -> None:
    import argparse

    import psycopg

    from pipeline.config import Settings  # noqa: F401 — used in both branches

    ap = argparse.ArgumentParser(description="MB contribution lane (ADR-019)")
    ap.add_argument("--target", choices=("test", "live"), default="test",
                    help="MB server: test rehearses on test.musicbrainz.org (DEFAULT); "
                         "live touches the real commons and must be explicit")
    ap.add_argument("--consent", action="store_true", help="print the one-time consent URL")
    ap.add_argument("--code", help="exchange the consent code for a refresh token")
    ap.add_argument("--queue", type=int, default=0, help="queue N eligible artists for spot-check")
    ap.add_argument("--submit-tags", type=int, default=0, help="submit tags for N mbid artists")
    ap.add_argument("--create-artists", type=int, default=0,
                    help="phase-1b: create N staged artists via the edit system "
                         "(live: approved payloads only; test: spot_check ok)")
    ap.add_argument("--approve", help="bless a spot_check submission id for live")
    import sys

    argv = [a for i, a in enumerate(sys.argv[1:]) if not (a == "--" and i == 0)]
    args = ap.parse_args(argv)
    if args.consent:
        ru = _redirect_uri()
        host = urllib.parse.urlsplit(ru)
        if host.scheme in ("http", "https") and host.hostname in ("localhost", "127.0.0.1"):
            code = catch_code_locally(host.port or 80)
            with psycopg.connect(Settings().database_url) as conn:
                exchange_code(conn, code, target=args.target)
                conn.commit()
            print("refresh token stored — the lane is armed")
        else:
            print(consent_url())
        return
    with psycopg.connect(Settings().database_url) as conn:
        if args.code:
            exchange_code(conn, args.code, target=args.target)
            print(f"refresh token stored for target {args.target}")
        if args.queue:
            print(f"queued for spot-check: {queue_eligible(conn, args.queue)}")
            conn.commit()  # per-phase commit: a tag-lane crash must not roll back queue work
        if args.approve:
            n = conn.execute(
                "UPDATE mb_submission SET status='approved' WHERE id=%s AND status='spot_check'",
                (args.approve,)).rowcount
            print(f"approved: {n}")
        if args.submit_tags:
            print(f"[{args.target}] tag submissions sent: "
                  f"{submit_tags(conn, args.submit_tags, target=args.target)}")
        if args.create_artists:
            from pipeline.mb_artist_create import submit_artists

            print(f"[{args.target}] artist creation: "
                  f"{submit_artists(conn, target=args.target, limit=args.create_artists)}")
        conn.commit()


if __name__ == "__main__":
    main()
