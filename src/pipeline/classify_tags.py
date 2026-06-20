"""Auto-classify the tag vocabulary for the genre-only curation policy (#35
follow-up). Decides each UNDECIDED tag (not already in tag_approved /
tag_manual_blocklist) and writes the verdict with source='auto', so the admin
Tags tab is pre-sorted and a human only reviews the ambiguous residual.

Policy: KEEP genres/styles; BLOCK everything else (location, mood/descriptor,
instrument, label, meta). Three stages, cheap → expensive:

  1. MB genre vocab  — tag is a canonical MB genre/alias ⇒ genre ⇒ approve.
  2. Heuristics      — CONSERVATIVE patterns that only fire on CLEAR non-genres
                       (locations, label suffixes, instruments, meta/mood) ⇒ block.
                       Deliberately narrow so it never blocks a real genre; the
                       ambiguous middle (real long-tail subgenres MB lacks, e.g.
                       'dungeon synth') falls through.
  3. LLM (optional)  — the residual is classified genre/nongenre by an LLM when
                       TAG_CLASSIFIER_LLM=anthropic + ANTHROPIC_API_KEY are set;
                       otherwise the residual is LEFT UNDECIDED for the human tab.

Human decisions (source='human') are never touched. Idempotent: re-runs only act
on still-undecided tags.

  poe classify-tags            # heuristics + MB vocab only (no LLM)
  poe classify-tags --llm      # also classify the residual via the LLM lane
  poe classify-tags --limit N  # cap how many undecided tags to process
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request

import psycopg

from pipeline.config import Settings

# --- Heuristic vocabularies (conservative: only CLEAR non-genres) -------------

# Countries + a spread of music cities/regions that recur as Bandcamp "genre"
# tags. Bare place name = location → block. (Compound genres like 'chicago house'
# are NOT bare place names and won't match.)
LOCATIONS: frozenset[str] = frozenset(
    {
        "usa", "u.s.a.", "us", "united states", "uk", "u.k.", "united kingdom", "england", "scotland",
        "wales", "ireland", "canada", "australia", "new zealand", "germany", "france", "italy", "spain",
        "portugal", "netherlands", "belgium", "sweden", "norway", "denmark", "finland", "iceland",
        "poland", "russia", "ukraine", "japan", "china", "korea", "south korea", "brazil", "argentina",
        "chile", "mexico", "colombia", "india", "indonesia", "greece", "turkey", "austria", "switzerland",
        "czech republic", "hungary", "romania", "south africa",
        # cities / regions
        "london", "manchester", "bristol", "glasgow", "berlin", "hamburg", "cologne", "paris", "lyon",
        "amsterdam", "rotterdam", "stockholm", "oslo", "copenhagen", "helsinki", "reykjavik", "dublin",
        "lisbon", "madrid", "barcelona", "rome", "milan", "vienna", "zurich", "tokyo", "osaka", "kyoto",
        "seoul", "beijing", "shanghai", "sydney", "melbourne", "auckland", "toronto", "montreal",
        "vancouver", "new york", "new york city", "nyc", "brooklyn", "los angeles", "la", "san francisco",
        "bay area", "oakland", "seattle", "portland", "chicago", "detroit", "atlanta", "austin",
        "nashville", "new orleans", "boston", "philadelphia", "washington dc", "denver", "minneapolis",
        "cdmx", "mexico city", "sao paulo", "rio de janeiro", "buenos aires", "bogota",
    }
)

INSTRUMENTS: frozenset[str] = frozenset(
    {
        "piano", "guitar", "acoustic guitar", "electric guitar", "drums", "saxophone", "sax", "violin",
        "cello", "viola", "flute", "trumpet", "trombone", "clarinet", "accordion", "banjo", "harp",
        "ukulele", "mandolin", "harmonica", "organ", "double bass", "percussion",
    }
)

# Mood / descriptor / meta — clear non-genres. (Genre-adjacent words like
# 'ambient', 'bass', 'synth', 'lo-fi' are intentionally NOT here — they're real
# genres/styles; leave them to MB-vocab or the LLM.)
META: frozenset[str] = frozenset(
    {
        "female vocals", "male vocals", "female vocalist", "male vocalist", "female fronted", "vocals",
        "vocal", "instrumental", "seen live", "favorites", "favourites", "diy", "vinyl", "cassette",
        "cassette tape", "demo", "compilation", "live", "ep", "lp", "album", "single", "mixtape", "split",
        "self released", "self-released", "free download", "free", "all", "various", "various artists",
        "soundtrack", "score", "atmospheric", "chill", "chilled", "dark", "dreamy", "melancholic",
        "melancholy", "energetic", "cinematic", "ethereal", "moody", "uplifting", "relaxing", "sad",
        "happy", "epic", "experimental music", "good", "best", "new", "fun", "cool",
    }
)

_LABEL_SUFFIX = re.compile(r"\b(records|recordings|tapes|tape|label|productions|recs)$")
_NUMERIC = re.compile(r"^[\d\s.,'\-]+$")  # year-only / number-only tags


def heuristic_category(tag: str) -> str | None:
    """Return a non-genre category to BLOCK, or None if not a clear non-genre.
    Never returns 'genre' — keeps are decided by MB-vocab or the LLM."""
    t = tag.strip().lower()
    if not t:
        return None
    if t in LOCATIONS:
        return "location"
    if t in INSTRUMENTS:
        return "instrument"
    if t in META:
        return "meta"
    if _LABEL_SUFFIX.search(t):
        return "label"
    if _NUMERIC.match(t):
        return "meta"
    return None


def mb_genre_vocab(conn: psycopg.Connection) -> frozenset[str]:
    """All canonical MB genre names + aliases (lowercased) — the "definitely a
    genre" allowlist."""
    rows = conn.execute(
        "SELECT lower(name) FROM mb_raw.genre UNION SELECT lower(name) FROM mb_raw.genre_alias"
    ).fetchall()
    return frozenset(r[0] for r in rows)


# --- LLM residual classifier (optional) ---------------------------------------

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
# Small/fast model; override via TAG_CLASSIFIER_MODEL. (Provider/model is the one
# open decision — see module docstring.)
DEFAULT_MODEL = os.environ.get("TAG_CLASSIFIER_MODEL", "claude-haiku-4-5-20251001")


def llm_enabled() -> bool:
    return os.environ.get("TAG_CLASSIFIER_LLM") == "anthropic" and bool(os.environ.get("ANTHROPIC_API_KEY"))


def llm_classify(tags: list[str], *, batch: int = 60) -> dict[str, str]:
    """Classify each tag as 'genre' or 'nongenre' via the Anthropic Messages API
    (stdlib urllib — no SDK dependency). Returns {tag: 'genre'|'nongenre'};
    omits any tag the model didn't return (treated as unsure → left undecided).
    No-op returning {} unless TAG_CLASSIFIER_LLM=anthropic + ANTHROPIC_API_KEY."""
    if not (tags and llm_enabled()):
        return {}
    key = os.environ["ANTHROPIC_API_KEY"]
    out: dict[str, str] = {}
    for i in range(0, len(tags), batch):
        chunk = tags[i : i + batch]
        prompt = (
            "You label music tags for a discovery app that wants ONLY genre/style/"
            "scene tags. For each tag, answer 'genre' if it is a music genre, style, "
            "or scene; otherwise 'nongenre' (location, mood/descriptor, instrument, "
            "record label, person, or metadata). Reply ONLY with a JSON object "
            'mapping each tag to "genre" or "nongenre".\n\nTags:\n' + json.dumps(chunk)
        )
        body = json.dumps(
            {
                "model": DEFAULT_MODEL,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        req = urllib.request.Request(
            ANTHROPIC_URL,
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted host)
            payload = json.loads(resp.read())
        text = "".join(p.get("text", "") for p in payload.get("content", []))
        try:
            parsed = json.loads(text[text.index("{") : text.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            continue
        for tag, verdict in parsed.items():
            v = str(verdict).strip().lower()
            if v in ("genre", "nongenre"):
                out[tag.strip().lower()] = v
    return out


# --- Orchestration ------------------------------------------------------------

def undecided_tags(conn: psycopg.Connection, limit: int | None) -> list[str]:
    """Tags in the corpus (tag_review_freq) with no decision yet, highest-df first."""
    sql = (
        "SELECT f.tag FROM tag_review_freq f "
        "WHERE NOT EXISTS (SELECT 1 FROM tag_manual_blocklist b WHERE b.tag=f.tag) "
        "AND NOT EXISTS (SELECT 1 FROM tag_approved a WHERE a.tag=f.tag) "
        "ORDER BY f.df DESC, f.tag"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [r[0] for r in conn.execute(sql).fetchall()]


def _approve(conn: psycopg.Connection, tag: str, category: str) -> None:
    conn.execute(
        "INSERT INTO tag_approved (tag, category, source) VALUES (%s,%s,'auto') ON CONFLICT (tag) DO NOTHING",
        (tag, category),
    )


def _block(conn: psycopg.Connection, tag: str, category: str) -> None:
    conn.execute(
        "INSERT INTO tag_manual_blocklist (tag, reason, source, category) "
        "VALUES (%s,%s,'auto',%s) ON CONFLICT (tag) DO NOTHING",
        (tag, f"auto:{category}", category),
    )


def run_classify(conn: psycopg.Connection, *, limit: int | None = None, use_llm: bool = False) -> dict[str, int]:
    """Classify undecided tags against the CURRENT tag_review_freq snapshot.
    Returns counts per outcome. The caller refreshes the MV first (REFRESH
    CONCURRENTLY can't run inside a transaction, so it's not done here)."""
    vocab = mb_genre_vocab(conn)
    tags = undecided_tags(conn, limit)
    counts = {"genre_mb": 0, "block_heuristic": 0, "genre_llm": 0, "block_llm": 0, "undecided": 0}
    residual: list[str] = []

    for tag in tags:
        t = tag.strip().lower()
        if t in vocab:
            _approve(conn, t, "genre")
            counts["genre_mb"] += 1
            continue
        cat = heuristic_category(t)
        if cat:
            _block(conn, t, cat)
            counts["block_heuristic"] += 1
            continue
        residual.append(t)

    if use_llm and residual:
        verdicts = llm_classify(residual)
        for t in residual:
            v = verdicts.get(t)
            if v == "genre":
                _approve(conn, t, "genre")
                counts["genre_llm"] += 1
            elif v == "nongenre":
                _block(conn, t, "llm")
                counts["block_llm"] += 1
            else:
                counts["undecided"] += 1
    else:
        counts["undecided"] += len(residual)

    return counts  # caller owns the commit (keeps the fn transaction-agnostic)


def main() -> None:
    ap = argparse.ArgumentParser(description="auto-classify tags (genre-only policy)")
    ap.add_argument("--limit", type=int, default=None, help="cap undecided tags processed")
    ap.add_argument("--llm", action="store_true", help="classify the residual via the LLM lane")
    args = ap.parse_args()
    if args.llm and not llm_enabled():
        print("warning: --llm set but TAG_CLASSIFIER_LLM=anthropic + ANTHROPIC_API_KEY not configured; "
              "residual will be left undecided")
    with psycopg.connect(Settings().database_url) as conn:
        # Refresh the freq snapshot first so newly-discovered tags are seen.
        # CONCURRENTLY must be its own statement (not inside the classify tx).
        conn.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY tag_review_freq")
        conn.commit()
        counts = run_classify(conn, limit=args.limit, use_llm=args.llm)
        conn.commit()
    print("classified:", json.dumps(counts))


if __name__ == "__main__":
    main()
