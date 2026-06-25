"""Deterministic junk-tag sweep over the WHOLE undecided queue (reaches the
~102k df=1 tail that hand-batching can't). NO fuzzy ML — every block is an exact
rule, and --dry-run (default) prints exactly what it would block, grouped by
reason, so a human reviews before anything writes.

Rules (a tag is blocked only if it is NOT in the MB genre vocab and NOT already
approved — those exemptions protect real genres):
  1. heuristic_category()  — the tested classifier patterns (location/instrument/
     meta/mood/era/label/numeric/city-ST).
  2. artist-name match     — the tag exactly equals a corpus artist's name (the
     scalable catcher for obscure singleton band-names: megadeth, periphery, ...).
  3. curated non-genre set — common nouns/topics/moods hand-identified during the
     manual passes (trains, mountain, murder, creative, ...).

Pattern candidates that should ALIAS rather than block (#hashtag, dotted
obfuscation) are reported separately, never auto-blocked.

  python scripts/sweep_junk_tags.py            # dry-run: report only
  python scripts/sweep_junk_tags.py --apply    # write blocks (source='ai')
"""

from __future__ import annotations

import argparse
import json
import re

import psycopg

from pipeline.classify_tags import heuristic_category
from pipeline.config import Settings

# Common non-genre words/topics/moods seen recurring in the manual passes. Kept to
# things that are clearly NOT a music genre in any reading.
CURATED_JUNK: frozenset[str] = frozenset({
    "trains","mountain","murder","creative","pain","sunset","truth","sick","hype","honest",
    "relationships","beauty","unique","whimsical","vibey","bouncy","cathartic","drinking","anger",
    "bummer","lonely","massive","lol","ominous","stars","twilight","everything","faith","fiction",
    "holy","intelligent","conspiracy","demons","dust","failure","frogs","animal","bullshit","change",
    "communism","socialism","soviet","leftist","activism","pride","autism","astronomy","body","youth",
    "wall","unity","virtual","variety","true","stories","spirit","solstice","trees","whiskey","corona",
    "escapism","garden","glitter","heroic","horse","killer","memory","paranoia","red","revenge",
    "revolutionary","tour","trouble","crime","consciousness","dank","downer","experiment","fetish","fox",
    "gamer","high","insomnia","landscapes","latinx","literary","mother","machines","musician","sky",
    "stomp","stoned","teenage","tiktok","toys","vikings","werewolf","woods","young","mushrooms","manga",
    "movies","religion","neon","production","installation","fawm","sass","singalong","bilingual","bootlegs",
    "community","hentai","idol","nanoloop","computergaze","trump","wrestling","hollywood","heartfelt",
    "sample packs","drinking music","headphone music","local music","scary music","dreamy music","heavy music",
    "drum machines","four track","soundfont","cover album","debut ep","vinyl release","cassettes","you","to",
    "in","a","etc","hop","lmms","laptop","iphone","ios","mega drive","game jam","game of thrones",
    "animal crossing","backrooms","end of the world","mental illness","mental health awareness","covid 19",
    "black lives matter","first nations","non-binary","female artist","female rapper","male","meta",
    "growls","chops","arpeggios","peak time","one take","new song","multigenre","real rap","version",
    "sound sculpture","static noise","tape hiss","summer music","architecture","archival","anniversary",
    "exploration","field","outer space","paranormal","producers","sample based","samplepack","split 7\"",
    "women empowerment","inspirational music","wizard","comfy","bright","sublime","portal",
    "free album","record store day","kbd","sad bastard music","video game music cover",
})

HASHTAG = re.compile(r"^#")
DOTTED = re.compile(r"^([a-z0-9]\.){2,}[a-z0-9]\.?$")  # e.l.e.c.t.r.o  (single chars + dots)
# Genre-keyword guard: never auto-block an artist-name match that LOOKS like a
# genre (a band named for a genre) — those go to human review, not the blocklist.
GENRE_KW = re.compile(
    r"(core|wave|metal|punk|house|techno|synth|gaze|grind|doom|jazz|folk|hop|ambient|drone|noise|pop|rock|beat|"
    r"bass|trap|emo|disco|funk|soul|blues|country|sludge|crust|industrial|dnb|garage|grunge|ska|dub|drill|grime|"
    r"step|vapor|glitch|gospel|cumbia|flamenco|salsa|fado|klezmer|reggae|dancehall|tango|chanson|raga)")

# Bare demonym/nationality as a tag = location-ish junk (the genre would be
# "<nationality> <genre>"; bare won't false-match those).
NATIONALITIES = frozenset({
    "american","british","canadian","australian","german","french","italian","spanish","portuguese","dutch",
    "belgian","swedish","norwegian","danish","finnish","icelandic","polish","russian","ukrainian","japanese",
    "chinese","korean","brazilian","argentine","argentinian","chilean","mexican","colombian","indian","indonesian",
    "greek","turkish","austrian","swiss","czech","hungarian","romanian","bulgarian","filipino","iranian","irish",
    "scottish","welsh","english","israeli","egyptian","nigerian","ethiopian","lithuanian","latvian","estonian",
    "slovak","croatian","serbian","catalan","basque","mongolian","vietnamese","thai","peruvian","venezuelan",
    "ecuadorian","cuban","jamaican","south african","south american","sudamerica","worldwide",
})
MEDIA = frozenset({
    "star trek","the legend of zelda","super mario","donkey kong","elden ring","elder scrolls","warhammer",
    "warhammer 40k","friendship is magic","brony music","hp lovecraft","harry potter","game of thrones","star wars",
    "minecraft","undertale","final fantasy","pokemon","sonic the hedgehog","zelda","mario","kirby","earthbound",
    "animal crossing","cyberpunk2077","skyrim","fallout","norse mythology","greek mythology","horror movies",
    "anime","manga","movies","video games","cartoons","comics","video game cover","video games music",
})
EXTRA_INSTRUMENTS = frozenset({
    "bassoon","keytar","duduk","charango","bandoneon","kantele","cajon","kazoo","cowbell","gongs","gong","shakuhachi",
    "nyckelharpa","zither","mbira","flugelhorn","clarinette","clarinet","oboe","piccolo","harmonium","sitar","tabla",
    "didgeridoo","theremin","melodica","glockenspiel","vibraphone","marimba","xylophone","timpani","harpsichord",
    "concertina","dulcimer","hammered dulcimer","chapman stick","baritone sax","tenor sax","alto sax","soprano sax",
    "double bass","upright bass","slap bass","bass solo","drum solo","violin solo","pipe organ","church organ",
    "hammond b3","wurlitzer","rhodes","mellotron","harp music","flute music","guitarra","violon","8 string","7 string",
})


def music_suffix_junk(tag: str, mbvocab: frozenset[str], approved: frozenset[str]) -> bool:
    """'<x> music' where x is NOT itself a genre → redundant/junk (icelandic music,
    harp music, dream music). '<genre> music' (soul music) is kept (x is a genre)."""
    if not tag.endswith(" music"):
        return False
    pre = tag[: -len(" music")].strip()
    return bool(pre) and pre not in mbvocab and pre not in approved


def main() -> None:
    ap = argparse.ArgumentParser(description="deterministic junk-tag sweep")
    ap.add_argument("--apply", action="store_true", help="write blocks (default: dry-run report only)")
    args = ap.parse_args()

    with psycopg.connect(Settings().database_url) as conn:
        undecided = [r[0] for r in conn.execute(
            "SELECT f.tag FROM tag_review_freq f "
            "WHERE NOT EXISTS (SELECT 1 FROM tag_manual_blocklist b WHERE b.tag=f.tag) "
            "AND NOT EXISTS (SELECT 1 FROM tag_approved a WHERE a.tag=f.tag)").fetchall()]
        mbvocab = frozenset(r[0] for r in conn.execute(
            "SELECT lower(name) FROM mb_raw.genre UNION SELECT lower(name) FROM mb_raw.genre_alias").fetchall())
        approved = frozenset(r[0] for r in conn.execute("SELECT tag FROM tag_approved").fetchall())
        artist_names = frozenset(r[0] for r in conn.execute(
            "SELECT DISTINCT lower(display_name) FROM artist WHERE display_name IS NOT NULL").fetchall())

    blocks: dict[str, list[str]] = {}
    review: dict[str, list[str]] = {}
    for tag in undecided:
        if tag in mbvocab:  # real genre per MB — never touch
            continue
        cat = heuristic_category(tag)
        if cat:
            blocks.setdefault(cat, []).append(tag)
            continue
        if HASHTAG.search(tag) or DOTTED.match(tag):
            review.setdefault("alias-candidate", []).append(tag)
            continue
        if tag in CURATED_JUNK:
            blocks.setdefault("curated", []).append(tag)
            continue
        if tag in NATIONALITIES:
            blocks.setdefault("nationality", []).append(tag)
            continue
        if tag in MEDIA:
            blocks.setdefault("media", []).append(tag)
            continue
        if tag in EXTRA_INSTRUMENTS:
            blocks.setdefault("instrument", []).append(tag)
            continue
        if music_suffix_junk(tag, mbvocab, approved):
            blocks.setdefault("music-suffix", []).append(tag)
            continue
        if tag in artist_names and not GENRE_KW.search(tag):
            blocks.setdefault("artist-name", []).append(tag)
            continue

    total = sum(len(v) for v in blocks.values())
    print(f"undecided={len(undecided)}  would-block={total}  alias-candidates={sum(len(v) for v in review.values())}")
    for reason in sorted(blocks, key=lambda r: -len(blocks[r])):
        print(f"  {reason:14} {len(blocks[reason])}")
    json.dump(blocks, open("/tmp/sweep_blocks.json", "w"))
    json.dump(review, open("/tmp/sweep_alias_candidates.json", "w"))
    print("\nfull lists -> /tmp/sweep_blocks.json , /tmp/sweep_alias_candidates.json")

    if args.apply:
        flat = sorted({t for v in blocks.values() for t in v})
        with psycopg.connect(Settings().database_url) as conn:
            conn.cursor().executemany(
                "INSERT INTO tag_manual_blocklist (tag, reason, source, category) "
                "VALUES (%s,'sweep','ai',%s) ON CONFLICT (tag) DO NOTHING",
                [(t, next(r for r in blocks if t in blocks[r])) for t in flat])
            conn.commit()
        print(f"APPLIED {len(flat)} blocks")


if __name__ == "__main__":
    main()
