# Pipeline workflows — as built

Diagrams of what actually runs (ADR-016/ADR-017). Maintained per-slice: if a
slice changes a flow, its commit updates the diagram. Dashed elements are
designed-but-not-built; everything solid has run for real against the factory
DB. Verified live 2026-06-09: Burial (deezer 6281) — 12 tracks discovered,
12 MuQ clips embedded, centroid committed.

## IngestArtistWorkflow — per-artist orchestration

One workflow per (artist × platform identity), id `ingest-{platform}-{platform_id}`
(deterministic → seeding is idempotent). The Tier-C park is the reason Temporal
exists here: a parked workflow survives crashes for as long as review takes.

What each step actually does:

1. **Classify** — "do we already know what kind of page this is?" A local DB
   read: MB-derived pages were classified at bootstrap (artist/label/etc.).
   No network. Unknown pages stay unknown until a future slice classifies live.
2. **Bind** — "can we PROVE this page belongs to this artist?" Today the only
   accepted proof is a MusicBrainz url-rel (→ Tier A). No proof → the workflow
   ends `unbound` and **nothing is crawled, searched, or guessed**. Search/
   crawler-based binding (Tier B/C evidence scoring) does not exist yet — it is
   design-gated: thorough investigation + empirical testing before any code,
   then the 1k-artist calibration before it touches the corpus (ADR-017 §3).
3. **Review park (Tier C)** — when bind one day yields ambiguous evidence, the
   workflow freezes here, crash-safe, until a human signals a verdict. Wired
   but unreachable today (nothing produces Tier C yet).
4. **Discover** — "ask the platform what this artist's tracks are." Runs on the
   platform's own rate-capped queue through the fetch cache. Deezer only, so
   far: top-12 previews, albums fallback, main-artist tracks only.
5. **Embed** — "turn the tracks into corpus vectors." Download previews, run
   MuQ on the GPU, store model-stamped clip vectors, refresh the artist's
   centroid. The artist is now in the similarity space.

```mermaid
flowchart TD
    START(["seeded from the Tier-A pool<br/>one run per artist × platform page"]) --> CLASSIFY

    subgraph PQ ["task queue: pipeline"]
        CLASSIFY["1 · What kind of page is this?<br/>local lookup — bootstrap already<br/>classified MB-derived pages"]
        BIND["2 · Can we prove this page is this artist?<br/>only proof today: a MusicBrainz url-rel → Tier A<br/>no proof → stop; we never guess"]
        EMBED["5 · Make the artist searchable<br/>download previews → MuQ on GPU →<br/>stamped vectors + artist centroid"]
    end

    CLASSIFY --> BIND
    BIND -->|"no authoritative link"| UNBOUND(["unbound — artist skipped<br/>crawler/search binding: not built,<br/>design-gated (investigate first)"])
    BIND -->|"ambiguous evidence (Tier C)<br/>unreachable today"| PARK["3 · ⏸ frozen for human review<br/>crash-safe, indefinitely"]
    PARK -->|"reviewer verdict"| DECIDE{approved?}
    DECIDE -->|no| REJECTED(["rejected — never embedded"])
    DECIDE -->|yes| DISC
    BIND -->|"proven (Tier A)"| DISC{does this platform have<br/>track discovery built?}

    subgraph DQ ["task queue: deezer-io · server-capped 10/s"]
        DDISC["4 · Ask Deezer for the artist's tracks<br/>top-12 previews, albums fallback,<br/>this artist's own tracks only, cached"]
    end

    DISC -->|deezer| DDISC
    DISC -->|"not yet (bandcamp/sc/yt later)"| EMBED
    DDISC --> EMBED
    EMBED --> DONE(["embedded — in the corpus<br/>with tier + provenance recorded"])
```

## System data flow — bootstrap to corpus

```mermaid
flowchart LR
    subgraph MB ["MusicBrainz (twice-weekly fullexport)"]
        DUMP["mbdump.tar.bz2 · 7GB<br/>+ mbdump-derived.tar.bz2<br/>(artist_tag/tag live HERE)"]
    end

    DUMP -->|"tar extract · 8 tables<br/>MD5-verified"| BOOT["poe mb-bootstrap<br/>COPY + schema-drift guard"]
    BOOT --> MBRAW[("mb_raw schema<br/>2.89M artists · 19.9M urls<br/>6.15M artist-url rels")]
    MBRAW -->|"derive: host-pattern match<br/>ended rels skipped"| IDS[("artist 543,271<br/>platform_identity 1,025,797<br/>Tier-A pool")]

    IDS --> SEED["seed_ingest CLI<br/>deterministic workflow ids"]
    SEED --> TEMPORAL["Temporal server<br/>IngestArtistWorkflow per (artist × platform)"]

    subgraph FLEET ["worker fleet (one process)"]
        W0["pipeline queue<br/>workflows + classify/bind/embed"]
        W1["deezer-io · 10/s server-enforced"]
        W2["bandcamp-io · 5/s (next slice)"]
        W3["soundcloud-io · 5/s · tidal-io 0.2/s<br/>youtube-io 0.1/s · musicbrainz-io 1/s"]
    end

    TEMPORAL --> FLEET
    W1 -->|GET api.deezer.com| CACHE["fetch_cache<br/>gzip blobs + DB index<br/>NEVER refetch · 404 = negative"]
    CACHE --> TRACKS[("audio_track<br/>tier A · verified<br/>preview urls · evidence jsonb")]
    TRACKS --> EMBEDP["embed path<br/>(next diagram)"]
    EMBEDP --> CLIPS[("clip_embedding<br/>model-stamped · 1024-d")]
    EMBEDP --> CENT[("artist_embedding<br/>l2-normalized centroid")]

    CENT -.->|"publish workflow (future)<br/>batch upsert + watermark"| APP[("app DB<br/>local dev → cloud")]
    TEMPORAL -.->|"Tier C parks"| REVIEW["admin panel queue (future)<br/>decision table + poller → signal"]
```

## Inside embed_artist

```mermaid
sequenceDiagram
    participant WF as IngestArtistWorkflow
    participant ACT as embed_artist (pipeline queue)
    participant DB as factory Postgres :5440
    participant CDN as Deezer CDN
    participant MUQ as MuQ (CUDA, lazy singleton)

    WF->>ACT: embed_artist(artist_id)
    ACT->>DB: pending_tracks(artist, model)<br/>has audio_url · not rejected/quarantined<br/>no row for (track, 0, model) — idempotent
    DB-->>ACT: [(track, preview_url, 30s), ...]
    loop each track
        ACT->>CDN: GET preview (real UA, status-checked)
        CDN-->>ACT: bytes → sniff magic → name .mp3<br/>(libsndfile mp3 detect is extension-gated)
    end
    ACT->>MUQ: embed(clips) — batch 8, L2-normalized
    MUQ-->>ACT: vectors (1024-d)
    ACT->>DB: INSERT clip_embedding<br/>(track, segment, MODEL STAMP, dim, vector)
    ACT->>DB: UPSERT artist_embedding<br/>l2_normalize(avg(clips)) per (artist, model)
    Note over ACT: temp audio deleted — never archived (ADR-017 §5)
    ACT-->>WF: embedded count
```

## Status legend

| Built + verified live | Designed, not built (dashed) |
|---|---|
| MB bootstrap, Tier-A bind/classify, deezer-io discovery, fetch cache, embed path, centroids, seeder | Publish workflow, admin review wiring, B-tier search (3d), Bandcamp/SC/YT discovery, Tidal trickle |

**Explicitly not built and design-gated: crawler/search-based artist discovery.**
Binding artists without an MB url-rel (platform search, evidence scoring,
triangulation — Tier B1/B2/C) requires its own investigation + empirical
testing cycle before any implementation, and the 1k-artist calibration run
before it feeds the corpus (ADR-017 §3). Until then, no-proof artists end
`unbound` — by design, not omission.
