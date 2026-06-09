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

```mermaid
flowchart TD
    START(["start_workflow<br/>id = ingest-{platform}-{platform_id}"]) --> CLASSIFY

    subgraph PQ ["task queue: pipeline"]
        CLASSIFY["classify_page(platform, platform_id)<br/>DB-truth: platform_identity.page_type"]
        BIND["bind_source(artist, platform, platform_id)<br/>Tier-A from MB url-rel evidence"]
        EMBED["embed_artist(artist_id)<br/>MuQ via registry · stamped clips · centroid"]
    end

    CLASSIFY --> BIND
    BIND -->|"None (no authoritative link)"| UNBOUND(["status: unbound<br/>(B-tier search = slice 3d)"])
    BIND -->|"tier C"| PARK["⏸ wait_condition<br/>parked, crash-safe"]
    PARK -->|"signal: submit_review_decision"| DECIDE{decision}
    DECIDE -->|rejected| REJECTED(["status: rejected_by_review"])
    DECIDE -->|approved| DISC
    BIND -->|"tier A / B1 / B2"| DISC{platform in<br/>DISCOVERY_ACTIVITIES?}

    subgraph DQ ["task queue: deezer-io · server-capped 10/s"]
        DDISC["discover_deezer_tracks(artist_id)<br/>top-12 → albums fallback<br/>source-correctness: main artist only"]
    end

    DISC -->|deezer| DDISC
    DISC -->|"other (no discovery yet)"| EMBED
    DDISC -->|"discovered: n"| EMBED
    EMBED --> DONE(["status: embedded<br/>{tier, page_type, discovered, embedded}"])
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
