# Pipeline workflows — as built

Diagrams of what actually runs (ADR-016/ADR-017). Maintained per-slice: if a
slice changes a flow, its commit updates the diagram. Dashed elements are
designed-but-not-built; everything solid has run for real against the factory
DB. Verified live 2026-06-09: Burial (deezer 6281) — 12 tracks discovered,
12 MuQ clips embedded, centroid committed.

## IngestArtistWorkflow — the audio-source cascade

One workflow per ARTIST, id `ingest-artist-{artist_id}` (deterministic →
re-seeding is idempotent; scan verdicts make re-runs cheap). The cascade walks
the artist's audio-role identities in signal-priority order and embeds from
exactly one source — centroid purity is enforced in SQL, not by convention.

What each step actually does:

1. **Plan** — "which of this artist's pages can carry audio, and which still
   need scanning?" Local DB read over MB-derived identities. Tidal/Apple/Qobuz
   are playback assets and never enter the cascade. No audio identities at all
   → `unbound`; **nothing is crawled, searched, or guessed** (crawler binding
   stays design-gated).
2. **Cascade scan** — for each pending source in priority order (deezer →
   bandcamp → soundcloud → youtube): discover its tracks on the platform's
   rate-capped queue, write the TERMINAL scan verdict (`scanned`/`empty` —
   transient errors leave `pending` for retry), and **stop early when a source
   meets its floor** (deezer 10 · bc/sc 3 · yt experimental). Sources without
   an ingestion flow yet are skipped and stay pending.
3. **Choose** — floors double as equal-signal normalizers: first source meeting
   its floor (priority order) wins; if none does, the best THIN source wins by
   floor-ratio (2 BC tracks at 2/3 beat 1 Deezer preview at 1/10). Nothing
   anywhere → `no_signal`.
4. **Embed, source-locked** — download the winner's audio (self-healing URL
   refresh on 403), MuQ on the GPU queue, stamped clips, centroid built ONLY
   from the winning source's clips, `signal_ratio` recorded for downstream
   gating, `artist.embedding_source` locked. A later richer source supersedes
   by re-running this — the centroid flips wholesale, never blends.

```mermaid
flowchart TD
    START(["seeded per ARTIST from the Tier-A pool"]) --> PLAN

    subgraph PQ ["task queue: pipeline"]
        PLAN["1 · Which pages can carry audio,<br/>which still need scanning?<br/>(playback-only platforms excluded)"]
        VERDICT["2b · Record terminal scan verdict<br/>scanned / empty · errors stay pending"]
        CHOOSE["3 · Pick the embedding source<br/>floor met (priority order), else<br/>best floor-ratio thin source"]
    end

    PLAN -->|"no audio identities"| UNBOUND(["unbound — never guessed<br/>(crawler binding: design-gated)"])
    PLAN --> LOOP{next pending source<br/>in priority order?}

    subgraph DQ ["per-platform IO queues · server-rate-capped"]
        DDISC["2a · Discover the artist's tracks<br/>(deezer-io 10/s live; bc/sc/yt later)<br/>own tracks only · fetch-cached"]
    end

    LOOP -->|"has ingestion flow"| DDISC
    LOOP -->|"no flow yet — skip,<br/>stays pending"| LOOP
    DDISC --> VERDICT
    VERDICT -->|"floor met → stop early"| CHOOSE
    VERDICT -->|"under floor → continue"| LOOP
    LOOP -->|exhausted| CHOOSE
    CHOOSE -->|"nothing usable"| NOSIG(["no_signal — scanned, recorded,<br/>nothing worth embedding"])
    CHOOSE -->|"winner + signal_ratio"| EMBED

    subgraph GQ ["task queue: gpu · concurrency-capped"]
        EMBED["4 · Embed source-locked<br/>self-healing URL refresh → MuQ →<br/>pure centroid + signal_ratio"]
    end

    EMBED --> DONE(["embedded — one source, ratio recorded,<br/>supersede-ready when richer sources land"])
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
| MB bootstrap, Tier-A bind/classify, deezer-io + bandcamp-io discovery, fetch cache, windowed embed path (RMS peaks), Wave-1 analysis heads (fingerprint/MIR/integrity flags/MuLan tags, decode-once), centroids, cascade, seeder | Publish workflow, admin review wiring, B-tier search (3d), SC/YT discovery, Tidal trickle, Wave-2/3 heads, tag-score calibration |

**Explicitly not built and design-gated: crawler/search-based artist discovery.**
Binding artists without an MB url-rel (platform search, evidence scoring,
triangulation — Tier B1/B2/C) requires its own investigation + empirical
testing cycle before any implementation, and the 1k-artist calibration run
before it feeds the corpus (ADR-017 §3). Until then, no-proof artists end
`unbound` — by design, not omission.
