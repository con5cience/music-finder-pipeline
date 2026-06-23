# ADR-018 — MusicBrainz periodic refresh (inbound MB → crates)

**Status:** Accepted. Validated end-to-end 2026-06-23 (dry-run *and* `--apply`).

## Context
Our corpus drifts from MusicBrainz over time: artists get renamed, duplicates
get merged, and new artists appear. We need a periodic inbound re-sync that
keeps canonical names/identities fresh and trickles in new MB artists — without
disrupting the live factory.

MusicBrainz publishes **full database exports ~twice weekly** at
`data.metabrainz.org/.../fullexport`. It does **not** publish incremental dump
tarballs — the only MB incrementals are hourly *replication packets* (a mirror
mechanism this pipeline does not consume). ListenBrainz is the project with
incremental dumps; MB is fullexport-only. So every refresh works from a full
snapshot; "incremental" here means the *derive-diff*, not an incremental download.

## Decision
`poe mb-sync` (acquisition) + `poe mb-refresh` (apply), both **dry-run by default**:

1. `mb_sync`: LATEST serial → skip if already applied (`mb_refresh_run`) →
   download `mbdump.tar.bz2` + `mbdump-derived.tar.bz2` + `MD5SUMS` (direct, no
   proxy) → md5-verify (fail-closed) → extract only our ~11 tables.
2. `mb_refresh`: shadow-load into `mb_raw_next` (live `mb_raw` keeps serving) →
   **fail-closed sanity gates** (next ≥ 0.98× current artists / 0.95× url-rels) →
   derive-diff via existing `derive_identities` (idempotent → only NEW lands) →
   renames → merges via `artist_gid_redirect` (**both-embedded conflicts → Tier-C
   review queue, never auto-picked**) → transactional schema swap (`mb_raw_old`
   kept one cycle). `--apply` is deliberate (three-clean-cycles discipline).

## Validation (2026-06-23)
Run on a **throwaway pgvector copy of the factory DB** (isolated; live never
touched): bootstrap `mb_raw` from the 06-06 fullexport (baseline) → refresh
against the 06-20 fullexport.

| Signal | Result |
|---|---|
| Sanity gates | PASS (artists 2,892,692 → 2,904,489; url-rels 6,149,898 → 6,185,499) |
| `--apply` artist delta | **+3,938** (545,599 → 549,537) |
| new identities | +11,818 |
| renames / merges / reviews | 100 / 28 / 2 |

**Conclusion:** the refresh is **maintenance/self-heal** (identity enrichment +
rename/merge reconciliation) plus a **~4k-artist/fortnight trickle** — not bulk
ingestion. New artist rows are mbid stubs gated by the GPU analysis admission
bar (the wave seeder embeds them over time). Corpus *growth* remains the
Bandcamp discovery path (ADR-019), not MB.

## Consequences / operational notes
- **`mb_raw` must be bootstrapped** (`poe mb-bootstrap`) for the genre vocabulary
  (`mb_raw.genre`); an empty `mb_raw` silently **skips the audio tag head**.
- The dry-run **`adds`** preview must count the *platform-matched* population the
  apply actually mints, not MB artists with *any* url — the latter over-reported
  by ~360× (1.43M vs 3,938). Fixed via the shared `matched_artist_url_cte`
  (con5cience/music-finder-pipeline#1).
- Each refresh re-downloads the **full ~8 GB export** (no MB incrementals) — run
  in a network-quiet window.
- **Not scheduled** today — manual `poe mb-sync`. Cadence automation is TBD.
- **Testing**: restore an isolated copy with `pg_dump --section=pre-data
  --section=data` (skip the post-data **HNSW index rebuild** — the dominant,
  avoidable cost; a full restore cost ~1h once), targeted tables, and a
  throwaway-tuned PG (`synchronous_commit=off`). Run via `PIPELINE_DATABASE_URL`
  override against the isolated instance.
