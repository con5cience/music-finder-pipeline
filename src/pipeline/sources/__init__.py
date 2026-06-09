"""Per-platform discovery modules (ADR-017): each turns a bound identity into
audio_track rows under the source-correctness law. IO goes through the fetch
cache; activities register on the platform's rate-capped task queue."""
