"""End-to-end incremental MB sync (poe mb-sync): the acquisition front half
of ADR-018's refresh.

LATEST serial → skip if already applied (mb_refresh_run.serial) → download
mbdump.tar.bz2 + mbdump-derived.tar.bz2 + MD5SUMS DIRECT (deliberately no
proxy: MetaBrainz's CDN is not crawl traffic, and the proxy's bandwidth
belongs to prep) → md5-verify → extract ONLY our 11 tables (artist_tag/tag
live in the DERIVED archive — bootstrap-era lesson) → run_refresh (dry-run
unless --apply, per the ADR's three-clean-cycles discipline).

Run in a network-quiet window: ~10GB of download shares the box's pipe with
the mass build even without the proxy.
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

from psycopg import Connection

BASE = "https://data.metabrainz.org/pub/musicbrainz/data/fullexport"
CORE_TABLES = ["artist", "artist_alias", "url", "l_artist_url", "link", "link_type",
               "genre", "genre_alias", "artist_gid_redirect"]
DERIVED_TABLES = ["artist_tag", "tag"]
_CHUNK = 1 << 20


def latest_serial() -> str:
    with urllib.request.urlopen(f"{BASE}/LATEST", timeout=30) as r:
        return r.read().decode().strip()


def already_applied(conn: Connection, serial: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM mb_refresh_run WHERE serial = %s AND applied_at IS NOT NULL", (serial,)
    ).fetchone()
    return row is not None


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(_CHUNK):
            f.write(chunk)


def verify_md5(archive: Path, md5sums: Path) -> None:
    """Fail-closed: a corrupt download must abort before extraction."""
    want = None
    for line in md5sums.read_text().splitlines():
        if line.strip().endswith(archive.name):
            want = line.split()[0]
            break
    if want is None:
        raise RuntimeError(f"{archive.name} not present in MD5SUMS")
    h = hashlib.md5()
    with open(archive, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    if h.hexdigest() != want:
        raise RuntimeError(f"md5 mismatch for {archive.name} — corrupt download, aborting")


def extract_tables(archive: Path, tables: list[str], out_dir: Path) -> None:
    members = [f"mbdump/{t}" for t in tables]
    with tarfile.open(archive, "r:bz2") as tar:
        for m in members:
            ti = tar.getmember(m)
            ti.name = Path(m).name  # flatten into out_dir
            tar.extract(ti, out_dir, filter="data")


def sync(conn: Connection, work_dir: Path, *, apply: bool, force: bool = False) -> dict:
    from pipeline.mb_refresh import run_refresh

    serial = latest_serial()
    if not force and already_applied(conn, serial):
        return {"serial": serial, "skipped": "already applied"}
    work_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = work_dir / serial
    dump_dir.mkdir(exist_ok=True)
    md5 = work_dir / "MD5SUMS"
    _download(f"{BASE}/{serial}/MD5SUMS", md5)
    for archive_name, tables in (("mbdump.tar.bz2", CORE_TABLES),
                                 ("mbdump-derived.tar.bz2", DERIVED_TABLES)):
        archive = work_dir / archive_name
        if not archive.exists():  # resumable across reruns of the same serial
            _download(f"{BASE}/{serial}/{archive_name}", archive)
        verify_md5(archive, md5)
        extract_tables(archive, tables, dump_dir)
    report = run_refresh(conn, dump_dir, apply=apply)
    conn.execute(
        "UPDATE mb_refresh_run SET serial = %s WHERE id = (SELECT max(id) FROM mb_refresh_run)",
        (serial,),
    )
    report["serial"] = serial
    return report


def main() -> None:
    import argparse
    import json

    import psycopg

    from pipeline.config import Settings

    ap = argparse.ArgumentParser(description="incremental MB sync (download + refresh; dry-run default)")
    ap.add_argument("--work-dir", default="/tmp/mb-sync")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run an already-applied serial")
    args = ap.parse_args()
    with psycopg.connect(Settings().database_url) as conn:
        report = sync(conn, Path(args.work_dir), apply=args.apply, force=args.force)
        conn.commit()
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
