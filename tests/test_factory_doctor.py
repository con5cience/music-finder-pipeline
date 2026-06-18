"""factory-doctor / maintenance-window coordination.

The doctor force-recreates workers after two zero-embed strikes; a
maintenance window legitimately stops worker-gpu for hours. Without a
shared lock the doctor boots the GPU worker INTO the window's VRAM
(near-miss caught 2026-06-11, ~70min before strike two). The lock is a
repo-root file: the window script creates/removes it, the doctor (repo
mounted ro at /workspace) stands down while it exists. These pins keep
the two scripts referring to the same path.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ".maintenance-window"


def test_window_script_creates_and_clears_the_lock():
    src = (ROOT / "scripts" / "maintenance-window.sh").read_text()
    assert f"touch {LOCK}" in src or f'touch "{LOCK}"' in src
    # cleared in restore_fleet so EVERY exit path (trap EXIT) removes it
    restore = src.split("restore_fleet()")[1].split("}")[0]
    assert LOCK in restore


def test_doctor_stands_down_while_lock_exists():
    src = (ROOT / "scripts" / "factory-doctor.sh").read_text()
    assert f"/workspace/{LOCK}" in src
    # the stand-down must also reset the strike counter — a window that
    # ends between checks must not inherit a pre-window strike
    standdown = src.split(LOCK)[1].split("fi")[0]
    assert "ZEROES=0" in standdown


def test_lock_is_gitignored():
    assert LOCK in (ROOT / ".gitignore").read_text()


def test_doctor_warms_up_after_recreate():
    # 2026-06-18: a force-recreate triggers a ~35min GPU cold-start (embedder +
    # XLM-Roberta tag heads reload). The doctor measured zero embeds during that
    # warmup and logged strike 1/2 against a worker that was loading normally —
    # one strike from recreating it mid-load and resetting the cold-start (a
    # restart loop). After recreating, it must defer the next judgment by a
    # warmup grace that exceeds the cold-start.
    src = (ROOT / "scripts" / "factory-doctor.sh").read_text()
    assert "WARMUP=" in src
    warmup = int(src.split("WARMUP=")[1].split()[0])
    assert warmup >= 2400  # > observed ~35min cold-start
    # the recreate (wedge) branch defers via the warmup grace, then re-loops
    wedge = src.split("force-recreate worker-gpu worker-io")[1]
    assert 'sleep "$WARMUP"' in wedge
    assert "continue" in wedge.split('sleep "$WARMUP"')[1][:40]


def test_doctor_boot_grace_covers_cold_start():
    # the pre-loop boot grace must also cover the cold-start, not the old 120s
    src = (ROOT / "scripts" / "factory-doctor.sh").read_text()
    assert "sleep 120" not in src
    assert 'sleep "$WARMUP"' in src


def test_doctor_branches_flood_vs_wedge():
    # 2026-06-15: a Temporal flood (not a wedged worker) stalled embeds, and the
    # recreate-only doctor flapped uselessly. The remedy must branch on the
    # running-workflow count.
    src = (ROOT / "scripts" / "factory-doctor.sh").read_text()
    assert "workflow count" in src        # detects the flood by counting runs
    assert "FLOOD" in src                  # threshold defined
    assert "stop seeder" in src           # flood remedy: halt the source
    assert "force-recreate worker-gpu worker-io" in src  # wedge remedy: recreate
