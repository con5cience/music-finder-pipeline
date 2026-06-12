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
